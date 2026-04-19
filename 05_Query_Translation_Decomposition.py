"""
================================================================================
Part 7: Decomposition（问题分解 / 子问题分解）
================================================================================
【核心问题】
用户问题很复杂，包含多个子任务或需要多步推理，单次检索无法覆盖全部信息。

【解决思路】
让 LLM 把复杂大问题拆成多个更小的子问题（sub-questions），然后分别处理：

  方法一：递归回答（Answer Recursively）
    → 串行处理子问题，上一个子问题的答案作为下一个子问题的上下文
    → 适用于子问题之间有依赖/递进关系的情况
      （如"先解释概念A，再解释概念B，最后比较A和B的异同"）

  方法二：独立回答后汇总（Answer Individually）
    → 每个子问题独立检索、独立回答
    → 最后把所有子问题的答案汇总，作为上下文回答原始总问题
    → 适用于子问题之间相对独立的情况
      （如"LLM Agent 有哪些组件？每个组件分别做什么？"）

【与 03 Multi Query 的核心区别】
  03 Multi Query：不改问题粒度，只是"换种说法问同一个问题"
  05 Decomposition：改变问题粒度，把"一个大问题"拆成"多个不同的小问题"

  03 的合并对象是【文档】→ 扩大检索覆盖面
  05 的合并对象是【答案】→ 每个子问题都有独立的检索+生成链

【两种子问题处理方式的对比】
  ┌─────────────┬──────────────────────────┬──────────────────────────┐
  │   维度      │   递归回答               │   独立回答               │
  ├─────────────┼──────────────────────────┼──────────────────────────┤
  │ 执行方式    │  串行                    │  串行（可优化为并发）    │
  │ 上下文传递  │  前序答案传给后续子问题   │  无传递，各自独立         │
  │ 适用场景    │  子问题有依赖/递进关系    │  子问题相互独立           │
  │ LLM调用次数 │  n次子生成               │  n次子生成 + 1次汇总生成  │
  │ 最终输出    │  最后一份答案即最终结果   │  汇总所有子答案再生成一次 │
  └─────────────┴──────────────────────────┴──────────────────────────┘

【数据流示意 —— 递归回答】
  总问题 → 生成 [Q1, Q2, Q3]
    Q1 → 检索 → LLM生成 A1
    Q2 + A1作为上下文 → 检索 → LLM生成 A2
    Q3 + A1+A2作为上下文 → 检索 → LLM生成 A3（最终答案）

【数据流示意 —— 独立回答】
  总问题 → 生成 [Q1, Q2, Q3]
    Q1 → 检索 → LLM生成 A1
    Q2 → 检索 → LLM生成 A2  （与A1无关）
    Q3 → 检索 → LLM生成 A3  （与A1、A2无关）
    [A1, A2, A3] 汇总为上下文 → LLM生成最终答案
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 过滤掉 LangChain 的弃用警告和 beta 警告，终端更干净
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*", category=UserWarning)
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from operator import itemgetter

# 加载 .env 环境变量
load_dotenv()

# ==============================================================================
# 01. 加载本地HTML文档
# ==============================================================================
loader = BSHTMLLoader(
    file_path='./documents/LLM Powered Autonomous Agents _ Lil Log.html',
    open_encoding='utf-8',  # 关键：强制用 UTF-8 解码
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        ),
        features="html.parser"
    )
)
blog_docs = loader.load()


# ==============================================================================
# 02. Split（本地操作，不扣费）
# ==============================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs)


# ==============================================================================
# 03. Index（Embedding）[扣费]
# ==============================================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
    dimensions=1024,
    chunk_size=10,
    # 关掉 tiktoken 校验，避免阿里云不兼容报错
    check_embedding_ctx_length=False
)

PERSIST_DIR = "./chroma_storage"

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # 库已存在：直接加载，不调用 Embedding API（不扣费）
    print("[INFO] 检测到已有向量库，直接加载，跳过 Embedding 建库...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_knowledge_base"
    )
else:
    # 首次运行：建库，会对所有 splits 调用 Embedding API（扣费）
    print("[INFO] 首次运行，正在建库并持久化（此步骤产生 Embedding 费用）...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="my_knowledge_base",
        persist_directory=PERSIST_DIR
    )


# ==============================================================================
# 04. Retriever（检索时 Embedding）[扣费]
# ==============================================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ==============================================================================
# 05. 公共部分：子问题生成链
# ==============================================================================
# 让 LLM 把复杂问题拆成 3 个可以独立回答的子问题
# 注意：Prompt 用英文，LLM 对英文指令的遵循度通常更高
template_decomposition = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)

llm = ChatOpenAI(
    model='deepseek-v3.2',
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)

# 把 LLM 输出按行拆分，过滤空行，得到子问题列表
generate_queries_decomposition = (
    prompt_decomposition
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

# 测试用问题（与图片中的示例一致）
question = "What are the main components of an LLM-powered autonomous agent system?"


# ==============================================================================
# 06. 方法一：Answer Recursively（递归回答）
# ==============================================================================
# 核心思想：串行处理每个子问题，前一个子问题的答案作为后一个子问题的上下文。
# 这样后面的子问题可以基于前面已经获得的知识进行更深入的推理。

# Prompt 模板：需要三个输入变量
#   - question:  当前要回答的子问题
#   - q_a_pairs: 前面所有子问题的 Q&A 对（作为背景知识）
#   - context:   当前子问题检索到的文档片段
template_recursive = """Here is the question you need to answer:
---
{question}
---

Here is any available background question + answer pairs:
---
{q_a_pairs}
---

Here is additional context relevant to the question:
---
{context}
---

Use the above context and any background question + answer pairs to answer the question: {question}"""

prompt_recursive = ChatPromptTemplate.from_template(template_recursive)


def format_qa_pair(question, answer):
    """把一个 Q&A 对格式化为字符串，方便追加到历史上下文中。"""
    return f"Question: {question}\nAnswer: {answer}\n\n"


print("\n" + "=" * 60)
print("【方法一】Answer Recursively（递归回答）")
print("=" * 60)

# 生成子问题列表
sub_questions = generate_queries_decomposition.invoke({"question": question})
print(f"\n[INFO] 生成的子问题（共 {len(sub_questions)} 个）：")
for i, q in enumerate(sub_questions, 1):
    print(f"  {i}. {q}")

# 逐个串行处理子问题
q_a_pairs = ""  # 累积的 Q&A 历史
for i, q in enumerate(sub_questions, 1):
    print(f"\n--- 正在处理子问题 {i}/{len(sub_questions)}: {q} ---")

    # 构建递归链：
    #   itemgetter("question") 从输入字典取出当前子问题
    #   | retriever           对该子问题做向量检索
    #   → 作为 "context"
    #   itemgetter("q_a_pairs") 取出历史 Q&A
    #   → 作为 "q_a_pairs"
    rag_chain_recursive = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs")
        }
        | prompt_recursive
        | llm
        | StrOutputParser()
    )

    # 调用链：传入当前子问题 + 历史 Q&A 对
    answer = rag_chain_recursive.invoke({"question": q, "q_a_pairs": q_a_pairs})

    # 把当前子问题的 Q&A 追加到历史中，供下一个子问题使用
    q_a_pairs += "\n---\n" + format_qa_pair(q, answer)

    print(f"[子问题 {i} 答案]: {answer[:200]}...")

print("\n" + "=" * 60)
print("【方法一】最终答案（最后一个子问题的回答）：")
print("=" * 60)
print(answer)


# ==============================================================================
# 07. 方法二：Answer Individually（独立回答后汇总）
# ==============================================================================
# 核心思想：每个子问题独立检索、独立回答，互不干扰。
# 最后把所有子答案打包，让 LLM 基于这些子答案生成对原始总问题的完整回答。

# 标准 RAG Prompt（每个子问题独立使用，无需历史上下文）
template_rag = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt_rag = ChatPromptTemplate.from_template(template_rag)


def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """
    对总问题分解出的每个子问题，独立做检索和回答。

    参数：
      question:                   原始总问题
      prompt_rag:                 标准 RAG Prompt 模板
      sub_question_generator_chain: 子问题生成链

    返回：
      (rag_results, sub_questions)
      - rag_results:    每个子问题对应的答案列表
      - sub_questions:  生成的子问题列表
    """
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    rag_results = []

    for sub_q in sub_questions:
        # 对每个子问题独立检索文档
        retrieved_docs = retriever.invoke(sub_q)

        # 用标准 RAG 链独立回答该子问题
        answer = (prompt_rag | llm | StrOutputParser()).invoke(
            {"context": retrieved_docs, "question": sub_q}
        )
        rag_results.append(answer)

    return rag_results, sub_questions


def format_qa_pairs(questions, answers):
    """把多组 Q&A 对格式化为一个字符串，用于最终汇总。"""
    formatted_string = ""
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {q}\nAnswer {i}: {a}\n\n"
    return formatted_string.strip()


print("\n\n" + "=" * 60)
print("【方法二】Answer Individually（独立回答后汇总）")
print("=" * 60)

# 步骤 1：独立回答每个子问题
answers, sub_questions = retrieve_and_rag(
    question, prompt_rag, generate_queries_decomposition
)

print(f"\n[INFO] 各子问题的独立答案：")
for i, (q, a) in enumerate(zip(sub_questions, answers), 1):
    print(f"\n--- 子问题 {i} ---")
    print(f"Q: {q}")
    print(f"A: {a[:200]}...")

# 步骤 2：把所有子答案格式化为上下文
context = format_qa_pairs(sub_questions, answers)

# 步骤 3：用汇总 Prompt 让 LLM 基于所有子答案回答原始总问题
template_synthesize = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
prompt_synthesize = ChatPromptTemplate.from_template(template_synthesize)

final_rag_chain = (
    prompt_synthesize
    | llm
    | StrOutputParser()
)

final_answer = final_rag_chain.invoke({"context": context, "question": question})

print("\n" + "=" * 60)
print("【方法二】最终汇总答案：")
print("=" * 60)
print(final_answer)
