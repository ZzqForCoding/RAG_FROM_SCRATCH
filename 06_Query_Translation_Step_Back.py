"""
================================================================================
Part 8: Step Back（回退 / 抽象提升）
================================================================================
【核心问题】
用户问题可能过于具体，导致检索到的文档碎片缺乏宏观背景，回答片面。

【解决思路】
让 LLM 先把具体问题"抽象提升"到一个更宏观的 Step-back Question，
然后同时用【原问题】和【宏观问题】两条路检索，把宏观背景 + 具体细节
融合后回答。

【三种查询翻译策略的对比】（如图示）
  ┌─────────────────┬───────────────────────────────────────────────────┐
  │   策略          │   抽象方向                                        │
  ├─────────────────┼───────────────────────────────────────────────────┤
  │  Multi Query    │   水平方向：同一粒度，多种表述（03）                │
  │  RAG-Fusion     │   水平方向：同上 + RRF 重排序（04）                 │
  │  Decomposition  │   向下拆解：大问题 → 多个小问题（05）               │
  │  Step Back      │   向上抽象：具体问题 → 更宏观的问题（本文件）       │
  └─────────────────┴───────────────────────────────────────────────────┘

【Step Back 的核心价值】
  - 原问题检索（normal_context）：获取具体细节（细粒度）
  - 回退问题检索（step_back_context）：获取宏观背景（粗粒度）
  - 融合回答：既有细节又有全局视野，避免"见树不见林"

【示例】
  原问题："What is task decomposition for LLM agents?"
  Step-back："What are the core capabilities of LLM agents?"
  
  原问题检索 → 得到"任务分解的具体定义和实现"
  回退问题检索 → 得到"LLM Agent 的整体架构和核心能力"
  融合 → 回答不仅解释任务分解，还说明了它在 Agent 体系中的位置

【论文】
  Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models
  https://arxiv.org/pdf/2310.06117.pdf
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
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 加载 .env 环境变量
load_dotenv()


# ==============================================================================
# 公共工具函数
# ==============================================================================
def format_docs(docs):
    """将 Document 列表格式化为 \n\n 分隔的字符串，用于 Prompt 中的 {context}。"""
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================================================================
# 01. 加载本地HTML文档
# ==============================================================================
loader = BSHTMLLoader(
    file_path='./documents/LLM Powered Autonomous Agents _ Lil Log.html',
    open_encoding='utf-8',
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
    check_embedding_ctx_length=False
)

PERSIST_DIR = "./chroma_storage"

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("[INFO] 检测到已有向量库，直接加载，跳过 Embedding 建库...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_knowledge_base"
    )
else:
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ==============================================================================
# 05. LLM 初始化
# ==============================================================================
llm = ChatOpenAI(
    model='deepseek-v3.2',
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)


# ==============================================================================
# 06. Step-Back Question 生成链（Few-Shot Prompting）
# ==============================================================================
# 核心思想：通过 Few-Shot 示例，教 LLM 学会"抽象提升"——
# 把具体的问题改写成更宏观、更通用的 step-back 问题。
#
# 示例规律：
#   具体 → 宏观
#   "乐队成员能否合法逮捕？" → "乐队成员能做什么？"
#   "某人出生在哪个国家？" → "某人的个人历史是什么？"

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history",
    },
]

# 每个示例的格式：Human 问 input，AI 答 output
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# FewShotChatMessagePromptTemplate：把示例列表注入到对话历史中
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 完整 Prompt：System 指令 + Few-Shot 示例 + 用户当前问题
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at world knowledge. Your task is to step back "
            "and paraphrase a question to a more generic step-back question, "
            "which is easier to answer. Here are a few examples:",
        ),
        # Few-Shot 示例会自动展开为多条对话消息
        few_shot_prompt,
        # 当前用户的问题
        ("user", "{question}"),
    ]
)

# Step-Back 问题生成链：接收 {"question": ...} → 输出一个宏观问题字符串
generate_queries_step_back = prompt | llm | StrOutputParser()


# ==============================================================================
# 07. 测试：看看 Step-Back Question 长什么样
# ==============================================================================
question = "What is task decomposition for LLM agents?"

print("\n" + "=" * 60)
print("【Step Back】生成回退问题")
print("=" * 60)
print(f"原始问题: {question}")
step_back_question = generate_queries_step_back.invoke({"question": question})
print(f"回退问题: {step_back_question}")


# ==============================================================================
# 08. 最终 RAG 链：双路检索 + 融合回答
# ==============================================================================
# 核心设计：同时获取两套上下文
#   - normal_context:   用原问题检索 → 具体细节
#   - step_back_context: 用回退问题检索 → 宏观背景
# 然后把两套上下文都传给 LLM，让它融合回答。

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question.
Your response should be comprehensive and not contradicted with the following context if they are relevant.
Otherwise, ignore them if they are not relevant.

# Normal Context (from the original question):
{normal_context}

# Step-Back Context (from the abstracted question):
{step_back_context}

# Original Question: {question}
# Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# 构建 LCEL 链
# 字典中的三个键会并行执行（LangChain 会自动处理分支）
chain = (
    {
        # 分支1：用原始问题直接检索（获取具体细节）
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        # 分支2：先生成 step-back 问题，再用它检索（获取宏观背景）
        "step_back_context": generate_queries_step_back | retriever | format_docs,
        # 分支3：把原始问题透传到下游
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

print("\n" + "=" * 60)
print("【Step Back】最终融合回答")
print("=" * 60)
answer = chain.invoke({"question": question})
print(answer)
print("=" * 60)


# ==============================================================================
# 09. 补充：对比实验（可选）—— 单路 vs 双路
# ==============================================================================
# 下面的代码演示：只用原问题检索（normal only）和只用回退问题检索（step-back only）
# 分别是什么效果，帮助理解 Step Back 的价值。

print("\n" + "=" * 60)
print("【对比实验】单路检索的效果")
print("=" * 60)

# --- 只用原问题 ---
chain_normal_only = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "step_back_context": lambda x: "(无回退上下文)",
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

print("\n>>> 仅使用原问题检索：")
answer_normal = chain_normal_only.invoke({"question": question})
print(answer_normal[:500] + "..." if len(answer_normal) > 500 else answer_normal)

# --- 只用回退问题 ---
chain_stepback_only = (
    {
        "normal_context": lambda x: "(无原始上下文)",
        "step_back_context": generate_queries_step_back | retriever | format_docs,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

print("\n>>> 仅使用回退问题检索：")
answer_stepback = chain_stepback_only.invoke({"question": question})
print(answer_stepback[:500] + "..." if len(answer_stepback) > 500 else answer_stepback)
