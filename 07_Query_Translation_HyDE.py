"""
================================================================================
Part 9: HyDE（Hypothetical Document Embeddings / 假设文档嵌入）
================================================================================
【核心问题】
用户查询（Query）和文档片段（Document）在 Embedding 空间中天然分布不同。
用户问的是"问题"（简短、疑问句式），而知识库中存储的是"答案片段"（陈述句、段落）。
这导致直接用用户问题做向量检索时，语义匹配效果往往不佳。

【解决思路】
HyDE 的核心洞察：与其直接用"问题"去检索，不如让 LLM 先根据问题
生成一篇"假设的、包含答案的文档"（Hypothetical Document），
然后用这篇假设文档去向量库检索。

为什么假设文档检索效果更好？
  - 假设文档在语义空间上更接近真实文档（都是陈述性文本，而非疑问句）
  - 假设文档包含了问题背后的"意图"和"上下文"，是一个更丰富的检索查询
  - 相当于把"查询空间"的向量，桥接到了"文档空间"

【论文】
  Precise Zero-Shot Dense Retrieval without Relevance Labels
  https://arxiv.org/abs/2212.10496

【HyDE 与 Multi Query 的本质区别】
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │   策略          │   核心操作                                            │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │  Multi Query    │   横向扩展：1个问题 → N个不同措辞的问题 → 分别检索    │
  │  (03, 04)       │   始终停留在【查询空间】                               │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │  HyDE (本文件)  │   纵向转换：1个问题 → 1篇假设文档 → 用文档检索        │
  │                 │   从【查询空间】跨越到【文档空间】                     │
  └─────────────────┴──────────────────────────────────────────────────────┘

  Multi Query 是"换种说法问同一个问题"，HyDE 是"先写一份可能的答案，再拿答案去找资料"。
  前者解决"表述不同"的问题，后者解决"查询和文档语义空间不对齐"的问题。

【五种查询翻译策略总览】
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │   策略          │   抽象方向 / 操作方式                                 │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │  Multi Query    │   水平方向：同一粒度，多种表述（03）                  │
  │  RAG-Fusion     │   水平方向：同上 + RRF 重排序（04）                   │
  │  Decomposition  │   向下拆解：大问题 → 多个小问题（05）                 │
  │  Step Back      │   向上抽象：具体问题 → 更宏观的问题（06）             │
  │  HyDE           │   空间转换：问题 → 假设文档（本文件）                 │
  └─────────────────┴──────────────────────────────────────────────────────┘

【HyDE 的适用场景】
  ✅ 查询非常短/模糊，与文档用词差异大（如专业术语 vs 通俗说法）
  ✅ 跨语言检索：用户用中文提问，但知识库是英文文档
    → 生成英文假设文档，用英文文档去检索英文知识库
  ✅ 零样本场景：没有用户历史查询做参考，LLM 的"猜测"比原始查询更有信息量
  ✅ 查询涉及隐含意图，需要展开解释（如"为什么这个方法有效"）

【不适用场景】
  ❌ 查询本身就很长很详细（已经接近文档片段的长度）
  ❌ 计算资源受限（HyDE 需要额外一次 LLM 调用生成假设文档）
  ❌ 对 LLM 幻觉零容忍（假设文档可能包含错误信息，尽管只是用于检索）
  ❌ 知识库本身质量很差（HyDE 再强也救不了垃圾数据）

【流程示意】
  用户问题: "What is task decomposition for LLM agents?"
      ↓
  [LLM 生成假设文档]
      ↓
  假设文档: "Task decomposition is a core capability of LLM agents..."
      ↓
  [用假设文档做 Embedding → 向量检索]
      ↓
  检索到的相关文档片段
      ↓
  [原始问题 + 检索文档 → 最终回答]
      ↓
  答案
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from operator import itemgetter

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ==============================================================================
# 05. LLM（生成假设文档 + 最终回答）
# ==============================================================================
llm = ChatOpenAI(
    model='deepseek-v3.2',
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)


# ==============================================================================
# 06. HyDE：生成假设文档
# ==============================================================================
# HyDE Prompt：让 LLM 根据问题写一段包含答案的"假设文档"
# 注意：这里要求 LLM 写的是一个"段落"（passage），而不是直接回答问题。
# 这段假设文档将作为检索的"查询向量"，所以它应该：
#   - 包含尽可能多的相关关键词和术语
#   - 采用陈述性语气（像文档片段，不像回答问题）
#   - 尽量详细、具体，以便在向量空间中与真实文档对齐

hyde_template = """Please write a scientific paper passage to answer the question.
Write it as if it were a paragraph from a research paper or technical documentation.

Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(hyde_template)

# 假设文档生成链：问题 → Prompt → LLM → 假设文档（纯文本）
generate_docs_for_retrieval = (
    prompt_hyde
    | llm
    | StrOutputParser()
)


# ==============================================================================
# 07. 检索 + RAG 回答（分步调用，教学更清晰）
# ==============================================================================
# HyDE 的完整流程分三步：
#   ① 生成假设文档  ② 用假设文档检索  ③ 用检索结果 + 原始问题生成答案

question = "What is task decomposition for LLM agents?"

print("\n" + "=" * 60)
print(f"【原始问题】{question}")
print("=" * 60)

# Step ①：生成假设文档
print("\n[Step 1] 正在生成假设文档（HyDE）...")
hypothetical_doc = generate_docs_for_retrieval.invoke({"question": question})
print(f"\n【假设文档】\n{hypothetical_doc}\n")

# Step ②：用假设文档做向量检索
# 关键：这里传入 retriever 的不是原始问题，而是 LLM 生成的假设文档！
print("[Step 2] 正在用假设文档进行向量检索...")
retrieved_docs = retriever.invoke(hypothetical_doc)
print(f"检索到 {len(retrieved_docs)} 个相关文档片段\n")

# Step ③：最终 RAG 回答（原始问题 + 检索文档）
rag_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt_rag = ChatPromptTemplate.from_template(rag_template)

final_rag_chain = (
    prompt_rag
    | llm
    | StrOutputParser()
)

print("[Step 3] 正在生成最终答案...")
answer = final_rag_chain.invoke({
    "context": format_docs(retrieved_docs),
    "question": question
})
print(f"\n【最终答案】\n{answer}")


# ==============================================================================
# 08. 也可以组合成完整 LCEL 链
# ==============================================================================
# 上面的分步调用教学意义更强。如果你需要一条完整的 Runnable 链，
# 可以用 RunnableLambda 把"生成假设文档 + 检索 + 格式化"包装成一个步骤：
# 
#   hyde_retrieval = RunnableLambda(
#       lambda x: {
#           "context": format_docs(retriever.invoke(
#               generate_docs_for_retrieval.invoke(x)
#           )),
#           "question": x["question"]
#       }
#   )
# 
#   full_chain = hyde_retrieval | prompt_rag | llm | StrOutputParser()
#   answer = full_chain.invoke({"question": question})


# ==============================================================================
# 09. 思考题：HyDE 的局限性与改进方向
# ==============================================================================
"""
【思考题】

1. 假设文档如果包含错误信息，会怎样？
   → 假设文档只用于"检索"，不直接用于生成答案。即使假设文档有幻觉，
     只要检索到的真实文档是正确的，最终答案仍然可靠。
     但极端情况下，错误的假设文档可能导致检索偏离目标。

2. 能否把 HyDE 和 Multi Query 结合起来？
   → 可以！先用 HyDE 生成一篇假设文档，再用 Multi Query 的思路把这篇
     假设文档改写成多个不同角度的版本，分别检索。这是"纵向转换 + 横向扩展"
     的组合策略，在复杂场景中可能效果更好（但成本也更高）。

3. HyDE 的假设文档长度如何控制？
   → 太短：信息不足，检索效果差。
   → 太长：可能引入过多无关信息，稀释核心语义。
   → 可以通过 Prompt 中的指令（如"Write a concise 200-word passage"）
     或调整 LLM 的 max_tokens 来控制长度。

4. 为什么 temperature=0？
   → 假设文档需要尽量确定、一致。温度太高会导致每次生成的假设文档差异很大，
     检索结果不稳定。但极低的温度也可能限制创造性，在探索性查询中可以适当提高。

【论文参考】
  - HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels
    https://arxiv.org/abs/2212.10496
  - LangChain Cookbook 实现：
    https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb
"""
