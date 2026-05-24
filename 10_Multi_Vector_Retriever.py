"""
================================================================================
Part 10: Multi-Vector Retriever（多向量检索器 / 多表示索引）
================================================================================
【核心问题】
RAG 中有一个经典的 chunk 尺寸两难困境：
  - 小 chunk：检索更精准，但上下文不完整，LLM 回答缺乏背景
  - 大 chunk：上下文完整，但检索时噪声多，容易召回无关内容

能不能"检索用小块，回答用大块"？这就是 Multi-Vector Retriever 要解决的问题。

【核心思路】
不要把原始文档直接存入向量库。而是：
  1. 对每个文档生成一个"更适合检索"的表示（摘要 / 小 chunk / 假设问题）
  2. 把这个"小表示"存入向量库（用于检索）
  3. 把原始完整文档存入另一个存储（DocStore）
  4. 两者通过 doc_id 关联
  5. 检索时：用查询匹配小表示 → 返回关联的原始完整文档

【类比】
  图书馆的索引卡片系统：
  - 卡片上只有书名、作者、摘要（小表示）
  - 你通过卡片找到书 → 去书架上拿完整书籍（原始文档）
  - 你不会把整本书的内容抄在卡片上

【适用场景】
  ✅ 文档很长，直接切片会丢失上下文
  ✅ 检索精度和回答质量需要分别优化
  ✅ 需要多粒度检索（摘要匹配 + 原文返回）
  ❌ 文档都很短（直接用普通 Retriever 即可）
  ❌ 不需要完整上下文（额外存储和 LLM 摘要成本不划算）

【参考资料】
  - LangChain Multi-Vector Retriever:
    https://python.langchain.com/docs/how_to/multi_vector/
  - RAG From Scratch — Multi-Representation Indexing:
    https://www.youtube.com/watch?v=gTCU9I6X4C8
================================================================================
"""

import os
import uuid
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*", category=UserWarning)

from langchain_core.stores import InMemoryByteStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ==============================================================================
# 01. 配置 Embedding 和 LLM
# ==============================================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
    check_embedding_ctx_length=False,
)

llm = ChatOpenAI(
    model="deepseek-v3.2",
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
)


# ==============================================================================
# 02. 加载文档
# ==============================================================================
"""
这里加载 Lilian Weng 的两篇经典博客：
  - "LLM Powered Autonomous Agents"（Agent 系统设计）
  - "Human Data Quality"（数据质量）

WebBaseLoader 直接从 URL 抓取 HTML，自动提取正文文本。
"""

print("=" * 60)
print("【Step 1】加载文档")
print("=" * 60)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
print(f"  加载第一篇: {len(docs)} 个 Document")

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())
print(f"  加载第二篇，共 {len(docs)} 个 Document")

for i, doc in enumerate(docs):
    print(f"  [{i}] {doc.page_content[:80]}... (总长 {len(doc.page_content)} 字符)")


# ==============================================================================
# 03. 生成文档摘要（用 LLM 批量处理）
# ==============================================================================
"""
对每个长文档，用 LLM 生成一段简洁的摘要。
这个摘要就是"小表示"——用来做向量检索的索引条目。

chain.batch() 并发处理多个文档，max_concurrency=5 控制并行度。
"""

print("\n" + "=" * 60)
print("【Step 2】用 LLM 为每个文档生成摘要（这会调用 LLM API）")
print("=" * 60)

summarize_chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
        "用一段话总结以下文档的核心内容，不超过200字：\n\n{doc}"
    )
    | llm
    | StrOutputParser()
)

# batch 并发处理：两个文档同时发给 LLM，比逐个调用快
summaries = summarize_chain.batch(docs, {"max_concurrency": 5})

for i, (doc, summary) in enumerate(zip(docs, summaries)):
    print(f"\n  [{i}] 原文档长度: {len(doc.page_content)} 字符")
    print(f"      摘要长度:   {len(summary)} 字符 (压缩率 {len(summary)/len(doc.page_content)*100:.1f}%)")
    print(f"      摘要内容:   {summary[:120]}...")


# ==============================================================================
# 04. 构建 MultiVectorRetriever（核心：双存储架构）
# ==============================================================================
"""
MultiVectorRetriever 维护两个独立的存储：

┌─────────────────────────────────────────────────────────────┐
│  MultiVectorRetriever                                       │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  VectorStore         │    │  DocStore (ByteStore)       │ │
│  │  (Chroma)            │    │  (InMemoryByteStore)        │ │
│  │                      │    │                             │ │
│  │  存：摘要的 Embedding  │    │  存：原始完整文档             │ │
│  │  用途：语义相似度搜索   │    │  用途：返回给 LLM 做上下文    │ │
│  │                      │    │                             │ │
│  │  ┌──────────────┐    │    │  ┌───────────────────────┐  │ │
│  │  │ summary_0    │────┼────┼─→│ doc_id="abc123"       │  │ │
│  │  │ doc_id=abc123│    │    │  │ page_content="完整..." │  │ │
│  │  └──────────────┘    │    │  └───────────────────────┘  │ │
│  │  ┌──────────────┐    │    │  ┌───────────────────────┐  │ │
│  │  │ summary_1    │────┼────┼─→│ doc_id="def456"       │  │ │
│  │  │ doc_id=def456│    │    │  │ page_content="完整..." │  │ │
│  │  └──────────────┘    │    │  └───────────────────────┘  │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                                             │
│  检索流程：                                                  │
│    用户查询 → VectorStore.similarity_search() → 匹配到摘要    │
│    → 取出 doc_id → DocStore 查找原始文档 → 返回完整文档       │
└─────────────────────────────────────────────────────────────┘
"""

print("\n" + "=" * 60)
print("【Step 3】构建 MultiVectorRetriever")
print("=" * 60)

# VectorStore：存摘要的 Embedding，用于语义检索
vectorstore = Chroma(
    collection_name="multi_vector_summaries",
    embedding_function=embeddings,
)

# DocStore：存原始文档的"大本营"
docstore = InMemoryByteStore()
id_key = "doc_id"  # 关联 VectorStore 和 DocStore 的桥梁

# 创建多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=docstore,
    id_key=id_key,
)

# 为每个原始文档生成唯一 ID
doc_ids = [str(uuid.uuid4()) for _ in docs]
print(f"  生成了 {len(doc_ids)} 个 doc_id:")
for i, did in enumerate(doc_ids):
    print(f"    [{i}] {did}")

# ---------------------------------------------------------------------------
# 4a. 将"摘要 Document"存入 VectorStore（同时标记 doc_id）
# ---------------------------------------------------------------------------
"""
这里的关键：每个摘要 Document 的 metadata 里存了对应原始文档的 doc_id。
当检索命中这个摘要时，MultiVectorRetriever 会通过 doc_id 去 DocStore 取原始文档。
"""
summary_docs = [
    Document(
        page_content=summary,
        metadata={id_key: doc_ids[i]},  # ← 通过 doc_id 关联到原始文档
    )
    for i, summary in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
print(f"\n  已向 VectorStore 存入 {len(summary_docs)} 个摘要")

# ---------------------------------------------------------------------------
# 4b. 将"原始文档"存入 DocStore（key = doc_id, value = Document）
# ---------------------------------------------------------------------------
retriever.docstore.mset(list(zip(doc_ids, docs)))
print(f"  已向 DocStore 存入 {len(docs)} 个原始文档")


# ==============================================================================
# 05. 检索演示：对比摘要匹配 vs 原始文档返回
# ==============================================================================
"""
检索时发生了什么：

  用户查询: "Memory in agents"
       ↓
  VectorStore.similarity_search("Memory in agents")
       ↓  （在摘要中做语义匹配）
  命中 summary_0 → doc_id = "abc123..."
       ↓
  docstore.mget(["abc123..."]) → 原始完整文档
       ↓
  返回：完整的 Lilian Weng Agent 博客
"""

print("\n" + "=" * 60)
print("【Step 4】检索演示")
print("=" * 60)

query = "Memory in agents"

# --- 4a. 直接查 VectorStore（返回的是摘要，不是原始文档） ---
print(f"\n  查询: \"{query}\"")
print(f"\n  --- 直接查 VectorStore（返回摘要） ---")
sub_docs = vectorstore.similarity_search(query, k=1)
for i, d in enumerate(sub_docs):
    print(f"  [{i}] 类型: 摘要")
    print(f"      内容: {d.page_content[:200]}...")
    print(f"      metadata: {d.metadata}")

# --- 4b. 通过 MultiVectorRetriever 检索（返回原始完整文档） ---
print(f"\n  --- 通过 MultiVectorRetriever（返回原始文档） ---")
retrieved_docs = retriever.invoke(query)
for i, d in enumerate(retrieved_docs[:1]):  # 只看第一个结果
    print(f"  [{i}] 类型: 原始完整文档")
    print(f"      长度: {len(d.page_content)} 字符")
    # 对比摘要大小
    summary_size = len(summaries[0]) if summaries else 0
    print(f"      对比: 摘要仅 {summary_size} 字符, 原始文档 {len(d.page_content)} 字符")
    print(f"      倍率: 原始文档是摘要的 {len(d.page_content)/summary_size:.1f}x")
    print(f"      >>> 检索用摘要匹配, LLM 看到的是完整原文 <<<")


# ==============================================================================
# 06. 对比：普通 Retriever vs MultiVectorRetriever
# ==============================================================================
"""
┌──────────────────────┬─────────────────────────────┬──────────────────────────────┐
│ 对比维度              │ 普通 Retriever               │ MultiVectorRetriever          │
├──────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ 存入向量库的内容       │ 原始文档切片                  │ 文档摘要（或其他小表示）        │
│ 检索时匹配的对象       │ 文档切片                      │ 摘要                          │
│ 返回给 LLM 的内容     │ 匹配到的切片本身               │ 摘要对应的原始完整文档          │
│ 检索精度              │ 受切片质量影响                 │ 更高（摘要更聚焦核心语义）      │
│ 生成质量              │ 受切片大小限制                 │ 更好（完整文档提供更多上下文）  │
│ 额外成本              │ 无                            │ 需要 LLM 生成摘要              │
│ 适用场景              │ 通用                          │ 长文档、检索与回答需要分别优化  │
└──────────────────────┴─────────────────────────────┴──────────────────────────────┘
"""


# ==============================================================================
# 07. 其他"小表示"类型（不只是摘要）
# ==============================================================================
"""
除了 LLM 摘要，MultiVectorRetriever 还支持其他"小表示"类型：

1. 摘要（Summaries）—— 本文件演示的方式
   → LLM 将长文档压缩为短摘要，摘要 Embedding 存入向量库

2. 小切片（Smaller Chunks）
   → 把文档切成 200 token 的小块存入向量库，检索后返回父文档
   → 不需要 LLM 调用，成本更低

3. 假设问题（Hypothetical Questions）
   → 对每个文档，让 LLM 生成"用户可能会怎么问这个问题"
   → 存入向量库的是问题，问题与用户查询在语义空间更接近
   → 这就是 HyDE（07）的反向思路

4. 关键词 / 实体列表
   → 提取文档中的关键词或实体作为索引

方案选择：
  - 有 LLM 预算、文档很长 → 摘要
  - 不想额外调 LLM → 小切片
  - 查询和文档用词差异大 → 假设问题
"""


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("【运行完成】")
    print("=" * 60)


# ==============================================================================
# 多向量检索器 · 总结
# ==============================================================================
"""
【一句话总结】
MultiVectorRetriever 实现了"检索用小块，回答用大块"——摘要做索引，原文做上下文。

【核心组件】
  - VectorStore：存"小表示"（摘要）的 Embedding，负责语义匹配
  - DocStore：存"大原文"（完整文档），负责返回给 LLM
  - doc_id：两者之间的桥梁

【与其他策略的关系】
  - 与 HyDE（07）互补：HyDE 是"用假设文档去检索"，MultiVector 是"用摘要做索引"
    两者可以结合：用假设问题做小表示 → MultiVectorRetriever → 返回原文
  - 与 Query Analysis（09）正交：Query Analysis 负责提取 filter，
    MultiVector 负责检索架构，可以在 MultiVector 外层加 filter

【思考题】

1. 摘要丢了细节怎么办？
   → 这正是 MultiVector 的设计初衷——摘要只负责"找到"文档，
     细节在原始文档里，LLM 最终看到的是完整文档。
   → 但如果摘要没概括到的信息恰好是用户要的，就会漏召回。
     解决：可以混合摘要 + 小切片，多粒度索引。

2. 为什么用 InMemoryByteStore 而不是持久化？
   → 简化演示。生产环境可以用 LocalFileStore 或 Redis 持久化 DocStore。

3. 摘要生成失败或质量差怎么办？
   → 可以加验证步骤，摘要长度太短或太长的重新生成。
   → 或者退回到"小切片"模式，不依赖 LLM 摘要质量。
"""
