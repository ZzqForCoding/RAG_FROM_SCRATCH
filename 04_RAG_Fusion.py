"""
================================================================================
Part 6: RAG-Fusion（互逆排序融合）
================================================================================
【核心问题】
用户问题模糊/表述单一时，向量检索召回不足，导致 LLM 回答质量差。

【解决思路】
与 03_Multi_Query 类似：把原问题改写成多个不同措辞的查询，分别检索。
但核心差异在于【合并方式】：

  03 Multi Query：简单并集去重（get_unique_union）
    → 只保留"出现过哪些文档"，不保留排名信息，文档无序

  04 RAG-Fusion：RRF 互逆排序融合（Reciprocal Rank Fusion）
    → 利用每轮检索的排名信息加权打分，多轮都排在前列的文档获得更高权重
    → 最终得到一个【有序】的文档列表

【RRF 公式】
  score(d) = Σ [ 1 / (rank_i(d) + k) ]
  其中：
    - rank_i(d)：文档 d 在第 i 轮检索中的排名位置（从0开始）
    - k：平滑常数，通常取 60，防止排名靠后的文档得分过度衰减
    - Σ：对多轮检索的结果求和

【直观理解】
  - 一篇文档如果在多个查询中都排第1名，它的得分 = 4 × 1/(0+60) = 0.067，非常高
  - 一篇文档只在某个查询中排第10名，它的得分 = 1/(9+60) = 0.014，很低
  - 因此：【被多轮查询共同认可且排名靠前的文档 → 排名更靠前】

【与 03 的对比总结】
  ┌─────────────┬──────────────────┬─────────────────────────────┐
  │   维度      │   03 Multi Query │   04 RAG-Fusion             │
  ├─────────────┼──────────────────┼─────────────────────────────┤
  │ 查询生成    │  多个改写        │  多个改写（相同）           │
  │ 检索方式    │  retriever.map() │  retriever.map()（相同）    │
  │ 合并算法    │  简单并集去重    │  RRF 互逆排序融合           │
  │ 输出顺序    │  无序            │  按融合得分降序排列         │
  │ 核心优势    │  扩大覆盖面      │  扩大覆盖面 + 智能排序      │
  └─────────────┴──────────────────┴─────────────────────────────┘

【完整数据流】
  用户问题 → LLM生成4个改写查询 → retriever.map()并行检索(4×k个结果)
    → RRF融合排序 → 得到有序Document列表 → 作为context进入final_rag_chain
    → LLM生成答案
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

# ===========================
# 01. 加载本地HTML文档
# ===========================
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


# ===========================
# 02. Split（本地操作，不扣费）
# ===========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs)


# ===========================
# 03. Index（Embedding）[扣费]
# ===========================
# 加 persist_directory + 目录存在性检查，避免重复建库扣费
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
    dimensions=1024,
    chunk_size=10,
    # OpenAIEmbeddings 不只是个 HTTP 客户端，它捆绑了 OpenAI 生态的假设（tiktoken）。
    # 虽然阿里云百炼兼容了 OpenAI 的 API 格式，但 tiktoken 这个"附带品"不兼容阿里云模型。
    # 加 check_embedding_ctx_length=False 可以关掉它；如果还报错，直接用你手写的 CustomEmbeddings 是最干净的方案。
    check_embedding_ctx_length=False
)

PERSIST_DIR = "./chroma_storage"

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # ✅ 库已存在：直接加载，不调用 Embedding API（不扣费）
    print("[INFO] 检测到已有向量库，直接加载，跳过 Embedding 建库...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_knowledge_base"  # 必须与 01/03 文件一致！
    )

else:
    # 💰 首次运行：建库，会对所有 splits 调用 Embedding API（扣费）
    print("[INFO] 首次运行，正在建库并持久化（此步骤产生 Embedding 费用）...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="my_knowledge_base",  # 必须与加载时一致
        persist_directory=PERSIST_DIR
    )


# ===========================
# 04. Retriever（检索时 Embedding）[扣费]
# ===========================
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ===========================
# 05. 生成多个查询（Query Generation）
# ===========================
# 与 03_Multi_Query 完全一致：让 LLM 把原问题改写成多个不同措辞的查询
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):
Do not add numbers, bullet points, or any prefix to each query."""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)

# 把"一个问题"变成"四个问题的数组"，交给下游批量检索
generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(
        model='deepseek-v3.2',
        temperature=0,
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("API_BASE")
    )
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


# ===========================
# 06. RRF 互逆排序融合（核心差异！）
# ===========================
from langchain_core.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    """
    互逆排序融合（Reciprocal Rank Fusion, RRF）

    输入：list[list[Document]]
      外层列表 = 每个改写查询的检索结果（4个查询就有4个子列表）
      内层列表 = 该查询检索到的 k 个文档片段（按相似度从高到低排序）

    输出：list[Document]（按 RRF 融合得分从高到低排序的文档列表）

    【RRF 原理】
    1. 对每轮检索结果，按排名给分：score = 1 / (rank + k)
       - rank=0（第1名）→ score = 1/60 = 0.0167
       - rank=1（第2名）→ score = 1/61 = 0.0164
       - rank=9（第10名）→ score = 1/69 = 0.0145

    2. 同一篇文档可能出现在多轮检索中，得分累加：
       - doc_A 在 Q1 排第1，在 Q2 排第2
       - doc_A 总得分 = 1/(0+60) + 1/(1+60) = 0.0167 + 0.0164 = 0.0331

    3. 按总得分降序排列，得分越高说明该文档被越多查询"共同认可"

    【为什么 k=60？】
    k 是平滑参数。没有 k 的话，rank=0 时 score=1/0=无穷大，太激进。
    k=60 可以让排名差异的影响更平滑，防止某个查询的第一名过度主导最终排序。
    """
    # ① 初始化：用字典记录每个文档的融合得分
    fused_scores = {}

    # ② 遍历每轮检索结果
    for docs in results:
        # ③ 遍历该轮中的每个文档，记录其排名（enumerate 从0开始）
        for rank, doc in enumerate(docs):
            # Document 是自定义对象，不能直接当 dict key，先序列化成字符串
            doc_str = dumps(doc)

            # 如果这篇文档第一次出现，初始得分为 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # ④ 累加 RRF 得分：1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # ⑤ 按融合得分降序排列
    #    sorted(..., reverse=True) → 得分高的排在前面
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # ⑥ 反序列化回 Document 对象，只返回文档列表（score 已用于排序，不再保留）
    return [loads(doc_str) for doc_str, score in reranked]


# ===========================
# 07. 组装 RAG-Fusion 检索链
# ===========================
# 数据流示意：
#   "What is task decomposition for LLM agents?"
#     ↓ generate_queries
#   ["What is task decomposition...", "How do LLM agents break down...", ...]  ← 4个改写
#     ↓ retriever.map()
#   [[doc1, doc2], [doc2, doc3], [doc1, doc4], [doc5, doc6]]  ← 4×2=8个结果，每组内部有序
#     ↓ reciprocal_rank_fusion
#   [doc2, doc1, doc3, doc4, doc5, doc6]  ← 按RRF得分重新排序（doc2被多轮命中且排名靠前）
#
# 与 03 的关键区别：
#   03 → get_unique_union：只去重，不排序，[doc1, doc2, doc3, ...] 无序
#   04 → reciprocal_rank_fusion：去重 + 排序，被多轮查询共同认可的文档排在前面

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion


# ===========================
# 08. 组装最终 RAG 链（RAG-Fusion + Generation）
# ===========================
# itemgetter("question") 的作用：
#   从输入字典中提取 "question" 字段的值。
#   例如：输入 {"question": "What is task decomposition?"} → 输出 "What is task decomposition?"

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model='deepseek-v3.2',
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)

# final_rag_chain 的数据流示意：
#   {"question": "What is task decomposition for LLM agents?"}
#     ↓
#   ├─ "context" 分支：retrieval_chain_rag_fusion
#   │     接收 question → generate_queries（生成4个改写）
#   │                   → retriever.map()（4次检索）
#   │                   → reciprocal_rank_fusion（RRF排序融合）
#   │     输出：按 RRF 得分排序的 Document 列表（自动转成字符串）
#   │
#   └─ "question" 分支：itemgetter("question")
#   │     从输入字典中提取原问题
#   │     输出："What is task decomposition for LLM agents?"
#   │
#   合并成：{"context": "...拼接的文档内容...", "question": "What is task decomposition..."}
#     ↓ prompt → LLM → StrOutputParser() → 最终答案

final_rag_chain = (
    {
        # context：复用上面定义的 retrieval_chain_rag_fusion
        # 注意：retrieval_chain_rag_fusion 的输出是 list[Document]，prompt 中的 {context} 需要字符串。
        # LangChain 会自动把 Document 列表转成字符串（通过内置的 format_docs 逻辑）。
        "context": retrieval_chain_rag_fusion,

        # question：从输入字典里把原问题原样掏出来，透传给 prompt
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ===========================
# 09. 运行示例
# ===========================
if __name__ == "__main__":
    question = "What is task decomposition for LLM agents?"

    # [扣费] RAG-Fusion 检索：1次LLM生成4个问题 + 4次Embedding检索 + RRF排序
    print("\n=== RAG-Fusion 检索结果 ===")
    fused_docs = retrieval_chain_rag_fusion.invoke({"question": question})
    print(f"共检索到 {len(fused_docs)} 篇去重后的文档（已按 RRF 得分排序）")
    for i, doc in enumerate(fused_docs[:3], 1):
        print(f"\n--- 第 {i} 名文档 ---")
        print(doc.page_content[:200] + "...")

    # [扣费] 完整 RAG：RAG-Fusion 检索 + LLM 生成答案
    print("\n=== 最终答案 ===")
    answer = final_rag_chain.invoke({"question": question})
    print(answer)
