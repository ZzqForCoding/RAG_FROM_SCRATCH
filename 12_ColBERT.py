"""
================================================================================
Part 12: ColBERT — Contextualized Late Interaction over BERT（上下文迟交互）
================================================================================
【核心问题】
传统 Embedding 模型（如 text-embedding-v4）将一整段文本压成一个向量：
  查询 "Python 的 GIL 如何影响多线程" → [0.23, -0.15, 0.78, ...]  (1个向量)
  文档 "Python GIL 详解..."          → [0.19, -0.21, 0.82, ...]  (1个向量)
  相似度 = cos(查询向量, 文档向量)

问题：一段 500 字的文本怎么可能用一个向量精确表达？信息被"压缩"丢失了。
尤其细粒度匹配（专有名词、数字、日期）时，单向量表示力不从心。

ColBERT 的思路：不要压缩！每个 token 各自独立编码，检索时做 token 级别的交互。

  ColBERT 编码：
  查询 "GIL 多线程" →  [v_GIL, v_多线程]            (2个向量，每个 token 一个)
  文档 "Python GIL 详解" → [v_Python, v_GIL, v_详解] (3个向量，每个 token 一个)

  检索时 MaxSim 打分：
    对查询的每个 token，找文档中与其最相似的 token，求和：
    Score = MaxSim(v_GIL, [v_Python, v_GIL, v_详解])
          + MaxSim(v_多线程, [v_Python, v_GIL, v_详解])
          = max(cos(v_GIL, doc_tokens))
          + max(cos(v_多线程, doc_tokens))

  → "GIL" 精确匹配到文档中的 "GIL"，得分很高
  → 多线程虽然没有直接匹配，但也找到了最相似的部分

这就是"迟交互"（Late Interaction）：
  - 编码阶段：查询和文档独立编码（不交互）— 高效，可预计算
  - 检索阶段：token 级别交互计算相似度 — 精细，保留细节

【ColBERT vs 传统 Embedding 对比】

  传统单向量 Embedding（如 text-embedding-v4）：
    查询 → [单个向量]          文档 → [单个向量]         cos 点积 → 得分
    ✅ 检索极快（向量库标准索引）
    ❌ 压缩损失，细节丢失

  Cross-Encoder（如 BERT 直接打分）：
    查询 + 文档 → [CLS] query [SEP] doc [SEP] → BERT → 得分
    ✅ 最精确（查询和文档深度交互）
    ❌ 每条 (查询, 文档) 都要跑一遍 BERT，检索 1 万条就得跑 1 万次

  ColBERT（迟交互）：
    查询 → [每个 token 一个向量]   文档 → [每个 token 一个向量]   MaxSim → 得分
    ✅ 保留 token 级别细节，比单向量精确得多
    ✅ 文档可预编码（离线索引），检索时只编码查询
    ❌ 存储量大（每个 token 一个向量，约为单向量的 128x）
    ❌ 需要特殊索引结构（PLAID / FAISS IVF），不能直接用 ChromaDB

【参考资料】
  - ColBERT 论文: https://arxiv.org/abs/2004.12832
  - ColBERTv2 论文: https://arxiv.org/abs/2112.01488
  - RAGatouille 文档: https://github.com/AnswerDotAI/ragatouille
  - LangChain ColBERT: https://python.langchain.com/docs/integrations/retrievers/ragatouille
================================================================================
"""

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()


# ==============================================================================
# 01. ColBERT 核心概念图解
# ==============================================================================
"""
【单向量 Embedding 的问题】

  假设查询 = "Python 内存管理"

  text-embedding-v4 的做法：
    把整句变成一个向量 [0.1, -0.3, 0.7, ...]
    在向量库里找最近的文档向量

  问题场景：
    文档 A: "Python 的内存管理使用引用计数 + GC"
    文档 B: "Java 的内存管理使用分代回收算法"
    文档 C: "Python 是一种解释型编程语言"

  单向量编码后：
    Doc A 向量 ≈ Doc C 向量（因为都和 "Python" 相关）
    Doc B 向量 ≈ 离得远

  但如果用户真正关心的是"内存管理机制"（而不是"Python"），
  单向量可能召回 Doc C（讲 Python 但不讲内存），漏掉 Doc A（讲 Python 内存）。

【ColBERT 的做法】

  查询 token 化: ["Python", "内存", "管理"]
  每个 token 独立编码:
    v_Python = [0.22, -0.11, 0.43, ...]   ← 在上下文中理解 Python
    v_内存   = [0.15, 0.67, -0.23, ...]   ← 在上下文中理解 内存
    v_管理   = [-0.08, 0.31, 0.55, ...]   ← 在上下文中理解 管理

  文档 A 也 token 化独立编码:
    v_Python_A = [0.21, -0.10, 0.44, ...]
    v_的_A     = [...]
    v_内存_A   = [0.14, 0.68, -0.22, ...]
    v_管理_A   = [-0.07, 0.30, 0.56, ...]
    v_使用_A   = [...]
    ...（每个 token 都保留完整语义）

  MaxSim 计算：
    对查询的 v_Python：找文档中所有 token 与 v_Python 的最大 cos 相似度
      → v_Python_A 最匹配 (0.99)，得分为 0.99

    对查询的 v_内存：找文档中所有 token 与 v_内存 的最大 cos 相似度
      → v_内存_A 最匹配 (0.98)，得分为 0.98

    对查询的 v_管理：找文档中所有 token 与 v_管理 的最大 cos 相似度
      → v_管理_A 最匹配 (0.97)，得分为 0.97

    总分 = 0.99 + 0.98 + 0.97 = 2.94

  而文档 C ("Python 是一种解释型编程语言")：
    只有 v_Python 匹配，v_内存 和 v_管理 都找不到对应
    → 总分低很多

  结果：ColBERT 能区分 "Python 内存管理" 和 "Python 简介"，
       而单向量 Embedding 很难做到这种细粒度区分。

【存储代价】

  单向量：       1 个向量 / 文档片段 =        1,536 float
  ColBERTv2：    最多 128 个向量 / 文档片段 = 128 × 128 = 16,384 float
  存储大约是 11x，但换来的是 token 级别的精确匹配能力。

  ColBERT 使用 PLAID（Efficient Passage Retrieval）索引：
    - 对所有文档 token 的向量做 k-means 聚类（1024 个中心）
    - 每个 token 向量用"最近的聚类中心 ID + 残差"存储
    - 检索时先粗筛（聚类中心匹配），再精排（残差计算精确距离）
"""


# ==============================================================================
# 02. 环境检查 & 安装提示
# ==============================================================================
def check_ragatouille():
    """检查 ragatouille 是否已安装，未安装则提示。"""
    try:
        import ragatouille  # noqa: F401
        return True
    except ImportError:
        print("=" * 60)
        print("  ragatouille 未安装")
        print("  请运行: pip install ragatouille")
        print("=" * 60)
        print()
        print("  ragatouille 依赖 PyTorch，安装可能较慢（~2GB）。")
        print("  如果只是学习 ColBERT 原理，阅读本文件的注释和文档即可。")
        print()
        return False


HAS_RAGATOUILLE = check_ragatouille()


# ==============================================================================
# 03. 加载 ColBERT 模型（核心）
# ==============================================================================
"""
ColBERTv2 是 ColBERT 的改进版本，主要优化：
  - 使用 KL 散度做知识蒸馏，用大 Cross-Encoder 教 ColBERT
  - 使用更高效的索引架构（PLAID）
  - 去噪训练：hard negative mining

RAGPretrainedModel.from_pretrained() 会：
  1. 从 HuggingFace 下载 colbert-ir/colbertv2.0 权重（~440MB）
  2. 加载到 PyTorch（CPU 或 GPU）
  3. 预热的 tokenizer

首次运行需要下载模型，之后缓存到本地。
"""

RAG = None  # 延迟初始化

if HAS_RAGATOUILLE:
    from ragatouille import RAGPretrainedModel

    print("=" * 60)
    print("【ColBERT 模型加载】")
    print("=" * 60)
    print("  正在加载 colbert-ir/colbertv2.0 ...")
    print("  (首次使用会从 HuggingFace 下载 ~440MB 模型)")

    # 实际使用时取消注释：
    # RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    # print("  模型加载完成！")
    # print(f"  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    print("  [演示模式] 模型未实际加载，取消注释可启用")
    print()


# ==============================================================================
# 04. 准备文档 & 构建 ColBERT 索引
# ==============================================================================
"""
ColBERT 的索引流程与普通向量库完全不同：

  普通 ChromaDB：
    texts → Embedding API → 每个 text 生成 1 个向量 → 存入 ChromaDB

  ColBERT（ragatouille）：
    texts → ColBERT 模型（本地）→ 每个 text 生成 N 个向量（每个 token 一个）
         → PLAID 索引（k-means 聚类 + 残差压缩）→ 存入磁盘

  RAG.index() 参数说明：
    - collection:      文档列表（str 或 list[str]）
    - index_name:      索引名称，会创建 .ragatouille/colbert/indexes/<name>/ 目录
    - max_document_length: 每个片段的最大 token 数，超过的会再切分
    - split_documents: 是否自动切分长文档
    - use_faiss:       是否使用 FAISS（更快但安装麻烦），默认 False 用 PLAID
"""


def get_sample_documents():
    """准备示例文档（中文技术文档片段）。"""
    docs = [
        # RAG 相关
        """RAG（Retrieval-Augmented Generation）是一种将检索系统与生成模型结合的架构。
        在 RAG 中，用户查询首先被用于从知识库中检索相关文档，然后这些文档和原始查询一起
        被送入大语言模型生成回答。RAG 有效解决了 LLM 的知识截止日期问题和幻觉问题。""",

        # Embedding 相关
        """文本 Embedding 是将文本转换为固定维度向量的技术。常用的 Embedding 模型包括
        OpenAI 的 text-embedding-3-large、BGE-M3、以及 E5 系列。这些模型通过对比学习
        在大规模文本对上训练，使得语义相似的文本在向量空间中距离更近。""",

        # ColBERT 相关
        """ColBERT（Contextualized Late Interaction over BERT）是一种多向量检索模型。
        与传统的单向量 Embedding 不同，ColBERT 为每个 token 生成独立向量，在检索时使用
        MaxSim（Maximum Similarity）计算查询和文档之间的得分。这种方式保留了细粒度的
        语义信息，显著提升了检索精度，尤其在需要精确匹配的场景中。""",

        # Chunk 策略
        """文档切分（Chunking）是 RAG 系统中的关键预处理步骤。常见的切分策略包括：
        固定长度切分（按 token 数）、语义切分（按段落或句子）、以及递归切分（逐步拆分）。
        Chunk 大小的选择直接影响检索效果：太小则上下文不完整，太大则检索精度下降。
        实践中通常使用 256-512 token 作为平衡点。""",

        # 向量检索
        """向量检索使用近似最近邻（ANN）算法在海量向量中快速找到最相似的 K 个结果。
        常用算法包括 HNSW（Hierarchical Navigable Small World）和 IVF（Inverted File）。
        ChromaDB 底层使用 HNSW 算法，而 ColBERT 的 PLAID 使用 IVF 变体。
        ANN 检索以少量精度损失换取巨大的速度提升。""",

        # Agent
        """AI Agent 是具备自主决策和行动能力的智能体。它使用 LLM 作为核心控制器，
        通过 Planning（规划）、Memory（记忆）、Tool Use（工具使用）三个核心组件
        来完成复杂任务。典型的 Agent 架构包括 ReAct（推理+行动循环）和 Plan-and-Execute。
        AutoGPT 和 BabyAGI 是早期 Agent 系统的代表性实现。""",
    ]
    return docs


if HAS_RAGATOUILLE and RAG is not None:
    docs = get_sample_documents()
    print("=" * 60)
    print("【构建 ColBERT 索引】")
    print("=" * 60)
    print(f"  文档数量: {len(docs)}")
    print()

    # 索引文档
    # index_path = RAG.index(
    #     collection=docs,
    #     index_name="rag_tech_docs",
    #     max_document_length=256,
    #     split_documents=True,
    # )
    # print(f"  索引路径: {index_path}")
    # print(f"  索引完成！文档已编码并存入 PLAID 索引")
    # print()


# ==============================================================================
# 05. ColBERT 检索
# ==============================================================================
"""
检索时 ColBERT 做了什么：

  1. 编码查询：将查询文本通过 ColBERT 模型 → 每个 token 生成一个向量
     查询 "什么是 RAG" → [v_什么, v_是, v_RAG]（32 个 token 的向量，含填充）

  2. 粗筛（PLAID Phase 1）：
     - 查询的每个 token 向量找最近的聚类中心
     - 这些聚类中心对应的文档片段进入候选集
     - 从可能百万级文档筛到几千个候选

  3. 精排（PLAID Phase 2）：
     - 对候选集中的每个文档片段，用完整的残差向量计算 MaxSim
     - MaxSim(doc) = Σ_{q_i ∈ query} max_{d_j ∈ doc} cos(q_i, d_j)
     - 按 MaxSim 得分排序，返回 Top-K

  4. 返回结果：包含 content、score、rank 等信息
"""

if HAS_RAGATOUILLE and RAG is not None:
    print("=" * 60)
    print("【ColBERT 检索演示】")
    print("=" * 60)

    queries = [
        "什么是 RAG 技术",
        "ColBERT 和传统 Embedding 有什么区别",
        "文档切分有哪些策略",
    ]

    for query in queries:
        print(f"\n  查询: \"{query}\"")
        # results = RAG.search(query=query, k=2)
        # for r in results:
        #     print(f"    [rank={r['rank']}] score={r['score']:.2f}")
        #     print(f"      {r['content'][:100]}...")
        print("    [演示模式] RAG.search() 未实际执行")
    print()


# ==============================================================================
# 06. 作为 LangChain Retriever 使用
# ==============================================================================
"""
ColBERT 可以通过 RAG.as_langchain_retriever() 直接集成到 LangChain 的 RAG 链路中：

  from ragatouille import RAGPretrainedModel

  RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
  RAG.index(collection=docs, index_name="my_index", split_documents=True)

  # 获取 LangChain 兼容的 Retriever
  retriever = RAG.as_langchain_retriever(k=3)

  # 直接在 RAG Chain 中使用
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(...)
  template = "根据以下资料回答问题：\n\n{context}\n\n问题：{question}"
  prompt = ChatPromptTemplate.from_template(template)

  from langchain_core.runnables import RunnablePassthrough
  from langchain_core.output_parsers import StrOutputParser

  def format_docs(docs):
      return "\n\n".join(d.page_content for d in docs)

  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  result = rag_chain.invoke("什么是 RAG?")

这样就完整地将 ColBERT 的精确检索能力集成到了 RAG 流水线中。
"""


# ==============================================================================
# 07. ColBERT vs 本项目其他检索方案对比
# ==============================================================================
"""
┌──────────────────────┬──────────────┬──────────────────┬───────────────────────┐
│ 方案                  │ 存储复杂度    │ 检索精度           │ 适用场景               │
├──────────────────────┼──────────────┼──────────────────┼───────────────────────┤
│ 单向量 Embedding      │ 低 (1x)      │ 中等              │ 通用 RAG，快速检索      │
│ (01-09 的基础方案)     │              │                   │                       │
├──────────────────────┼──────────────┼──────────────────┼───────────────────────┤
│ Multi-Vector Retriever│ 低 (1x)      │ 中高              │ 检索用摘要/小chunk      │
│ (10)                  │ + DocStore   │ (摘要筛选+原文返回) │ 返回完整文档            │
├──────────────────────┼──────────────┼──────────────────┼───────────────────────┤
│ RAPTOR                │ 中 (树状)    │ 中高              │ 长文档跨chunk问答       │
│ (11)                  │ 摘要+原文     │ (层级摘要聚合)     │ 需要多粒度信息          │
├──────────────────────┼──────────────┼──────────────────┼───────────────────────┤
│ ColBERT (本文件)       │ 高 (~11x)    │ 最高              │ 精确匹配、专有名词       │
│                       │ 每个token向量 │ (token级交互)      │ 法律/医疗/技术文档       │
├──────────────────────┼──────────────┼──────────────────┼───────────────────────┤
│ Cross-Encoder         │ N/A (无索引) │ 终极              │ 重排序（Rerank）        │
│ (BERT 全交互)          │ 每次推理     │ (query-doc全交互)  │ Top-K 结果精排          │
└──────────────────────┴──────────────┴──────────────────┴───────────────────────┘

【选择建议】
  - 通用场景：单向量 Embedding 足够，直接用 ChromaDB
  - 长文档问答：Multi-Vector Retriever + RAPTOR 都可以
  - 精确匹配（法规/专利/代码）：ColBERT 是很好的选择
  - 极致精度但数据量小：Cross-Encoder 做 Reranker

  实际生产中常用组合：
    单向量粗筛（ChromaDB） → Top-100
    → ColBERT 精排 → Top-10
    → Cross-Encoder 重排序 → Top-3
    → LLM 生成回答
"""


# ==============================================================================
# 08. ColBERT 的局限性
# ==============================================================================
"""
【局限性】

1. 存储开销大
   - 每个文档片段的向量数 = token 数量（最多 128）
   - 传统方案：1 个文档片段 = 1 个向量（1536 float）
   - ColBERT：1 个文档片段 ≤ 128 个向量（128 × 128 = 16384 float，约 11x）

2. 不能使用标准向量数据库
   - ChromaDB / LanceDB 假设每个 doc 1 个向量
   - ColBERT 需要 PLAID 索引或 ColBERT 专用索引
   - 不能直接替换现有向量库方案，需要额外部署

3. 需要本地推理
   - ColBERT 模型（~440MB）必须在本地运行（CPU 或 GPU）
   - 不能调用 OpenAI API 来生成 ColBERT 风格的 token 级向量
   - 因为 ColBERT 的核心是"为每个 token 保留独立向量"，API 只返回池化后的单向量

4. Windows 兼容性
   - ragatouille 主要在 Linux/Mac 测试，Windows 可能遇到 C++ 扩展编译问题
   - 可以使用 colbert-ai 直接使用（底层库），但需要更多手动工作

5. 检索速度
   - MaxSim 计算比 cos 点积慢（需要双重循环找最大相似度）
   - PLAID 索引做了很多优化，但仍然比标准向量库慢一些

【什么时候不用 ColBERT】
  - 文档量巨大（>1M）且预算有限 → 用单向量 + 重排序
  - 延迟要求极高（<50ms）→ 单向量 + HNSW 索引
  - 只需要语义级匹配（不关心精确词汇匹配）→ 单向量足够
"""


# ==============================================================================
# 09. 不使用 ragatouille 的轻量替代方案
# ==============================================================================
"""
如果你的环境无法安装 ragatouille（Windows 编译问题等），可以考虑以下替代方案：

方案 A：用已有的 Multi-Vector 思想模拟 ColBERT
  已经学过的 10_Multi_Vector_Retriever 就是 ColBERT 的"简化版"：
  - ColBERT: token 级别多向量 → MaxSim 交互
  - Multi-Vector: 段落级别多向量（摘要、小chunk、假设问题）→ 单向量匹配
  虽然不是完全一样，但都是"多表示 → 精细检索"的思路。

方案 B：两阶段检索（近似 ColBERT 效果）
  1. 单向量粗筛：embedding → ChromaDB → Top-50
  2. 关键词精排：BM25 / TF-IDF 对 Top-50 重排序
  3. 效果接近 ColBERT（但不如 ColBERT 精确）

方案 C：使用 ColBERT 的在线 API（如果有的话）
  目前主流 Embedding API 都不提供 token 级别的向量。
  这是 ColBERT 的架构决定的——必须本地编码。
"""


# ==============================================================================
# 10. ColBERT 理论深入：MaxSim 公式
# ==============================================================================
"""
【MaxSim 正式定义】

给定：
  - 查询 Q，经过 ColBERT 编码为 token 向量集合: E_q = {e_1, e_2, ..., e_N}
    N = 查询的 token 数量（通常 32）
  - 文档 D，经过 ColBERT 编码为 token 向量集合: E_d = {d_1, d_2, ..., d_M}
    M = 文档的 token 数量（最多 128）

ColBERT 的 Late Interaction 得分：

  S(Q, D) = Σ_{i=1}^{N}  max_{j=1}^{M}  e_i · d_j
             ^^^^^^^^^    ^^^^^^^^^^^^    ^^^^^^^^^^
             对每个查询     找出文档中       cos 相似度
             token 求和    最相似的那个 token   (内积)

理解：
  1. 对于查询中的每个 token 向量 e_i（如 "RAG" 的向量）
  2. 遍历文档中的所有 token 向量 d_j，找到与 e_i 内积最大的那个
     → 就是文档中与 "RAG" 最相关的那个 token
  3. 把所有查询 token 的最大相似度求和 → 文档的最终得分

【为什么叫"迟交互"（Late Interaction）】

  早交互（Early Interaction / Cross-Encoder）：
    查询 token + 文档 token → 一起送入 BERT → 每一层的注意力都在 query-doc 之间交互
    特点：最强的信号，但每对 (Q,D) 都要重新跑 BERT

  无交互（No Interaction / Bi-Encoder）：
    查询 → Encoder → 1个向量 }  两个向量 cos 一下 → 得分，简单粗暴
    文档 → Encoder → 1个向量 }
    特点：最快，但损失最多信息

  迟交互（Late Interaction / ColBERT）：
    查询 → Encoder → N个向量 }  编码阶段各自独立（快）
    文档 → Encoder → M个向量 }  打分阶段做 token 级交互（精细）
    特点：编码效率和检索精度的折中

【为什么 ColBERT 效果比单向量好那么多？】

  例子：查询 = "苹果公司的 CEO"

  单向量 Embedding：
    整句压缩为 1 个向量 → "苹果"和"CEO"的信息混在一起
    文档 "苹果是一种水果" 和 "Tim Cook 是苹果 CEO"
    两个都相似（都含"苹果"语义）→ 难以区分

  ColBERT：
    v_苹果 和 v_CEO 各自独立检索
    "苹果是一种水果"：
      MaxSim(v_苹果, doc) 高，MaxSim(v_CEO, doc) 低 → 总分一般
    "Tim Cook 是苹果 CEO"：
      MaxSim(v_苹果, doc) 高，MaxSim(v_CEO, doc) 高 → 总分更高
    → ColBERT 能区分"苹果公司的 CEO"和"水果苹果"
"""


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  ColBERT — Contextualized Late Interaction over BERT")
    print("=" * 60)
    print()
    print("  核心要点：")
    print("  1. ColBERT 为每个 token 生成独立向量（不是压缩成 1 个）")
    print("  2. 检索时用 MaxSim：查询的每个 token 找文档中最相似的 token")
    print("  3. 比单向量 Embedding 精确，比 Cross-Encoder 高效")
    print("  4. 存储开销约为单向量的 11x，需要专用索引（PLAID）")
    print("  5. 适合精确匹配场景：法规、专利、医疗、代码搜索")
    print()
    print("  与传统方案的定位：")
    print("    单向量(ChromaDB) = 粗筛 → ColBERT = 精排 → LLM = 生成")
    print()
    if not HAS_RAGATOUILLE:
        print("  [提示] ragatouille 未安装，运行概念演示模式。")
        print("  如需运行完整代码，请: pip install ragatouille")
        print()
    print("=" * 60)
    print("【运行完成】")
    print("=" * 60)


# ==============================================================================
# ColBERT · 总结
# ==============================================================================
"""
【一句话总结】
ColBERT 用"每个 token 一个向量 + 迟交互 MaxSim 打分"取代"压缩全文为单向量"，
在检索精度和计算效率之间取得了一个很好的平衡。

【核心概念】
  - 多向量表示：每个 token 独立编码，保留细粒度语义
  - 迟交互（Late Interaction）：编码独立，打分交互
  - MaxSim：每个查询 token 找文档中"最像自己"的 token，求和得分
  - PLAID 索引：k-means 聚类 + 残差压缩，实现高效近似检索

【与已有知识的关系】
  - 与 Multi-Vector Retriever (10) 同属"多向量"思路，但粒度不同
    Multi-Vector：段落级多表示（摘要/小chunk/假设问题）
    ColBERT：token 级多向量（每个 token 一个向量）
  - 与 RAPTOR (11) 互补：
    RAPTOR 解决"跨 chunk 的全局理解"（纵向聚合）
    ColBERT 解决"token 级别的精确匹配"（横向展开）
  - 可以作为 HyDE (07) 的底层检索器：
    HyDE 生成假设文档 → 用 ColBERT 精确检索

【动手建议】
  1. 先理解原理（本文件注释 + 论文摘要）
  2. 有 GPU 环境再尝试安装 ragatouille 跑完整代码
  3. 思考：你的业务场景是真的需要 token 级精确匹配，
     还是单向量 + 好的 chunk 策略就足够了？
"""
