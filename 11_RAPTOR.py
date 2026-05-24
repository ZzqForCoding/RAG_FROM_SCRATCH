"""
================================================================================
Part 11: RAPTOR（递归抽象处理 · 树组织检索）
================================================================================
【核心问题】
传统 RAG 有一个根本性缺陷：文档被切成小块后，全局信息丢失。
  - 用户问"这本书的核心思想是什么？" → 没有任何一个 chunk 能回答
  - 用户问"第三章和第五章有什么联系？" → 知识分散在不同 chunk 中，检索不到

RAPTOR 通过"自底向上构建摘要树"来解决这个问题：
  - 叶子节点 = 原始文档 chunks
  - 中间节点 = LLM 对各簇 chunks 的摘要
  - 根节点 = 最高层摘要（全局视角）

检索时可以在任意层级命中——既能回答细节问题，也能回答宏观问题。

【RAPTOR 论文】
  Recursive Abstractive Processing for Tree-Organized Retrieval
  https://arxiv.org/pdf/2401.18059.pdf

【核心流程】
  原始文档 chunks（底层，细节丰富）
      ↓ UMAP 降维 + GMM 聚类
  聚类成 N 个语义组
      ↓ LLM 对每组生成摘要
  中间层 summaries（中层，主题概括）
      ↓ 再次 UMAP + 聚类 + 摘要
  更高层 summaries
      ↓ ... 递归直到只剩 1 个簇
  根节点 summary（顶层，全局视角）

  → 所有节点一起存入向量库（Collapsed Tree Retrieval）

【关键设计点】
  1. GMM 软聚类（不是 K-Means 硬聚类）
     - 一个文本可以属于多个簇（threshold=0.1）
     - 避免边界文档被强制归入某一类
  2. UMAP 降维
     - 高维 Embedding 中距离度量失效（维数灾难）
     - 先降维再聚类，聚类效果更好
  3. BIC 自动确定簇数
     - 不需要人工设定每层分几个簇
     - Bayesian Information Criterion 自动找最优值

【适用场景】
  ✅ 长文档（书、报告、论文）的综合性问答
  ✅ 需要跨章节推理的问题
  ✅ 宏观总结 + 细节定位混合型应用
  ❌ 文档很短、chunks 很少（聚类没有意义）
  ❌ 预算极度有限（LLM 摘要 + 多次 Embedding 成本高）

【参考资料】
  - RAPTOR 论文: https://arxiv.org/pdf/2401.18059.pdf
  - LangChain Cookbook: https://github.com/langchain-ai/langchain/blob/5656702b8dea5b008d8026b30274b23f23bdc041/cookbook/RAPTOR.ipynb
  - 本文件代码基于 LangChain Cookbook 改写，适配项目统一配置
================================================================================
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*", category=UserWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ==============================================================================
# 依赖检查：umap-learn 和 scikit-learn 不是 LangChain 的默认依赖
# ==============================================================================
try:
    import umap
except ImportError:
    print("=" * 60)
    print("[ERROR] 缺少 umap-learn 依赖，请执行以下命令安装：")
    print("  pip install umap-learn scikit-learn")
    print("=" * 60)
    raise

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    print("=" * 60)
    print("[ERROR] 缺少 scikit-learn 依赖，请执行以下命令安装：")
    print("  pip install scikit-learn")
    print("=" * 60)
    raise

load_dotenv()

# ==============================================================================
# 01. 配置 Embedding 和 LLM
# ==============================================================================
class BatchedEmbeddings:
    """
    OpenAIEmbeddings 的包装器，自动将 embed_documents 分批调用。

    text-embedding-v4 限制每批最多 10 条文本，Chroma.from_texts() 等
    LangChain 内置方法内部会调用 embed_documents()，一次性传入所有文本。
    这个包装器拦截调用并自动拆分为 ≤10 的批次，避免 API 400 错误。
    """

    def __init__(self, batch_size: int = 10):
        self._inner = OpenAIEmbeddings(
            model="text-embedding-v4",
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_BASE"),
            check_embedding_ctx_length=False,
        )
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._inner.embed_documents(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def __getattr__(self, name):
        # 透传其他属性/方法到内部 embeddings 对象
        return getattr(self._inner, name)


embeddings = BatchedEmbeddings(batch_size=10)

llm = ChatOpenAI(
    model="deepseek-v3.2",
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
)

RANDOM_SEED = 224  # 固定随机种子，保证可复现

# ==============================================================================
# 02. 构建模拟文档集（替代爬取 LCEL 文档，避免网络依赖）
# ==============================================================================
"""
为了演示 RAPTOR 的递归摘要能力，我们构造一组覆盖 4 个主题领域的文档片段：
  - RAG 检索增强生成（4 篇）
  - LangChain/LCEL 链式调用（3 篇）
  - Agent / Tool Use（3 篇）
  - Embedding / 向量化（2 篇）

这些文档在语义空间自然形成 4 个簇，RAPTOR 的聚类算法应该能自动发现它们。
"""

chunk_docs = [
    # ===== RAG 检索增强生成（簇1） =====
    Document(
        page_content=(
            "RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识库与大型语言模型"
            "结合的架构。它通过在生成回答之前先从向量数据库中检索相关文档，将检索到的内容作为上下文"
            "注入到 LLM 的 prompt 中，从而有效缓解 LLM 的知识截止问题和幻觉问题。"
            "RAG 的典型流程包含两个阶段：索引（Indexing）和检索生成（Retrieval & Generation）。"
            "索引阶段将文档切片、嵌入并存入向量数据库；检索生成阶段接收用户查询，通过语义相似度"
            "搜索找到最相关的文档片段，然后将它们与用户问题一起发送给 LLM 生成回答。"
        ),
        metadata={"topic": "rag", "doc_id": "rag_1"},
    ),
    Document(
        page_content=(
            "RAG 系统有多种高级变体。Multi-Query RAG 通过将原始问题改写为多个不同表述的查询，"
            "从多个角度检索同一知识库，然后合并去重，提高召回覆盖率。RAG-Fusion 在 Multi-Query "
            "基础上加入倒数排名融合（RRF）算法，对不同查询的检索结果进行加权重排序。"
            "Step-Back Prompting 则在检索前先提出一个更宏观的问题，用更通用的知识辅助检索。"
            "HyDE（Hypothetical Document Embeddings）不直接用用户问题进行检索，而是先用 LLM "
            "生成一个假设的答案文档，用这个假设文档的 Embedding 去检索——利用了'文档匹配文档'"
            "比'问题匹配文档'更准确的特性。"
        ),
        metadata={"topic": "rag", "doc_id": "rag_2"},
    ),
    Document(
        page_content=(
            "构建生产级 RAG 系统的关键在于索引质量和检索策略。文档分块（Chunking）的粒度直接影响"
            "检索精度——分块太大会引入噪声，分块太小会丢失上下文。多向量检索器（Multi-Vector "
            "Retriever）通过'小表示做索引，大文档做上下文'的方式解决这一困境：将文档摘要存入向量库"
            "用于检索匹配，匹配后返回原始完整文档给 LLM。元数据过滤（Metadata Filtering）允许在"
            "语义检索之上叠加业务约束，如时间范围、分类标签、权限级别等。"
        ),
        metadata={"topic": "rag", "doc_id": "rag_3"},
    ),
    Document(
        page_content=(
            "Self-RAG 和 CRAG（Corrective RAG）是两种自反思的 RAG 变体。Self-RAG 让 LLM "
            "决定是否需要检索、何时检索以及如何评价自己生成的内容质量。CRAG 在检索结果不够相关时"
            "自动进行二次检索，修正检索方向。这两种技术都可以通过 LangGraph 这样的状态图框架来"
            "实现，将检索、生成、反思、修正组织为一个有向无环图（DAG）。与标准 RAG 相比，"
            "Self-RAG/CRAG 在需要多步推理和知识验证的场景下表现更好。"
        ),
        metadata={"topic": "rag", "doc_id": "rag_4"},
    ),

    # ===== LangChain / LCEL（簇2） =====
    Document(
        page_content=(
            "LangChain 是一个用于构建 LLM 驱动应用的开源框架。它的核心抽象包括：Prompt（提示模板）、"
            "Model（LLM 接口）、Chain（组件串联）、Retriever（检索器）、Agent（自主决策）和 "
            "Memory（对话记忆）。每个抽象都是独立的组件，可以通过管道操作符 '|'（LCEL）灵活组合。"
            "LCEL（LangChain Expression Language）是一种声明式语言，用于将 LangChain 组件"
            "组合成可运行的链。它的语法类似于 Unix 管道：prompt | llm | output_parser，"
            "输出自动从上游组件传递给下游组件。"
        ),
        metadata={"topic": "langchain", "doc_id": "lc_1"},
    ),
    Document(
        page_content=(
            "LCEL 的核心优势在于它的同步/异步/流式支持是自动继承的——定义一次链，自动获得 "
            ".invoke()、.ainvoke()、.stream()、.batch() 四种调用模式。RunnablePassthrough "
            "用于透传数据，RunnableLambda 用于插入自定义函数，RunnableParallel 用于并行执行"
            "多个分支。itemgetter 和 RunnableMap 可以精确控制数据流中每个字段的来源和走向。"
            "这些原语使得复杂的 RAG 管道（如查询翻译、路由、多路融合）可以用极少的代码实现。"
        ),
        metadata={"topic": "langchain", "doc_id": "lc_2"},
    ),
    Document(
        page_content=(
            "LangChain 的 with_structured_output() 方法让 LLM 按 Pydantic 模型的结构输出 JSON。"
            "这是实现查询分析（Query Analysis）和路由（Routing）的关键——让 LLM 判断用户意图并"
            "输出结构化的路由决策（如 {'datasource': 'python_docs'}）。在 RAPTOR 中，我们也用到"
            "了 LangChain 的链式调用和 Prompt 模板来构建递归摘要管道。LangChain Hub 提供了大量"
            "社区维护的 Prompt 模板，可以直接用 hub.pull() 拉取使用。"
        ),
        metadata={"topic": "langchain", "doc_id": "lc_3"},
    ),

    # ===== Agent / Tool Use（簇3） =====
    Document(
        page_content=(
            "AI Agent 是一种能够自主决策、使用工具并与环境交互的 AI 系统。与传统的'输入-输出'"
            "LLM 使用模式不同，Agent 遵循'思考-行动-观察'（Thought-Action-Observation）循环："
            "它首先分析当前状态和任务，然后决定调用哪个工具或执行哪个操作，接着观察操作结果，"
            "根据结果更新自己的认知并决定下一步行动，直到任务完成。ReAct（Reasoning + Acting）"
            "和 Plan-and-Execute（先规划再执行）是两种主流的 Agent 架构范式。"
        ),
        metadata={"topic": "agent", "doc_id": "agent_1"},
    ),
    Document(
        page_content=(
            "工具调用（Tool Calling / Function Calling）是 Agent 能力的核心。LLM 通过工具描述"
            "来理解每个工具的用途和参数，在需要时生成符合工具 schema 的 JSON 调用。OpenAI、"
            "Anthropic 和开源的 DeepSeek 等模型都支持 Function Calling。在 LangChain 中，"
            "可以用 @tool 装饰器将任意 Python 函数包装为 Tool 对象，然后通过 bind_tools() "
            "绑定到 LLM。Agent Executor 负责解析 LLM 输出的工具调用、执行工具、将结果反馈给 LLM "
            "并管理整个对话循环。"
        ),
        metadata={"topic": "agent", "doc_id": "agent_2"},
    ),
    Document(
        page_content=(
            "多 Agent 系统（Multi-Agent Systems）将复杂任务分解给多个专业 Agent 协作完成。"
            "每个 Agent 有自己的工具集、知识范围和角色定位。LangGraph 是实现多 Agent 系统的"
            "理想框架——它支持有向图（DAG）、条件分支、循环和共享状态。典型的应用场景包括："
            "软件开发生命周期（一个 Agent 写代码，一个 Agent 审查，一个 Agent 写测试）、"
            "客户服务系统（路由 Agent → 专业 Agent → 汇总 Agent）。多 Agent 协作的挑战在于"
            "通信协议设计、任务分配策略和状态同步。"
        ),
        metadata={"topic": "agent", "doc_id": "agent_3"},
    ),

    # ===== Embedding / 向量化（簇4） =====
    Document(
        page_content=(
            "文本嵌入（Text Embedding）是将非结构化的自然语言文本映射到固定维度的稠密向量的技术。"
            "语义相似的文本在向量空间中距离近，反之距离远。Embedding 模型经历了从 Word2Vec、"
            "GloVe 到 BERT、Sentence-BERT 再到 OpenAI text-embedding 系列的演进。现代 Embedding "
            "模型（如 text-embedding-v4）不仅能捕捉词汇级别的相似度，还能理解段落级别的语义。"
            "Embedding 质量直接影响 RAG 系统的检索命中率——如果嵌入模型不理解领域术语的语义，"
            "再好的检索策略也无法召回正确的文档。"
        ),
        metadata={"topic": "embedding", "doc_id": "emb_1"},
    ),
    Document(
        page_content=(
            "向量数据库（Vector Database）是存储和搜索高维向量数据的专用数据库系统。"
            "常见的向量数据库包括 ChromaDB（轻量级，Python 原生）、LanceDB（基于 Lance 列式格式）、"
            "Pinecone（全托管云服务）、Weaviate（支持混合搜索）和 Qdrant（高性能 Rust 实现）。"
            "它们都支持近似最近邻搜索（ANN），常用算法包括 HNSW（分层可导航小世界图）和 IVF（倒排文件索引）。"
            "向量数据库的选择需要考虑部署方式（嵌入式 vs 服务式）、过滤能力、可扩展性和运维成本。"
            "元数据过滤是在向量搜索中叠加业务约束的关键——如只搜索 2024 年之后的文章。"
        ),
        metadata={"topic": "embedding", "doc_id": "emb_2"},
    ),
]

print("=" * 60)
print("【文档集概览】")
print("=" * 60)
topics = set(d.metadata["topic"] for d in chunk_docs)
for t in sorted(topics):
    count = sum(1 for d in chunk_docs if d.metadata["topic"] == t)
    print(f"  {t}: {count} 篇")


# ==============================================================================
# 03. RAPTOR 树构建核心算法
# ==============================================================================
"""
以下函数是 RAPTOR 的完整实现，从 Embedding 到聚类到递归摘要。

核心流程：
  embed_cluster_summarize_texts()   ← 单层：Embed → UMAP 降维 → GMM 聚类 → LLM 摘要
  recursive_embed_cluster_summarize() ← 递归：重复以上过程直到只剩 1 个簇或达到指定层数

所有函数注释翻译自 LangChain Cookbook。
"""


# ---------------------------------------------------------------------------
# 3a. UMAP 降维
# ---------------------------------------------------------------------------
def global_cluster_embeddings(
    emb: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    全局降维：将所有 Embedding 从高维空间映射到低维空间。
    高维空间中距离度量失效（维数灾难），降维后 GMM 聚类才能正常工作。

    参数:
      emb: 输入 Embedding 矩阵 (n_docs, embedding_dim)
      dim: 目标降维维度
      n_neighbors: 每个点的邻居数，默认为 sqrt(n_docs)
      metric: 距离度量，默认余弦距离（比欧氏距离更适合文本 Embedding）
    """
    if n_neighbors is None:
        n_neighbors = int((len(emb) - 1) ** 0.5)
    n_neighbors = max(n_neighbors, 2)  # UMAP 要求 n_neighbors >= 2
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED,
    ).fit_transform(emb)


def local_cluster_embeddings(
    emb: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    局部降维：对全局聚类后的每个簇内部做更精细的降维。
    局部 UMAP 可以更好地保留簇内结构，使第二次 GMM 聚类更准确。
    """
    num_neighbors = max(num_neighbors, 2)
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED,
    ).fit_transform(emb)


# ---------------------------------------------------------------------------
# 3b. GMM 聚类（自动确定簇数）
# ---------------------------------------------------------------------------
def get_optimal_clusters(
    emb: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    使用 BIC（Bayesian Information Criterion）自动确定最优簇数。

    BIC 同时考虑模型拟合度和复杂度：
      → 拟合度越高越好（聚类越紧凑越好）
      → 复杂度越低越好（簇数越少越好）
      → 选择 BIC 最小的 n（最佳平衡点）

    比 K-Means 的 Elbow Method 更严谨，不需要人工看图判断拐点。
    """
    max_clusters = min(max_clusters, len(emb))
    n_clusters_range = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters_range:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(emb)
        bics.append(gm.bic(emb))
    optimal = n_clusters_range[np.argmin(bics)]
    return optimal


def GMM_cluster(
    emb: np.ndarray, threshold: float, random_state: int = RANDOM_SEED
) -> Tuple[list, int]:
    """
    使用 GMM 进行软聚类。与 K-Means 不同，GMM 输出每个点属于每个簇的概率。

    软聚类的关键：
      → threshold=0.1：只要某点属于某簇的概率 > 10%，就分配给它
      → 一个文本可以同时属于多个簇（如"RAG + LangChain"同时属于两个语义组）
      → 这比 K-Means 的硬分配更合理，因为文档主题通常不是互斥的
    """
    n_clusters = get_optimal_clusters(emb)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(emb)
    probs = gm.predict_proba(emb)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


# ---------------------------------------------------------------------------
# 3c. 两阶段聚类（全局 → 局部）
# ---------------------------------------------------------------------------
def perform_clustering(
    emb: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    两阶段聚类：全局聚类 → 对每个簇内部做局部聚类。

    为什么需要两阶段？
      - 全局聚类先找出大的主题分组（如 RAG vs Agent vs LangChain）
      - 局部聚类在每个大主题内发现更细的结构（如 RAG 内部分为基础/高级/自反思）
      - 一次聚类容易把细粒度差异抹平，两次聚类保留层级结构
    """
    if len(emb) <= dim + 1:
        return [np.array([0]) for _ in range(len(emb))]

    # 第一阶段：全局降维 + 全局聚类
    reduced_global = global_cluster_embeddings(emb, dim)
    global_clusters, n_global = GMM_cluster(reduced_global, threshold)

    all_local = [np.array([]) for _ in range(len(emb))]
    total = 0

    # 第二阶段：对每个全局簇内部做局部降维 + 局部聚类
    for i in range(n_global):
        mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embs = emb[mask]

        if len(global_cluster_embs) == 0:
            continue
        if len(global_cluster_embs) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embs]
            n_local = 1
        else:
            reduced_local = local_cluster_embeddings(global_cluster_embs, dim)
            local_clusters, n_local = GMM_cluster(reduced_local, threshold)

        # 为全局簇内的每个文档分配局部簇 ID（全局唯一编号）
        for j in range(n_local):
            local_mask = np.array([j in lc for lc in local_clusters])
            local_embs = global_cluster_embs[local_mask]
            # 在原始 embeddings 中定位这些文档的索引
            for local_emb in local_embs:
                distances = np.linalg.norm(emb - local_emb, axis=1)
                idx = int(np.argmin(distances))
                all_local[idx] = np.append(all_local[idx], j + total)

        total += n_local

    return all_local


# ---------------------------------------------------------------------------
# 3d. Embedding + 聚类 + 摘要（单层）
# ---------------------------------------------------------------------------
def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    """
    对一个文本列表执行 Embedding → 聚类，返回包含文本、Embedding、簇标签的 DataFrame。
    """
    # 分批调用，text-embedding-v4 限制每批最多 10 条
    BATCH_SIZE = 10
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embs.extend(embeddings.embed_documents(batch))
    text_embeddings = np.array(all_embs)
    cluster_labels = perform_clustering(text_embeddings, dim=10, threshold=0.1)
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings)
    df["cluster"] = cluster_labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    将 DataFrame 中同一个簇的所有文本拼接为一个字符串，用分隔符隔开。
    这个拼接后的字符串会作为 LLM 摘要的输入。
    """
    unique_txt = df["text"].tolist()
    return "\n---\n".join(unique_txt)


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    单层 RAPTOR 步骤：Embed → 聚类 → 摘要。

    返回值:
      df_clusters: 原始文本 + Embedding + 簇标签
      df_summary: 每个簇的摘要 + 层级 + 簇 ID
    """
    # Step 1: Embed + 聚类
    df_clusters = embed_cluster_texts(texts)

    # Step 2: 展开 DataFrame（一个文档可能属于多个簇）
    expanded = []
    for _, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded.append({
                "text": row["text"],
                "embd": row["embd"],
                "cluster": cluster,
            })
    expanded_df = pd.DataFrame(expanded)
    all_clusters = expanded_df["cluster"].unique()

    print(f"  [Level {level}] {len(texts)} 个文本 → {len(all_clusters)} 个簇, 正在生成摘要...")

    # Step 3: LLM 为每个簇生成摘要
    template = """以下是若干技术文档片段的集合。请对这些内容进行详细总结。
总结应包括：
1. 这组文档涵盖的核心主题（1-2 句）
2. 关键概念和技术术语的解释
3. 不同文档之间的逻辑关系

文档内容：
{context}

详细总结："""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted = fmt_txt(df_cluster)
        summary = chain.invoke({"context": formatted})
        summaries.append(summary)

    df_summary = pd.DataFrame({
        "summaries": summaries,
        "level": [level] * len(summaries),
        "cluster": list(all_clusters),
    })

    return df_clusters, df_summary


# ---------------------------------------------------------------------------
# 3e. 递归入口
# ---------------------------------------------------------------------------
def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    RAPTOR 的主入口：递归地对文本执行 Embedding → 聚类 → 摘要，
    上一层的摘要作为下一层的输入，直到只剩 1 个簇或达到指定层数。

    参数:
      texts: 原始文档文本列表
      level: 当前层级（从 1 开始）
      n_levels: 最多递归几层

    返回:
      {level: (df_clusters, df_summary), ...}
    """
    results = {}

    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    results[level] = (df_clusters, df_summary)

    unique = df_summary["cluster"].nunique()
    if level < n_levels and unique > 1:
        # 将本层摘要作为下一层的输入
        new_texts = df_summary["summaries"].tolist()
        next_results = recursive_embed_cluster_summarize(new_texts, level + 1, n_levels)
        results.update(next_results)

    return results


# ==============================================================================
# 04. 构建 RAPTOR 树
# ==============================================================================
"""
将原始文档文本（page_content）输入递归管道，构建 3 层树。

Attention: 这一步会调用 Embedding API + LLM API，产生费用。
n_levels=3 含义：
  Level 1: 原始 12 个 chunks → 聚类 → 约 3-5 个摘要
  Level 2: 3-5 个摘要 → 聚类 → 约 1-2 个摘要
  Level 3: 1-2 个摘要 → 可能只剩 1 个簇，提前终止
"""


def build_raptor_tree(
    leaf_texts: list[str], n_levels: int = 3
) -> dict:
    """
    从叶子文本构建 RAPTOR 多层摘要树。

    返回值:
      {1: (df_clusters, df_summary), 2: (...), ...}
    """
    print("\n" + "=" * 60)
    print(f"【RAPTOR 树构建】输入 {len(leaf_texts)} 篇文档, 最多 {n_levels} 层")
    print("=" * 60)
    results = recursive_embed_cluster_summarize(
        leaf_texts, level=1, n_levels=n_levels
    )
    print(f"\n[OK] 树构建完成, 共 {len(results)} 层")
    return results


# ==============================================================================
# 05. 展平树：Collapsed Tree Retrieval
# ==============================================================================
"""
RAPTOR 论文推荐用 Collapsed Tree（展平树）做检索：
  → 将所有层级的节点（原始 chunk + 各级摘要）全部放入同一个向量库
  → 检索时统一做相似度搜索
  → 既保留细节（原始 chunk），又保留全局视角（高层摘要）

对比：
  - Tree Traversal（逐层遍历）：从根节点开始，比较查询与子节点，沿最优分支向下，
    直到叶子层。缺点是"贪心"——早期选错分支就无法恢复。
  - Collapsed Tree（展平搜索）：所有节点平齐，一次搜索。简单、鲁棒，论文报告的
    最佳效果。
"""


def flatten_tree(
    results: dict, leaf_texts: list[str]
) -> list[str]:
    """
    将 RAPTOR 树的所有节点展平为一个文本列表。

    参数:
      results: 树构建结果 {level: (df_clusters, df_summary)}
      leaf_texts: 原始叶子文档文本

    返回:
      所有节点文本的列表（叶子 + 各级摘要）
    """
    all_texts = leaf_texts.copy()
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)
        print(f"  第 {level} 层添加了 {len(summaries)} 个摘要")
    print(f"  总节点数: {len(all_texts)} (原始 {len(leaf_texts)} + 摘要 {len(all_texts) - len(leaf_texts)})")
    return all_texts


# ==============================================================================
# 06. 构建向量库 + RAG 链
# ==============================================================================
def build_raptor_rag(all_texts: list[str]):
    """
    将所有节点文本存入 Chroma 向量库，构建标准 RAG 链。
    """
    print("\n" + "=" * 60)
    print("【构建 RAPTOR RAG 链】")
    print("=" * 60)
    print(f"  正在将 {len(all_texts)} 个节点 Embedding 并存入向量库...")

    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        collection_name="raptor_collapsed_tree",
        persist_directory="./chroma_raptor_storage",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print(f"  向量库构建完成, Top-K = 5")

    # RAG Prompt
    prompt = ChatPromptTemplate.from_template("""你是一个技术专家助手。请根据以下检索到的文档，回答用户的问题。
如果文档中没有相关信息，请诚实说明你不知道。

检索到的文档：
{context}

用户问题：{question}

请用中文回答：""")

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return vectorstore, rag_chain


# ==============================================================================
# 07. 对比演示：RAPTOR vs 普通 RAG
# ==============================================================================
def demo_comparison(rag_chain, leaf_texts):
    """
    用同一个需要"跨文档综合理解"的问题，对比 RAPTOR 和普通 RAG 的效果。
    """
    print("\n" + "=" * 60)
    print("【检索对比 · RAPTOR vs 普通 RAG】")
    print("=" * 60)

    # 构建普通 RAG（只用原始 chunk）
    normal_vectorstore = Chroma.from_texts(
        texts=leaf_texts,
        embedding=embeddings,
        collection_name="normal_rag_baseline",
        persist_directory="./chroma_normal_storage",
    )
    normal_retriever = normal_vectorstore.as_retriever(search_kwargs={"k": 5})

    questions = [
        "RAG 系统有哪些不同的变体和改进方法？请全面总结。",
        "这些文档讨论的核心技术主题有哪些？它们之间有什么联系？",
    ]

    for q in questions:
        print(f"\n  问题: \"{q}\"")

        # 普通 RAG 检索
        print(f"\n  --- 普通 RAG（只检索原始 chunks）---")
        normal_docs = normal_retriever.invoke(q)
        print(f"  召回了 {len(normal_docs)} 个 chunks")
        for i, d in enumerate(normal_docs, 1):
            print(f"    [{i}] {d.page_content[:80]}...")

        # RAPTOR 检索
        print(f"\n  --- RAPTOR（原始 chunks + 多层摘要）---")
        answer = rag_chain.invoke(q)
        print(f"  回答: {answer[:300]}...")


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    import json
    import sys

    # 提取叶片文本（12 个 chunk 的 page_content）
    leaf_texts = [doc.page_content for doc in chunk_docs]

    # Step 1: 构建 RAPTOR 多层摘要树
    # 注意：此步骤产生 Embedding API + LLM API 费用
    # 如果只想测试后续流程，可以传 n_levels=1 减少成本
    results = build_raptor_tree(leaf_texts, n_levels=3)

    # Step 2: 打印树结构概览
    print("\n" + "=" * 60)
    print("【树结构概览】")
    print("=" * 60)
    for level in sorted(results.keys()):
        df_c, df_s = results[level]
        print(f"\n  Level {level}:")
        print(f"    输入 {len(df_c)} 个文本")
        print(f"    生成 {len(df_s)} 个摘要")
        for i, row in df_s.iterrows():
            print(f"    簇 {i}: {row['summaries'][:100]}...")

    # Step 3: 展平树，构建 RAG
    print("\n" + "=" * 60)
    print("【Collapsed Tree · 展平树】将所有节点放入同一向量库")
    print("=" * 60)
    all_texts = flatten_tree(results, leaf_texts)
    vectorstore, rag_chain = build_raptor_rag(all_texts)

    # Step 4: 测试对比
    demo_comparison(rag_chain, leaf_texts)

    print("\n" + "=" * 60)
    print("【运行完成】")
    print("=" * 60)


# ==============================================================================
# RAPTOR · 总结
# ==============================================================================
"""
【一句话总结】
RAPTOR 通过在文档块之上构建多层 LLM 摘要树，解决了传统 RAG 只能检索"碎片"
而无法理解"全局"的问题。

【关键组件】
  ┌──────────────────────┬──────────────────────────────────────────────┐
  │ 组件                  │ 作用                                         │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ UMAP 降维            │ 缓解高维空间距离失效，让聚类有意义            │
  │ GMM 软聚类           │ 一个文档可属多个簇，不强制互斥                │
  │ BIC 自动定簇数        │ 不让用户手动指定簇数，自动找到最优值          │
  │ LLM 摘要             │ 每层为每个簇生成精炼摘要，作为上层输入        │
  │ Collapsed Tree       │ 展平所有节点，简单高效的一次检索               │
  └──────────────────────┴──────────────────────────────────────────────┘

【RAPTOR vs 其他策略】
  ┌──────────────────────┬──────────────────┬──────────────────────────┐
  │ 策略                  │ 解决问题          │ 与 RAPTOR 的关系          │
  ├──────────────────────┼──────────────────┼──────────────────────────┤
  │ Multi-Query (03)     │ 查询角度单一       │ 互补：多角度查询 + 多层摘要│
  │ Decomposition (05)   │ 复杂问题需拆解     │ 互补：拆解问题再分别检索   │
  │ Step Back (06)       │ 问题太具体         │ 替代：RAPTOR 已含抽象层    │
  │ Multi-Vector (10)    │ 检索 vs 回答粒度   │ 互补：小表示检 + 树状索引  │
  │ RAPTOR (本文件)      │ 丢失全局/跨段信息  │ —                          │
  └──────────────────────┴──────────────────┴──────────────────────────┘

【成本估算】
  RAPTOR 的成本来自两部分：
  1. Embedding API: 所有节点（原始 chunks + 所有层摘要）的 Embedding
     对于 12 个 chunks + 约 5 个摘要 + 约 2 个摘要 ≈ 19 次 Embedding
  2. LLM API: 每层为每个簇生成摘要
     约 (4-5) + (1-2) + (1) ≈ 7-8 次 LLM 调用

  对于大型文档集（数千个 chunks），成本会显著增加。可以：
  - 调低 n_levels（3 → 2），减少递归深度
  - 增加 chunk_size，减少 chunk 数量
  - 在离线/索引阶段执行，不在查询时执行

【思考题】

1. 为什么 RAPTOR 使用 GMM 而非 K-Means？
   → GMM 是软聚类（输出概率），K-Means 是硬聚类（输出标签）。
     文档主题通常是重叠的（如一篇"RAG + LangChain"的文章同时涉及两个主题），
     硬聚类会丢失这种多属性。threshold=0.1 的低阈值保证覆盖面。

2. Collapsed Tree 比 Tree Traversal 好在哪？
   → Tree Traversal 从上往下逐层决策，但 LLM 在抽象层级可能选错分支，
     一旦选错就无法恢复。Collapsed Tree 所有节点同时被检索，不依赖决策链。
   → 对于抽象问题，高层摘要自然比底层 chunks 更相关（向量相似度更高）；
     对于细节问题，底层 chunks 更相关。不需要手动判断该查哪层。

3. UMAP 降维为什么必要？
   → 高维空间（如 1024 维 Embedding）中，所有点之间的距离趋于相等
     （距离集中现象/维数灾难）。降维到 10 维让距离重新有意义，
     GMM 聚类才能识别真实的语义分组。UMAP 比 PCA 更能保留局部结构。

4. RAPTOR 和 Multi-Vector Retriever（10）的区别？
   → Multi-Vector：每个文档一个摘要，索引与文档一一对应。
     RAPTOR：构建多层树，同一文档的内容出现在多个层级。
   → Multi-Vector 解决"检索用小块，回答用大块"。
     RAPTOR 解决"检索可以从任何抽象层级获取信息"。
   → 两者可以结合：Multi-Vector 中用小表示做索引，这个小表示可以来自 RAPTOR 树。

5. 生产环境如何评估 RAPTOR 构建质量？
   → 检查每层摘要是否保留了原始文档的关键信息（信息保真度）。
   → 检查聚类质量：同一簇内的文档是否语义一致，不同簇之间是否差异明显。
   → 对比 RAPTOR 和普通 RAG 在综合性问题上的回答完整性和准确性。
"""
