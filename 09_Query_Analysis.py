"""
================================================================================
Part 09: Structured Query Analysis（结构化查询分析）
================================================================================
【核心问题】
之前的 Metadata Filtering 中，过滤条件是人工硬编码的：
  filter_dict = {"year": {"$gte": 2024}}

实际业务中，过滤条件来自用户输入：
  "帮我看下 2023 年之后、少于 5 分钟的视频，要讲 RAG 的"

我们需要把自然语言 → 结构化过滤参数。这就是 Query Analysis 要解决的问题。

【Query Analysis vs Metadata Filtering 的关系】
  Metadata Filtering 解决的是"怎么用 filter 检索"，
  Query Analysis   解决的是"用户的话翻译成什么 filter"。
  两者配合形成完整链路：用户自然语言 → 结构化查询参数 → 带 filter 的向量检索。

【完整流程】
  用户: "rag from scratch videos under 5 minutes"
    ↓
  [Query Analyzer] LLM 结构化输出
    → TutorialSearch(content_search="rag from scratch",
                     max_length_sec=300,
                     ...)
    ↓
  [Metadata Filtering] 向量检索 + filter 条件
    → 返回符合条件的视频片段

【适用场景】
  ✅ 自然语言搜索（用户不想填表单，只想说话）
  ✅ 多字段联合筛选（关键词 + 时间 + 数值范围混合）
  ✅ 电商/视频/文档等多维过滤搜索
  ✅ 需要从用户问题中提取结构化参数的 Agent

【参考资料】
  - LangChain Query Analysis:
    https://python.langchain.com/docs/tutorials/qa_chat_history/
  - LangChain YouTube Loader:
    https://python.langchain.com/docs/integrations/document_loaders/youtube
  - LangChain Structured Output:
    https://python.langchain.com/docs/how_to/structured_output/
================================================================================
"""

import datetime
import os
import warnings
from typing import Literal, Optional

from dotenv import load_dotenv

# 过滤掉 LangChain 的弃用警告和 beta 警告，终端更干净
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*", category=UserWarning)

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 .env 环境变量
load_dotenv()

# 公共 Embedding 和 LLM 实例
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
# 01. 加载 YouTube 视频数据（首次运行会调用 YouTube API）
# ==============================================================================
"""
这里我们加载 LangChain 官方的一个 RAG 教程视频。
`add_video_info=True` 会从 YouTube API 获取标题、描述、播放量等元数据。

每个 Document 的结构：
  - page_content: 视频字幕/转译文本
  - metadata: {
      'source': 'video_id',
      'title': '视频标题',
      'description': '视频描述',
      'view_count': 播放量,
      'publish_date': 发布日期,
      'length': 视频秒数,
      ...
    }
"""

VIDEO_URL = "https://www.youtube.com/watch?v=pbAd8O1Lvm4"
PERSIST_DIR = "./chroma_youtube_storage"


def create_mock_video_docs() -> list[Document]:
    """
    当 YouTube API 不可用时（pytube 经常因 YouTube 接口变更而报 HTTP 400），
    使用模拟数据来演示完整的元数据结构和结构化查询分析的流程。

    模拟数据的 metadata 字段结构基于真实 YouTube Loader 输出：
      - source:        视频 ID
      - title:         视频标题
      - description:   视频描述
      - view_count:    播放量（int）
      - thumbnail_url: 缩略图 URL
      - publish_date:  发布日期（str，格式 "YYYY-MM-DD HH:MM:SS"）
      - length:        视频时长（int，秒）
      - author:        作者/频道名

    第一个视频的 metadata 来自 pbAd8O1Lvm4 的真实数据。
    """
    print("[INFO] YouTube API 不可用，使用模拟视频数据。")
    return [
        Document(
            page_content=(
                "Self-reflective RAG is an advanced technique that allows LLMs to "
                "reflect on their own outputs and retrieval results. With LangGraph, "
                "we can implement Self-RAG and CRAG (Corrective RAG) patterns. "
                "Self-RAG enables the model to decide whether to retrieve, and to "
                "critique its own generations. CRAG adds a corrective step that "
                "re-retrieves when the initial results are irrelevant."
            ),
            metadata={
                "source": "pbAd8O1Lvm4",
                "title": "Self-reflective RAG with LangGraph: Self-RAG and CRAG",
                "description": "Unknown",
                "view_count": 11922,
                "thumbnail_url": "https://i.ytimg.com/vi/pbAd8O1Lvm4/hq720.jpg",
                "publish_date": "2024-02-07 00:00:00",
                "length": 1058,
                "author": "LangChain",
            },
        ),
        Document(
            page_content=(
                "LangChain provides a powerful framework for building LLM applications. "
                "It offers chains, agents, and retrieval strategies out of the box. "
                "The ChatLangChain feature lets you build conversational AI that can "
                "interact with your documents in real time. RAG applications built with "
                "LangChain can be deployed as REST APIs or integrated into existing systems."
            ),
            metadata={
                "source": "2TJxQj6n4eA",
                "title": "Chat LangChain: Building Conversational RAG",
                "description": "How to build a chat interface on top of your RAG pipeline.",
                "view_count": 18500,
                "thumbnail_url": "https://i.ytimg.com/vi/2TJxQj6n4eA/hq720.jpg",
                "publish_date": "2023-11-15 00:00:00",
                "length": 920,
                "author": "LangChain",
            },
        ),
        Document(
            page_content=(
                "Multi-modal models can process both text and images. When building "
                "an agent with multi-modal capabilities, you need to handle different "
                "types of inputs. The agent should decide when to use vision models "
                "versus text-only models based on the task at hand. GPT-4V and Claude "
                "both support multi-modal inputs natively."
            ),
            metadata={
                "source": "multi_modal_001",
                "title": "Multi-Modal Agents: Beyond Text",
                "description": "Building agents that can see and read.",
                "view_count": 8500,
                "thumbnail_url": "https://i.ytimg.com/vi/multi_modal_001/hq720.jpg",
                "publish_date": "2024-06-10 00:00:00",
                "length": 300,
                "author": "AI Tutorials",
            },
        ),
        Document(
            page_content=(
                "Advanced RAG techniques include query translation, routing, "
                "and metadata filtering. Query translation helps reformulate user "
                "questions for better retrieval. Routing directs queries to the right "
                "knowledge base. These techniques together form a production-ready RAG "
                "system that can handle complex enterprise use cases."
            ),
            metadata={
                "source": "adv_rag_002",
                "title": "Advanced RAG Techniques in 2024",
                "description": "Deep dive into production RAG patterns.",
                "view_count": 32000,
                "thumbnail_url": "https://i.ytimg.com/vi/adv_rag_002/hq720.jpg",
                "publish_date": "2024-09-01 00:00:00",
                "length": 1500,
                "author": "RAG Experts",
            },
        ),
    ]


def load_youtube_video(video_url: str = VIDEO_URL) -> list[Document]:
    """
    加载 YouTube 视频，返回包含转译文本和元数据的 Document 列表。

    注意：pytube 库经常因 YouTube 接口变更而报 HTTP 400 错误。
    这不是代理问题，是 pytube 的已知兼容性问题。
    遇到错误时自动降级为模拟数据，不影响后续演示。
    """
    print(f"[INFO] 正在加载 YouTube 视频: {video_url}")
    try:
        docs = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=True,
        ).load()
        print(f"  加载完成，共 {len(docs)} 个 Document")
        return docs
    except Exception as e:
        print(f"  [WARN] YouTube 加载失败: {type(e).__name__}: {e}")
        return create_mock_video_docs()


# ==============================================================================
# 02. 定义结构化查询的 Schema（TutorialSearch）
# ==============================================================================
"""
这是 Query Analysis 的核心：定义一个 Pydantic 模型来描述"用户问题可能包含哪些过滤参数"。
每个字段对应一个过滤维度，LLM 会自动从用户输入中提取对应值。

字段说明：
  - content_search: 应用于视频转译文本的相似度搜索查询（主要检索词）
  - title_search:   应用于视频标题的搜索查询（精简关键词版）
  - min_view_count / max_view_count: 播放量范围过滤
  - earliest_publish_date / latest_publish_date: 发布日期范围过滤
  - min_length_sec / max_length_sec: 视频时长范围过滤（秒）
"""


class TutorialSearch(BaseModel):
    """将自然语言查询转换为结构化的教程视频搜索参数。"""

    content_search: str = Field(
        ...,
        description="应用于视频转译文本的相似度搜索查询。",
    )
    title_search: str = Field(
        ...,
        description=(
            "应用于视频标题的替代搜索查询。"
            "应该简洁，只包含可能出现在视频标题中的关键词。"
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="最低播放量过滤（包含）。仅在用户明确指定时使用。",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="最高播放量过滤（不包含）。仅在用户明确指定时使用。",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="最早发布日期过滤（包含）。仅在用户明确指定时使用。",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="最晚发布日期过滤（不包含）。仅在用户明确指定时使用。",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="最短视频时长过滤（秒，包含）。仅在用户明确指定时使用。",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="最长视频时长过滤（秒，不包含）。仅在用户明确指定时使用。",
    )

    def pretty_print(self) -> None:
        """格式化打印提取到的所有非默认字段。"""
        print("  [结构化查询参数]")
        for field_name, field_info in type(self).model_fields.items():
            value = getattr(self, field_name)
            default = field_info.default
            if value is not None and value != default:
                print(f"    {field_name}: {value}")

    def to_metadata_filter(self) -> dict:
        """
        将提取的参数转换为 ChromaDB / 向量库可用的 metadata filter。

        注意：content_search 和 title_search 用于语义检索，
        其他字段用于 metadata 过滤。
        """
        filters = []

        if self.min_view_count is not None:
            filters.append({"view_count": {"$gte": self.min_view_count}})
        if self.max_view_count is not None:
            filters.append({"view_count": {"$lt": self.max_view_count}})
        if self.earliest_publish_date is not None:
            filters.append({"publish_date": {"$gte": str(self.earliest_publish_date)}})
        if self.latest_publish_date is not None:
            filters.append({"publish_date": {"$lt": str(self.latest_publish_date)}})
        if self.min_length_sec is not None:
            filters.append({"length": {"$gte": self.min_length_sec}})
        if self.max_length_sec is not None:
            filters.append({"length": {"$lt": self.max_length_sec}})

        if len(filters) == 0:
            return {}
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}


# ==============================================================================
# 03. 构建 Query Analyzer（结构化 LLM + Prompt）
# ==============================================================================
"""
Query Analyzer = Prompt + LLM.with_structured_output(TutorialSearch)

当用户输入自然语言问题时，LLM 会：
  1. 理解用户的搜索意图
  2. 将自然语言中的约束条件映射到 TutorialSearch 的各个字段
  3. 返回结构化的 TutorialSearch 对象
"""

query_analysis_system = """你是一位将用户问题转换为数据库查询的专家。
你可以访问一个包含用于构建 LLM 驱动应用的软件库教程视频的数据库。
给定一个问题，返回一个优化过的结构化查询，以检索最相关的结果。

如果遇到不熟悉的缩写或词语，不要尝试改写它们。

关键规则：
- content_search: 保留用户原始搜索意图，可以是完整的自然语言查询
- title_search: 提取最关键的 1-3 个技术关键词，这些关键词可能出现在视频标题中
- 数值和时间过滤字段：只在用户明确提到时才设置，不要凭空推断"""

query_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_analysis_system),
        ("human", "{question}"),
    ]
)

# with_structured_output 让 LLM 严格按照 TutorialSearch schema 输出 JSON
structured_llm = llm.with_structured_output(TutorialSearch)

# 查询分析链：输入 {"question": "..."} → 输出 TutorialSearch 对象
query_analyzer = query_analysis_prompt | structured_llm


# ==============================================================================
# 04. 演示：结构化查询分析（提取过滤参数）
# ==============================================================================
def demo_query_analysis():
    """
    演示核心功能：将自然语言问题转换为结构化的搜索参数。
    不涉及实际检索——只展示 LLM 如何理解并提取用户意图中的过滤条件。
    """
    print("=" * 60)
    print("【Query Analysis · 自然语言 → 结构化查询参数】")
    print("=" * 60)

    test_questions = [
        "rag from scratch",
        "videos on chat langchain published in 2023",
        "videos that are focused on the topic of chat langchain "
        "that are published before 2024",
        "how to use multi-modal models in an agent, "
        "only videos under 5 minutes",
        "RAG tutorials with more than 10000 views, from 2024",
    ]

    for q in test_questions:
        print(f"\n查询: \"{q}\"")
        result = query_analyzer.invoke({"question": q})
        result.pretty_print()


# ==============================================================================
# 05. 整合：Query Analysis + 向量检索 + Metadata Filtering
# ==============================================================================
"""
上面的 demo 只展示了 Query Analysis（提取参数），
这里我们把它和实际的向量检索 + 元数据过滤整合成完整链路：

  用户自然语言
    ↓
  Query Analyzer → TutorialSearch (结构化参数)
    ↓
  用 content_search 做向量语义检索 + 用其他字段做 metadata filter
    ↓
  LLM 基于检索结果生成回答
"""


def build_youtube_vectorstore(video_urls: list[str]) -> Chroma:
    """从多个 YouTube 视频构建向量库。"""
    all_docs = []
    for url in video_urls:
        try:
            docs = YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
            all_docs.extend(docs)
            print(f"  [OK] 加载: {url}")
        except Exception as e:
            print(f"  [SKIP] 加载失败: {url}, 原因: {e}")

    if not all_docs:
        print("[WARN] 没有成功加载任何视频，使用模拟数据。")
        all_docs = create_mock_video_docs()

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"  切分完成，共 {len(splits)} 个 chunk")

    # 建向量库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="youtube_tutorials",
        persist_directory=PERSIST_DIR,
    )
    print(f"  向量库构建完成")
    return vectorstore


def rag_with_query_analysis(question: str, vectorstore: Chroma) -> str:
    """
    完整的 Query Analysis + RAG 链路：

    1. Query Analyzer 提取结构化参数
    2. 用 content_search 做语义检索（可选 title_search 辅助）
    3. 用其他参数做 metadata 过滤
    4. LLM 生成回答
    """
    # Step 1: 结构化查询分析
    params: TutorialSearch = query_analyzer.invoke({"question": question})
    print(f"\n  [Step 1] Query Analysis 结果:")
    params.pretty_print()

    # Step 2: 构建 filter 并检索
    metadata_filter = params.to_metadata_filter()
    search_kwargs = {"k": 4}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
        print(f"\n  [Step 2] 检索 filter: {metadata_filter}")
    else:
        print(f"\n  [Step 2] 无 metadata 过滤条件，纯语义检索")

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(params.content_search)
    print(f"  检索到 {len(docs)} 篇文档")

    if not docs:
        return "未找到相关视频内容。"

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "N/A")
        print(f"    [{i}] {title[:60]}...")

    # Step 3: RAG 生成
    context = "\n\n".join(doc.page_content for doc in docs)
    rag_prompt = ChatPromptTemplate.from_template(
        """你是一个技术教程助手。请根据以下视频转译内容回答用户问题。
如果内容不够，诚实说明。

视频内容：
{context}

用户问题：{question}

请用中文回答："""
    )
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer


def demo_full_pipeline():
    """演示完整的 Query Analysis + Metadata Filtering + RAG 链路。"""
    print("\n" + "=" * 60)
    print("【完整链路 · Query Analysis + RAG】")
    print("=" * 60)

    # 构建向量库（首次运行会调用 YouTube API）
    print("\n[准备] 构建 YouTube 视频向量库...")
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("  检测到已有向量库，直接加载...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="youtube_tutorials",
            persist_directory=PERSIST_DIR,
        )
    else:
        video_urls = [
            "https://www.youtube.com/watch?v=pbAd8O1Lvm4",  # RAG from scratch
            "https://www.youtube.com/watch?v=2TJxQj6n4eA",  # LangChain tutorial
        ]
        vectorstore = build_youtube_vectorstore(video_urls)

    if vectorstore is None:
        print("[ERROR] 向量库构建失败，请检查网络或 API Key。")
        return

    # 测试问题
    test_questions = [
        "what is RAG and how does it work?",
        "how to build a chatbot with langchain, videos from 2023",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"用户问题: \"{q}\"")
        answer = rag_with_query_analysis(q, vectorstore)
        print(f"\n  [Step 3] 回答:\n{answer[:300]}...")


# ==============================================================================
# 06. 加载单个视频并查看 Metadata 结构
# ==============================================================================
def demo_youtube_metadata():
    """演示：加载 YouTube 视频并查看其元数据结构。YouTube 不可用时自动使用模拟数据。"""
    print("=" * 60)
    print("【视频元数据结构预览】")
    print("=" * 60)

    docs = load_youtube_video()
    if not docs:
        return

    print(f"\n  共加载 {len(docs)} 个 Document\n")

    # 展示第一个 Document 的完整 metadata
    meta = docs[0].metadata
    print(f"  Document 元数据字段（来自 YouTube Loader）：")
    print(f"  {'─' * 52}")
    for key, value in meta.items():
        val_str = str(value)
        if len(val_str) > 70:
            val_str = val_str[:70] + "..."
        print(f"    {key:20s} | {val_str}")

    # 展示第一个 Document 的 page_content 片段
    print(f"\n  page_content 前 200 字符：")
    print(f"    {docs[0].page_content[:200]}...")

    # 列出所有视频的概览
    print(f"\n  所有视频概览：")
    print(f"  {'─' * 52}")
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "N/A")
        views = doc.metadata.get("view_count", 0)
        length = doc.metadata.get("length", 0)
        date = doc.metadata.get("publish_date", "N/A")
        print(f"    [{i}] {title[:40]:40s} | views={views:>6} | {length:>4}s | {date[:10]}")


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # ===== 第一部分：结构化查询分析（核心：不检索，只看 LLM 如何提取参数） =====
    demo_query_analysis()

    # ===== 第二部分：查看视频元数据结构（YouTube 不可用时使用模拟数据） =====
    demo_youtube_metadata()

    # ===== 第三部分：完整链路（Query Analysis + 检索 + RAG 生成） =====
    # 注意：这会调用 Embedding API + YouTube API + LLM API
    demo_full_pipeline()


# ==============================================================================
# 结构化查询分析 · 总结
# ==============================================================================
"""
【核心思路】
Query Analysis = LLM 结构化输出 + Pydantic Schema

把"用户自然语言中隐含的过滤条件"自动提取为结构化参数，
然后这些参数可以直接传给向量库的 metadata filter。

【关键设计点】

1. Schema 设计
   - 每个"用户可能提到的约束"对应一个字段
   - 可选字段用 Optional，Field(default=None)
   - Field description 要足够清晰，帮助 LLM 准确判断
   - 必要时加 Only use if explicitly specified 来限制 LLM 过度推断

2. content_search vs title_search 的区别
   - content_search: 用于向量相似度搜索（匹配视频转译文本），保留自然语言表述
   - title_search: 精简关键词，用于标题匹配（可配合 BM25 或精确匹配）
   - 这是"多字段检索"的典型应用：不同字段用不同检索策略

3. 与其他策略的关系
   - Query Analysis 和 Metadata Filtering 是上下游关系：
     Query Analysis 负责"参数提取"，Metadata Filtering 负责"参数执行"
   - 可以和 Routing 结合：先路由选库，再 Query Analysis 提取过滤条件
   - 可以和 Multi-Query 结合：每个改写查询都经过相同的 Query Analysis 提取参数

【不适用场景】
  ❌ 用户输入中几乎没有约束条件（如"给我推荐个视频"）
  ❌ 过滤维度频繁变化（每次都要改 Schema + Prompt）
  ❌ 对 LLM 的字段提取准确率要求极高但无法容忍错误
     → 可考虑用规则匹配兜底

【思考题】

1. 如果用户说了"大约5分钟"，LLM 能正确处理吗？
   → 需要看 LLM 的理解能力。可以在 max_length_sec 的 description 中
     加上示例："5 minutes → 300"，帮助 LLM 做转换。
   → 也可以让 LLM 输出原始表述 + 解析后的值，由代码做二次处理。

2. title_search 应该如何用于实际检索？
   → 可以同时做两次检索：一次用 content_search 做向量检索，
     一次用 title_search 做关键词/BM25 检索（需要向量库支持 title 字段索引）。
   → 最后合并结果 → RRF（Reciprocal Rank Fusion，参见 04_RAG_Fusion）。

3. 如何评估 Query Analysis 的准确率？
   → 准备一批标注数据（用户问题 + 正确的结构化参数）。
   → 对比 LLM 输出和标准答案的各字段差异。
   → 重点关注"不该填的填了"（过度推断）和"该填的没填"（遗漏）。

4. 除了 Pydantic，还有其他方式实现结构化输出吗？
   → JSON mode: 在 Prompt 中指定输出 JSON 格式，用 JsonOutputParser 解析。
   → Function Calling: 使用 OpenAI 的 function calling / tool use。
   → 规则匹配: 正则提取时间/数字，简单可靠但覆盖面窄。
"""
