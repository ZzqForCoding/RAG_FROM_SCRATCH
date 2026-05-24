# RAG 从入门到查询翻译 —— 学习笔记与架构思考

> 基于 LangChain 官方 RAG 教程的代码实践总结，覆盖基础原理、五种查询翻译策略，以及从代码到架构的延伸思考。

---

## 一、RAG 基础原理

### 1.1 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）的核心思想是：在让大模型回答问题之前，先从外部知识库中检索相关文档，把检索结果作为上下文塞进 Prompt，让模型基于这些资料作答。

**为什么需要 RAG？**
- 大模型有知识截止时间，无法知道最新信息
- 大模型对专业领域/私有文档一无所知
- 大模型容易产生幻觉（一本正经地胡说八道）

### 1.2 RAG 完整流程

```
用户问题 → 向量检索 → 组装 Prompt（问题+上下文） → LLM 生成 → 输出答案
```

**Indexing（建库）阶段**：
1. **加载文档**（Document Loaders）：支持 HTML、PDF、TXT、CSV、Word、PPT 等
2. **文本切分**（Text Splitters）：把长文档切成小块（chunk），方便精准检索
3. **向量化**（Embedding）：用模型把文本转成高维向量
4. **存入向量库**（Vectorstores）：如 Chroma，支持持久化

**Query（查询）阶段**：
5. **检索**（Retriever）：把用户问题向量化，去向量库搜 Top-K 相似文档
6. **Prompt 工程**：把检索到的文档片段塞进模板
7. **大模型生成**：基于上下文生成答案

### 1.3 关键数学概念

**Embedding（嵌入）**：把离散的文本映射到连续的高维向量空间（本项目用 1024 维）。语义相近的文本，在向量空间中的距离也更近。

**余弦相似度**：衡量两个向量有多"像"。
```
cosine_similarity(A, B) = (A·B) / (||A|| × ||B||)
```
夹角越小 → 余弦值越接近 1 → 语义越相似。

### 1.4 LCEL 链式语法

LangChain 用 `|` 管道符号把多个步骤串成一条生产线：
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
- `RunnablePassthrough()`：原样透传输入
- `retriever | format_docs`：先检索文档，再格式化成字符串
- `StrOutputParser()`：把 AIMessage 剥壳，只保留纯文本

---

## 二、查询翻译策略（Query Translation）

**核心问题**：用户的问题不一定是检索的最佳查询。直接拿用户原话去向量库搜，可能召回不足或召回偏差。

**解决思路**：在检索之前，让 LLM 先对问题做"改写/拆解/提升"，获取更全面的上下文。

### 2.1 五种策略总览

```
                    抽象程度 ↑
                             │
           Step Back ────────┤  向上抽象：具体问题 → 宏观背景
                             │
              Question ──────┼  原始问题（中间层）
                             │
    Decomposition ───────────┤  向下拆解：大问题 → 多个子问题
                             │
                    抽象程度 ↓

    Multi Query / RAG-Fusion ───→ 水平方向：同一粒度的多种表述
    HyDE ──────────────────────→ 空间转换：问题 → 假设文档

    Routing ────────────────────→ 意图分流：多库/多链 → 选最合适的一个
    Query Analysis ─────────────→ 参数提取：自然语言 → 结构化过滤条件
```

| 策略 | 方向 | 核心思想 | 适用场景 |
|------|------|---------|---------|
| **Multi Query** | 水平 | 同一问题的多种表述，批量检索取并集 | 问题表述模糊、同义词多 |
| **RAG-Fusion** | 水平 | 同上，但用 RRF 算法对多路结果重排序 | 需要更精准的排序融合 |
| **Decomposition** | 向下 | 复杂问题拆成多个子问题分别处理 | 问题包含多个独立子任务 |
| **Step Back** | 向上 | 具体问题提升到更宏观的层次 | 需要背景知识才能理解细节 |
| **HyDE** | 空间转换 | 生成假设文档，用文档语义去检索 | 查询短/模糊，与文档用词差异大 |
| **Routing** | 意图分流 | 多库/多链场景下先判断意图再选对的路 | 多知识库、多Prompt风格、多数据源 |
| **Query Analysis** | 参数提取 | 从自然语言提取结构化过滤条件 | 多维过滤搜索、需从用户输入解析约束 |

---

### 2.2 Multi Query —— 多角度改写

**思路**：让 LLM 把同一个问题改写成 5 个不同措辞的版本，分别检索，最后合并去重。

**与基础 RAG 的区别**：
- 基础 RAG：1 个问题 → 1 次检索 → k 个文档
- Multi Query：1 个问题 → LLM 生成 5 个改写 → 5 次检索 → 去重合并

**合并方式**：简单并集去重（`get_unique_union`）
- 用 `dumps` 把 Document 序列化成 JSON 字符串
- 用 `set` 去重
- 用 `loads` 还原成 Document 对象

**局限**：只保留"出现过哪些文档"，不保留排名信息，文档无序。

---

### 2.3 RAG-Fusion —— 互逆排序融合

**思路**：与 Multi Query 一样生成多查询，但合并时用 RRF（Reciprocal Rank Fusion）算法智能排序。

**RRF 公式**：
```
score(d) = Σ [ 1 / (rank_i(d) + k) ]
```
- `rank_i(d)`：文档 d 在第 i 轮检索中的排名（从 0 开始）
- `k`：平滑常数，通常取 60

**直观理解**：
- 一篇文档如果在多个查询中都排第 1，得分累加后非常高
- 一篇文档只排在后面，得分很低
- **被多轮查询共同认可且排名靠前的文档 → 排名更靠前**

**与 Multi Query 的核心差异**：

| 维度 | Multi Query | RAG-Fusion |
|------|-------------|------------|
| 合并算法 | 简单并集去重 | RRF 互逆排序融合 |
| 输出顺序 | 无序 | 按融合得分降序排列 |
| 核心优势 | 扩大覆盖面 | 扩大覆盖面 + 智能排序 |

---

### 2.4 Decomposition —— 问题分解

**思路**：把复杂大问题拆成多个更小的子问题，分别处理。

**两种实现方式**：

| 方式 | 执行模式 | 上下文传递 | 适用场景 |
|------|---------|-----------|---------|
| **递归回答** | 串行 | 前序答案传给后续子问题 | 子问题有依赖/递进关系 |
| **独立回答后汇总** | 串行（可优化为并发） | 无传递，各自独立 | 子问题相互独立 |

**与 Multi Query 的本质区别**：
- Multi Query：不改问题粒度，只是"换种说法问同一个问题"
- Decomposition：改变问题粒度，把"一个大问题"拆成"多个不同的小问题"
- Multi Query 合并的是**文档**，Decomposition 合并的是**答案**

**数据流示意 —— 递归回答**：
```
总问题 → [Q1, Q2, Q3]
  Q1 → 检索 → A1
  Q2 + A1 → 检索 → A2
  Q3 + A1+A2 → 检索 → A3（最终答案）
```

**数据流示意 —— 独立回答**：
```
总问题 → [Q1, Q2, Q3]
  Q1 → 检索 → A1
  Q2 → 检索 → A2
  Q3 → 检索 → A3
  [A1, A2, A3] → 汇总 → 最终答案
```

---

### 2.5 Step Back —— 抽象提升

**思路**：从具体问题上升到更高层次的抽象问题，先回答宏观问题作为"背景知识"，再基于背景回答原始问题。

**核心价值**：
- **原问题检索（normal_context）**：获取具体细节（细粒度）
- **回退问题检索（step_back_context）**：获取宏观背景（粗粒度）
- **融合回答**：既有细节又有全局视野，避免"见树不见林"

**示例**：
```
原问题："What is task decomposition for LLM agents?"
Step-back："What are the core capabilities of LLM agents?"
```

**实现方式**：通过 Few-Shot Prompting 教 LLM 学会"抽象提升"。

**双路检索融合**：
```python
{
    "normal_context": 原始问题 → retriever → format_docs,
    "step_back_context": 生成step-back问题 → retriever → format_docs,
    "question": 原始问题
}
```

---

### 2.6 HyDE —— 假设文档嵌入

**核心洞察**：用户查询（疑问句）和文档片段（陈述句）在 Embedding 空间中天然分布不同，直接用问题去检索效果往往不佳。

**思路**：与其用"问题"去检索，不如让 LLM 先根据问题生成一篇"假设的、包含答案的文档"，然后用这篇假设文档去向量库检索。

**为什么假设文档检索效果更好？**
- 假设文档在语义空间上更接近真实文档（都是陈述性文本）
- 假设文档包含了问题背后的"意图"和"上下文"
- 相当于把"查询空间"的向量，桥接到了"文档空间"

**流程示意**：
```
用户问题 → [LLM 生成假设文档] → 用假设文档做 Embedding → 向量检索
  → 检索结果 + 原始问题 → 最终回答
```

**与 Multi Query 的本质区别**：
- Multi Query：横向扩展，始终停留在【查询空间】
- HyDE：纵向转换，从【查询空间】跨越到【文档空间】

**适用场景**：
- ✅ 查询非常短/模糊，与文档用词差异大
- ✅ 跨语言检索
- ✅ 零样本场景
- ❌ 查询本身就很长很详细
- ❌ 计算资源受限（需要额外一次 LLM 调用）

### 2.7 Routing —— 意图分流

**核心问题**：当系统拥有多个知识库、多个 Prompt 模板或多个处理链时，如何将用户问题准确地路由到最合适的那一个？

**解决思路**：在检索之前，先做一个轻量级的"意图判断"，决定走哪条路。

**两种实现方式**：

| 方式 | 原理 | 成本 | 适用场景 |
|------|------|------|---------|
| **LLM Routing** | 用 `with_structured_output()` + Pydantic 让 LLM 判断意图 | 高（需 LLM 调用） | 领域差异大、需要语义理解 |
| **Embedding Routing** | 用向量相似度匹配查询与预设 Prompt/描述文本 | 低（仅需 Embedding） | 领域特征明显、可用关键词区分 |

**LLM Routing 示例**：
```python
class RouteQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs", "golang_docs"]

structured_llm = llm.with_structured_output(RouteQuery)
# 用户: "Python 的 asyncio 怎么用？" → datasource="python_docs"
```

**Embedding Routing 示例**：
```python
# 预计算每个 Prompt 模板的 Embedding
prompt_embeddings = embeddings.embed_documents([physics_template, math_template])
# 查询时计算相似度，选最佳匹配
query_embedding = embeddings.embed_query(user_query)
best_idx = cosine_similarity([query_embedding], prompt_embeddings).argmax()
```

**Routing 与查询翻译的关系**：二者是**正交**的——先路由选库，再用查询翻译策略优化检索。

**适用场景**：
- ✅ 多领域知识库（技术文档 / 产品手册 / HR 政策）
- ✅ 多回答风格（专家模式 / 通俗模式）
- ✅ 多数据源混合（向量库 / SQL / API）
- ❌ 只有一个知识库（路由多余）

---

### 2.8 Query Analysis —— 结构化参数提取

**核心问题**：Metadata Filtering 需要手动硬编码 filter 条件，实际业务中这些条件来自用户的自然语言输入。

**解决思路**：用 LLM 结构化输出，将自然语言中的约束条件自动提取为结构化参数。

**流程示意**：
```
用户: "2023 年之后、少于 5 分钟、讲 RAG 的视频"
         ↓
  [Query Analyzer] LLM 结构化输出
         ↓
  TutorialSearch(
      content_search = "RAG tutorial",
      title_search   = "RAG",
      earliest_publish_date = 2023-01-01,
      max_length_sec = 300,
  )
         ↓
  [向量检索] vectorstore.search(content_search, filter={...})
```

**关键技术点**：
- **Pydantic Schema**：定义所有可能的过滤维度（日期范围、数值区间、关键词等）
- **Optional 字段**：只在用户明确提到时才设置，避免过度推断
- **content_search vs title_search**：前者做语义检索，后者做标题关键词匹配

**与其他策略的关系**：
- Query Analysis 和 Metadata Filtering 是上下游：Query Analysis 负责"参数提取"，Metadata Filtering 负责"参数执行"
- 可以和 Routing 结合：先路由选库 → 再 Query Analysis 提取过滤条件
- 可以和 Multi-Query 结合：每个改写查询都经过相同的 Query Analysis

**适用场景**：
- ✅ 自然语言搜索（用户不想填表单）
- ✅ 多字段联合筛选（关键词 + 时间 + 数值范围）
- ✅ 电商/视频/文档等多维过滤搜索
- ❌ 用户输入中几乎没有约束条件
- ❌ 过滤维度频繁变化

---

## 三、从查询翻译到 AI 编排

写完代码后，一个自然的问题浮现：**用户的每个问题都要走固定的查询翻译链路吗？**

显然不是——简单事实问题直接检索就够了，模糊问题才需要 Multi Query，复杂问题才需要 Decomposition。这就引出了**编排（Orchestration）**的核心命题：

> **系统如何根据问题的特征，动态选择合适的处理链路？**

### 3.1 意图识别的边界（可参考 `08_Query_Routing.py` 实现）

一个完整系统的意图路由远不止"闲聊 / 搜索 / 代码"三个分支。`08_Query_Routing.py` 演示了两种路由实现：LLM 结构化输出路由（准确但贵）和 Embedding 相似度路由（快但泛化弱）：

```
用户输入
   ├──→ 闲聊（Chitchat）──────────────→ 直接生成
   ├──→ 知识检索（Knowledge RAG）─────→ 私有向量库 / 文档
   │       ├── 简单事实 → 直接检索
   │       ├── 模糊问题 → Multi Query
   │       ├── 复杂问题 → Decomposition
   │       └── 需背景   → Step Back
   ├──→ 实时信息（Web Search）────────→ 搜索引擎
   ├──→ 代码辅助（Code Assistant）────→ 代码专用模型
   ├──→ 数据分析（Data Analysis）─────→ SQL / Python
   ├──→ 图像生成（Image Generation）──→ DALL-E / SD
   ├──→ 多模态理解（Vision）──────────→ 图像理解 / OCR
   ├──→ 工具调用（Tool Use）──────────→ 调用外部 API
   ├──→ 记忆检索（Memory Recall）─────→ 用户历史偏好
   └──→ 任务执行（Agent / Workflow）──→ 多步规划
```

### 3.2 "检索"不只有向量数据库

RAG 的 "R"（Retrieval）**不限于向量检索**。只要能"把外部信息拉进上下文"，就是检索。

| 数据源类型 | 说明 | 例子 |
|-----------|------|------|
| **私有向量知识库** | 企业文档、产品手册 | 本项目的 `./chroma_storage` |
| **搜索引擎** | 实时互联网信息 | Google / Bing / 百度 |
| **企业内部系统** | 数据库、CRM、ERP | 通过 API 或 MCP 接入 |
| **结构化数据库** | SQL / NoSQL 实时查询 | `SELECT * FROM orders WHERE ...` |
| **API 服务** | 第三方数据接口 | 天气 API、股票 API |
| **知识图谱** | 实体关系网络 | GraphRAG |
| **实时流数据** | 消息队列、日志流 | Kafka |

### 3.3 静态知识库 vs 动态知识库

| 维度 | 静态知识库 | 动态知识库 |
|------|-----------|-----------|
| 存储内容 | PDF / HTML 文件 | 数据库实时查询结果 |
| 构造时机 | 预先切分好 | 每次查询时动态构造 |
| 更新方式 | 重新索引 | 数据本身就是最新的 |
| 部署形态 | 离线存储 | 在线 API 实时拉取 |

**动态知识库的实现**：
- **SQL RAG**：LLM 生成 SQL → 查询数据库 → 结果注入 Prompt
- **API RAG**：Tool 调用外部 API → 返回 JSON → 注入 Prompt
- **Web Search RAG**：搜索引擎实时抓取 → 网页片段 → 注入 Prompt
- **混合检索**：向量库 + 关键词搜索 + 图数据库同时召回

### 3.4 搜索引擎本身就是一种检索器

Web Search 完全可以作为 RAG 的"外部检索器"，甚至比向量库更适合获取**实时信息**：

```
用户问题 → 搜索引擎（代替向量检索）→ 获取文本片段 → Prompt → LLM → 答案
```

通过 **Tools / Function Calling**，可以让模型**自己决定**何时搜索、搜什么：
- "iPhone 16 发布日期" → 需要实时信息 → 调用 `web_search`
- "我公司的请假流程是什么" → 私有知识 → 调用 `knowledge_base_search`
- "1+1 等于几" → 不需要工具 → 直接回答

**MCP（Model Context Protocol）**是 Anthropic 推出的标准化协议，相当于给 AI 系统装了一个"通用接口"，让模型通过统一协议访问搜索、数据库、API 等异构数据源。

### 3.5 从 LCEL 管道到图结构

本项目使用 LangChain 的 LCEL（`|` 管道运算符）实现查询翻译，体验很好，但遇到复杂分支时会暴露限制：

| LCEL 管道 | 图结构（LangGraph 思想）|
|----------|------------------------|
| 线性流向，一条路走到黑 | 节点之间可以有条件分支、循环、并行 |
| 无状态，每次调用独立 | 有状态，可以追踪整个执行过程 |
| 无法人工介入 | 可以在关键节点暂停等待人类确认 |
| 适合固定策略 | 适合动态决策的复杂系统 |

**这些思考不是为了去重构代码，而是通过写代码的过程，理解为什么业界需要从"管道"进化到"图"**。

---

## 四、完整架构视角

```
用户输入
   │
   ├──→ 无需外部信息 ──────→ 直接生成（闲聊、常识推理）
   │
   └──→ 需要外部信息 ──────→ [路由决策]
            │
            ├──→ 实时/公开信息 ──→ Web Search / API Tool / MCP
            │                        │
            │                        └── 搜索结果 → Prompt → LLM
            │
            ├──→ 私有/历史信息 ──→ [Routing 选库] → 向量检索（RAG）+ 查询翻译策略
            │       │
            │       ├── 简单 → 直接检索
            │       ├── 模糊 → Multi Query
            │       ├── 复杂 → Decomposition
            │       ├── 需背景 → Step Back
            │       └── 要过滤 → Query Analysis + Metadata Filter
            │
            ├──→ 结构化数据 ─────→ SQL Tool / 数据库查询
            │
            ├──→ 操作型任务 ─────→ Function Calling（发邮件、订机票）
            │
            └──→ 不确定 ─────────→ 先搜索/检索 → 再判断下一步
                                         │
                                         └── 效果不好？重新生成查询，循环
```

---

## 五、技术细节与设计决策

### 5.1 技术栈

- **LLM**: DeepSeek-v3.2（通过阿里云百炼兼容 OpenAI 接口）
- **Embedding**: 阿里云百炼 `text-embedding-v4`（1024 维）
- **向量数据库**: Chroma（本地持久化）
- **框架**: LangChain + LCEL
- **文本处理**: BeautifulSoup4 + RecursiveCharacterTextSplitter

### 5.2 关键设计决策

| 决策 | 说明 |
|------|------|
| 保留旧版 API `get_relevant_documents` | `02` 文件故意保留，用于教学对比新版 `invoke()` |
| RRF 返回 `list[Document]` 而非 `list[tuple]` | 避免 tuple 污染下游 Prompt 的 `{context}` 变量 |
| Prompt 禁止编号 | Multi Query Prompt 要求 LLM 不输出编号，避免语义污染 |
| 过滤空字符串 | `CustomEmbeddings` 和查询翻译链都过滤空值，防止 API 报错 |
| 非 tiktoken 切分器 | 阿里云 Embedding 无需 tiktoken，避免不兼容报错 |

### 5.3 省钱技巧

所有文件都实现了**持久化检测**——如果 `./chroma_storage` 已存在，直接加载跳过 Embedding 建库，避免重复扣费。

**扣费点标注**：
- `embed_documents` / `embed_query`：Embedding API
- `retriever.invoke()`：检索时把问题向量化
- `chain.invoke` / `rag_chain.invoke`：包含检索 + LLM 生成
- `generate_queries_xxx.invoke()`：查询翻译额外调用 LLM

---

## 六、参考资源

| 资源 | 链接 |
|------|------|
| 原版 YouTube 视频 | [RAG From Scratch Playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) |
| 官方代码仓库 | [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) |
| Step-Back 论文 | [arXiv:2310.06117](https://arxiv.org/pdf/2310.06117.pdf) |
| HyDE 论文 | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
