# RAG From Scratch — 从检索增强到查询翻译与编排思考

本项目基于 LangChain 官方 RAG 教程，以**代码实践 + 架构思考**的方式，逐步探索检索增强生成（RAG）的核心机制，并延伸到对**大模型编排（Orchestration）**的理解。

> **项目定位**：这是一个**学习性质**的项目。我们通过手写代码理解 RAG 的每个环节，再通过对查询翻译策略的学习，引发对"大模型系统如何编排"这一架构问题的思考。

## 📚 参考资源

| 资源 | 链接 |
|------|------|
| **Bilibili 教程视频** | [BV153SCBBEme](https://www.bilibili.com/video/BV153SCBBEme?spm_id_from=333.788.player.switch&vd_source=99a406ddcc2214e5535b9e880c261d18&p=4) |
| **原版 YouTube 视频** | [RAG From Scratch Playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) |
| **官方代码仓库** | [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) |
| **Step-Back 论文** | [arXiv:2310.06117](https://arxiv.org/pdf/2310.06117.pdf) |

## 🏗️ 项目结构

```
rag-lancedb/
├── .env                              # API 密钥配置（已被 .gitignore 忽略）
├── .gitignore
├── README.md
├── numpy_practice.py                 # NumPy 向量运算练习
├── 01_Rag_From_Scraft_example.py     # 完整 RAG 应用（检索 + 生成）
├── 02_RAG_Basics_and_Indexing.py     # RAG 原理与建库流程
├── 03_Query_Translation_Multi_Query.py   # Multi-Query 多查询改写
├── 04_RAG_Fusion.py                  # RAG-Fusion：改写 + RRF 重排序
├── 05_Query_Translation_Decomposition.py # 问题分解：递归回答 & 独立汇总
├── 06_Query_Translation_Step_Back.py     # Step-Back：抽象提升 + 双路检索
├── 07_Query_Translation_HyDE.py          # HyDE：假设文档嵌入
└── chroma_storage/                   # Chroma 向量数据库持久化目录
```

| 文件 | 说明 | 核心策略 |
|------|------|---------|
| `01_Rag_From_Scraft_example.py` | **完整应用**：自动检测持久化库 → 检索 → Prompt → LLM → 输出 | 基础 RAG |
| `02_RAG_Basics_and_Indexing.py` | **原理学习**：Token / Embedding / 相似度 / 加载 / 切分 / 建库 | 基础 RAG |
| `03_Query_Translation_Multi_Query.py` | 问题模糊时，生成 5 个改写版本，批量检索后去重合并 | **横向改写** |
| `04_RAG_Fusion.py` | 同 03 生成多查询，但用 RRF 互逆排序融合重排结果 | **横向改写 + 重排序** |
| `05_Query_Translation_Decomposition.py` | 复杂问题拆成子问题，递归传递或独立回答后汇总 | **纵向拆解** |
| `06_Query_Translation_Step_Back.py` | 具体问题抽象成宏观问题，双路检索融合回答 | **抽象提升** |
| `07_Query_Translation_HyDE.py` | 让 LLM 生成假设文档，用文档语义做向量检索 | **空间转换** |

## 🎯 查询翻译策略总览

查询翻译（Query Translation）解决的核心问题：**用户的问题不一定是检索的最佳查询**。通过让 LLM 先"改写/拆解/提升"问题，可以获取更全面的上下文，从而生成更准确的答案。

```
                    抽象程度 ↑
                             │
           Step Back ────────┤  向上抽象：具体问题 → 宏观背景
          （回退策略）        │  "任务分解是什么？" → "LLM Agent 有哪些核心能力？"
                             │
              Question ──────┼  原始问题（中间层）
                             │
    Decomposition ───────────┤  向下拆解：大问题 → 多个子问题
   （问题分解/Least-to-Most） │  "Agent 系统有哪些组件？" → [组件A?, 组件B?, 组件C?]
                             │
                    抽象程度 ↓

    Multi Query / RAG-Fusion ───→ 水平方向：同一粒度的多种表述
                                  "任务分解" → ["任务拆解", "任务细分", "子任务划分"]
```

| 策略 | 方向 | 核心思想 | 适用场景 | 文件 |
|------|------|---------|---------|------|
| **Multi Query** | 水平 | 同一问题的多种表述，批量检索取并集 | 问题表述模糊、同义词多 | `03` |
| **RAG-Fusion** | 水平 | 同上，但用 RRF 算法对多路结果重排序 | 需要更精准的排序融合 | `04` |
| **Decomposition** | 向下 | 把复杂问题拆成多个子问题分别处理 | 问题包含多个独立子任务 | `05` |
| **Step Back** | 向上 | 把具体问题提升到更宏观的层次 | 需要背景知识才能理解细节 | `06` |
| **HyDE** | 空间转换 | 生成假设文档，用文档语义去检索 | 查询短/模糊，与文档用词差异大 | `07` |

## 🤔 延伸思考：从查询翻译到 AI 编排

写完 `03~06` 的代码后，一个自然的问题浮现：**用户的每个问题都要走固定的查询翻译链路吗？** 显然不是——简单事实问题直接检索就够了，模糊问题才需要 Multi Query，复杂问题才需要 Decomposition。这就引出了**编排（Orchestration）**的核心命题：

> **系统如何根据问题的特征，动态选择合适的处理链路？**

以下四个思考，是从代码实践走向架构理解的延伸。

---

### 思考一：意图识别的边界

查询翻译只解决了"检索策略的选择"，但一个完整系统的意图路由远不止"闲聊 / 搜索 / 代码"三个分支：

```
用户输入
   ├──→ 闲聊（Chitchat）──────────────→ 直接生成，无需检索
   ├──→ 知识检索（Knowledge RAG）─────→ 私有向量库 / 文档
   │       ├── 简单事实 → 直接检索
   │       ├── 模糊问题 → Multi Query
   │       ├── 复杂问题 → Decomposition
   │       └── 需背景   → Step Back
   ├──→ 实时信息（Web Search）────────→ 搜索引擎 / API
   ├──→ 代码辅助（Code Assistant）────→ 代码专用模型 / 代码向量库
   ├──→ 数据分析（Data Analysis）─────→ SQL / Python 执行 / 图表生成
   ├──→ 图像生成（Image Generation）──→ DALL-E / Stable Diffusion
   ├──→ 多模态理解（Vision）──────────→ 图像理解 / OCR / 视频分析
   ├──→ 工具调用（Tool Use）──────────→ 调用外部 API（发邮件、订机票）
   ├──→ 记忆检索（Memory Recall）─────→ 用户历史偏好 / 过往对话
   ├──→ 文件处理（Document Processing）→ PDF 解析 / 表格提取
   └──→ 任务执行（Agent / Workflow）──→ 多步规划 / 调用多个工具串联
```

意图路由的粒度可以非常细，甚至可以**嵌套**——比如"搜索"内部还要再分是用私有库还是搜互联网。

---

### 思考二："检索"不只有向量数据库

RAG 的 "R"（Retrieval）**不限于向量检索**。只要能"把外部信息拉进上下文"，就是检索。

| 数据源类型 | 说明 | 例子 |
|-----------|------|------|
| **私有向量知识库** | 企业文档、产品手册、论文 | 本项目的 `./chroma_storage` |
| **搜索引擎** | 实时互联网信息 | Google / Bing / 百度 / DuckDuckGo |
| **企业内部系统** | 数据库、CRM、ERP、飞书/钉钉文档 | 通过 API 或 MCP 接入 |
| **结构化数据库** | SQL / NoSQL 实时查询 | `SELECT * FROM orders WHERE ...` |
| **API 服务** | 第三方数据接口 | 天气 API、股票 API、地图 API |
| **知识图谱** | 实体关系网络 | GraphRAG（微软方案）|
| **实时流数据** | 消息队列、日志流 | Kafka、消息推送 |

---

### 思考三：知识库不一定是静态的

本项目使用的是**静态向量库**（预先加载好的 HTML 文档），但实际系统中知识往往是动态的：

| 维度 | 静态知识库 | 动态知识库 |
|------|-----------|-----------|
| 存储内容 | PDF / HTML 文件 | 数据库实时查询结果 |
| 构造时机 | 预先切分好 | 每次查询时动态构造 |
| 更新方式 | 重新索引 | 数据本身就是最新的 |
| 部署形态 | 离线存储 | 在线 API 实时拉取 |

**动态知识库的几种实现**：

- **SQL RAG**：LLM 生成 SQL → 查询数据库 → 结果注入 Prompt
- **API RAG**：Tool 调用外部 API → 返回 JSON → 注入 Prompt
- **Web Search RAG**：搜索引擎实时抓取 → 网页片段 → 注入 Prompt
- **增量向量库**：新文档实时入库，无需全量重建
- **混合检索**：向量库 + 关键词搜索 + 图数据库同时召回

---

### 思考四：搜索引擎本身就是一种检索器

Web Search 完全可以作为 RAG 的"外部检索器"，甚至比向量库更适合获取**实时信息**：

```
用户问题 → 搜索引擎（代替向量检索）→ 获取文本片段 → Prompt → LLM → 答案
```

更进一步，通过 **Tools / Function Calling**，可以让模型**自己决定**何时搜索、搜什么：

- "iPhone 16 发布日期" → 需要实时信息 → 调用 `web_search`
- "我公司的请假流程是什么" → 私有知识 → 调用 `knowledge_base_search`
- "1+1 等于几" → 不需要工具 → 直接回答

**MCP（Model Context Protocol）**则是 Anthropic 推出的标准化协议，相当于给 AI 系统装了一个"通用接口"，让模型通过统一协议访问搜索、数据库、API 等异构数据源。

---

### 思考五：从 LCEL 管道到图结构

本项目使用 LangChain 的 LCEL（`|` 管道运算符）实现查询翻译，体验很好，但遇到复杂分支时会暴露限制：

| LCEL 管道 | 图结构（LangGraph 思想）|
|----------|------------------------|
| 线性流向，一条路走到黑 | 节点之间可以有条件分支、循环、并行 |
| 无状态，每次调用独立 | 有状态，可以追踪整个执行过程 |
| 无法人工介入 | 可以在关键节点暂停等待人类确认 |
| 适合固定策略 | 适合动态决策的复杂系统 |

这些思考不是为了去重构代码，而是通过写代码的过程，**理解为什么业界需要从"管道"进化到"图"**。

---

### 一张图看懂编排视角下的完整架构

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
            ├──→ 私有/历史信息 ──→ 向量检索（RAG）+ 查询翻译策略
            │       │
            │       ├── 简单 → 直接检索
            │       ├── 模糊 → Multi Query
            │       ├── 复杂 → Decomposition
            │       └── 需背景 → Step Back
            │
            ├──→ 结构化数据 ─────→ SQL Tool / 数据库查询
            │
            ├──→ 操作型任务 ─────→ Function Calling（发邮件、订机票）
            │
            └──→ 不确定 ─────────→ 先搜索/检索 → 再判断下一步
                                         │
                                         └── 效果不好？重新生成查询，循环
```

## ⚙️ 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
- `langchain` / `langchain-openai` / `langchain-community`
- `chromadb`
- `python-dotenv`
- `beautifulsoup4`
- `langchain-text-splitters`

> **注意**：`langchain_community.vectorstores.Chroma` 已弃用，生产环境建议迁移至 `langchain-chroma`。

### 2. 配置 API 密钥

复制 `.env` 文件模板并填写你的阿里云百炼（或 OpenAI）API 密钥：

```bash
# .env
API_KEY=sk-your-api-key-here
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> `.env` 已加入 `.gitignore`，密钥不会意外提交到 Git。

### 3. 准备文档

将 `LLM Powered Autonomous Agents _ Lil Log.html` 放入 `./documents/` 目录（首次运行会自动建库）。

## 💰 费用说明

项目中标注了 `[扣费]` 的代码行会调用云端 API，产生费用：

| 操作 | 费用类型 | 说明 |
|------|---------|------|
| `embed_documents` / `embed_query` | Embedding API | 文本向量化，按 token 计费 |
| `retriever.invoke()` | Embedding API | 检索时把问题向量化 |
| `chain.invoke` / `rag_chain.invoke` | LLM API + Embedding | 包含检索 + 大模型生成 |
| `generate_queries_xxx.invoke()` | LLM API | 查询翻译（Multi Query / Decomposition / Step Back）额外调用 LLM |

**省钱技巧**：所有文件都实现了**持久化检测**——如果 `./chroma_storage` 已存在，会直接加载跳过 Embedding 建库，避免重复扣费。

## 🛠️ 技术栈

- **LLM**: DeepSeek-v3.2（通过阿里云百炼兼容 OpenAI 接口）
- **Embedding**: 阿里云百炼 `text-embedding-v4`（1024 维，batch_size=10）
- **向量数据库**: Chroma（本地持久化，`./chroma_storage`）
- **框架**: LangChain + LCEL（`|` 管道运算符）
- **查询翻译**: Few-Shot Prompting / RRF 排序 / 递归链 / 并行分支
- **文本处理**: BeautifulSoup4 + RecursiveCharacterTextSplitter（非 tiktoken）

## 📝 关键设计决策

| 决策 | 说明 |
|------|------|
| 保留旧版 API `get_relevant_documents` | `02` 文件故意保留，用于教学对比新版 `invoke()` |
| RRF 返回 `list[Document]` 而非 `list[tuple]` | 避免 tuple 污染下游 Prompt 的 `{context}` 变量 |
| Prompt 禁止编号 | `03` 的 Multi Query Prompt 要求 LLM 不输出编号，避免语义污染 |
| 过滤空字符串 | `CustomEmbeddings` 和查询翻译链都过滤空值，防止 API 报错 |
| 非 tiktoken 切分器 | 阿里云 Embedding 无需 tiktoken，避免不兼容报错 |

## 📄 License

学习项目，代码参考自 [LangChain RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch)。
