# RAG From Scratch

本项目基于 LangChain 官方 RAG 教程，从零实现一个检索增强生成（Retrieval-Augmented Generation）应用。

## 📚 参考资源

| 资源 | 链接 |
|------|------|
| **Bilibili 教程视频** | [BV153SCBBEme](https://www.bilibili.com/video/BV153SCBBEme?spm_id_from=333.788.player.switch&vd_source=99a406ddcc2214e5535b9e880c261d18&p=4) |
| **原版 YouTube 视频** | [RAG From Scratch Playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) |
| **官方代码仓库** | [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) |

## 🏗️ 项目结构

```
rag-lancedb/
├── .env                          # API 密钥配置（已被 .gitignore 忽略）
├── .gitignore
├── README.md
├── 01 Rag_From_Scraft - example.py   # 完整 RAG 应用（检索 + 生成）
├── 02_RAG_Basics_and_Indexing.py     # RAG 原理与建库流程
└── chroma_storage/               # Chroma 向量数据库持久化目录
```

| 文件 | 说明 |
|------|------|
| `01 Rag_From_Scraft - example.py` | **完整应用**：连接已有向量库 → 检索 → Prompt → LLM → 输出答案 |
| `02_RAG_Basics_and_Indexing.py` | **原理学习**：Token 计数 / Embedding / 相似度 / 文档加载 / 切分 / 建库 |

## ⚙️ 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
- `langchain` / `langchain-openai` / `langchain-community`
- `chromadb`
- `python-dotenv`
- `tiktoken`
- `numpy`

### 2. 配置 API 密钥

复制 `.env` 文件模板并填写你的阿里云百炼（或 OpenAI）API 密钥：

```bash
# .env
API_KEY=sk-your-api-key-here
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> **注意**：`.env` 已加入 `.gitignore`，密钥不会意外提交到 Git。

## 🔄 RAG 核心流程

### Part 1: Indexing（索引 / 建库）

```
原始文档（网页/HTML/TXT）
    ↓
Document Loader（加载文档）
    ↓
Text Splitter（切分成 chunk）
    ↓
Embedding（文本 → 向量）
    ↓
Vector Store（存入 Chroma 向量库）
```

### Part 2: Retrieval & Generation（检索与生成）

```
用户提问
    ↓
Retriever（向量检索 Top-K 相关片段）
    ↓
Prompt Engineering（把片段塞进 Prompt 模板）
    ↓
LLM（大模型基于上下文生成答案）
    ↓
StrOutputParser（解析为纯文本）
    ↓
最终答案
```

## 💰 费用说明

项目中标注了 `[扣费]` 的代码行会调用云端 API，产生费用：

| 操作 | 费用类型 | 说明 |
|------|---------|------|
| `embed_documents` / `embed_query` | Embedding API | 文本向量化，按 token 计费 |
| `retriever.get_relevant_documents` | Embedding API | 检索时把问题向量化 |
| `chain.invoke` / `rag_chain.invoke` | LLM API + Embedding | 包含检索 + 大模型生成 |
| `rag_chain.batch` | LLM API + Embedding | 批量调用 |
| `rag_chain.stream` | LLM API + Embedding | 流式调用 |

> Embedding 费用极低（短句通常免费额度内），LLM 按生成 token 计费。

## 🛠️ 技术栈

- **LLM**: DeepSeek-v3.2 / Qwen（通过阿里云百炼兼容接口）
- **Embedding**: 阿里云百炼 `text-embedding-v4`（1024 维）
- **向量数据库**: Chroma（本地持久化）
- **框架**: LangChain + LCEL（LangChain Expression Language）
- **文本处理**: BeautifulSoup4 + RecursiveCharacterTextSplitter

## 📝 学习笔记

### 01-09 流程索引

| 章节 | 内容 |
|------|------|
| 01 | Token 计数（tiktoken） |
| 02 | 文本向量化（Embedding） |
| 03 | 余弦相似度（Cosine Similarity） |
| 04 | 文档加载器（Document Loaders） |
| 05 | 文本切分器（Text Splitters） |
| 06 | 向量存储（Vectorstores） |
| 07 | 检索器（Retriever） |
| 08 | Prompt 工程（Prompt Engineering） |
| 09 | 大模型生成（LLM Generation） |

## 📄 License

学习项目，代码参考自 [LangChain RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch)。
