"""
================================================================================
RAG 完整流程总览（本文件对应 Part 2: Indexing / 索引阶段）
================================================================================

【前置知识演示】
01. Token 计数          : 用 tiktoken 计算文本有多少个 token
02. 文本向量化          : 用 Embedding 模型把文字转成 1024 维向量
03. 余弦相似度          : 计算两个向量有多"像"，理解语义搜索的数学原理

【Indexing 核心流程（建库）】
04. Document Loaders    : 加载原始文档（网页/HTML/TXT/PDF 等）
05. Text Splitters      : 把长文档切分成小块（chunk），方便精准检索
06. Vectorstores        : 把切分后的文本向量化，存入向量数据库（Chroma）

07. Retriever           : 根据用户问题检索最相关的文档片段
08. Prompt Engineering  : 把检索结果塞进 Prompt 模板
09. LLM Generation      : 大模型基于上下文生成最终答案

================================================================================
"""

import os
from dotenv import load_dotenv
import tiktoken

# 加载 .env 环境变量
load_dotenv()

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

# ==================== 01. 测token数，国内只能通过下载文件下来 ====================
# 创建一个本地缓存目录
cache_dir = os.path.expanduser(r".\\tiktoken\\tiktoken_cache")
os.makedirs(cache_dir, exist_ok=True)

# 告诉 tiktoken 从这里读取
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num = num_tokens_from_string(question, "cl100k_base")
print('token: ', num)

# ==================== 02. 转化成1024维向量 ====================
from langchain_openai import OpenAIEmbeddings
embd = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
    dimensions=1024,
    chunk_size=10,
    check_embedding_ctx_length=False
)
# [扣费] 调用 Embedding API（单条）
# query_result = embd.embed_query(question)
# [扣费] 调用 Embedding API（单条）
# document_result = embd.embed_query(document)

# print(len(query_result))

# ==================== 03. 算余弦相似度(测量这两个向量在数学空间里的"夹角") ====================
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)           # 向量点积
    norm_vec1 = np.linalg.norm(vec1)           # 向量1的模长
    norm_vec2 = np.linalg.norm(vec2)           # 向量2的模长
    return dot_product / (norm_vec1 * norm_vec2)  # 余弦相似度公式

# similarity = cosine_similarity(query_result, document_result)
# print("Cosine Similarity:", similarity)

# ==================== 04. Document Loaders（文档加载器） ====================
# 所有加载器都位于 langchain_community.document_loaders，用法统一：
#   loader = XXXLoader("路径或URL")
#   docs = loader.load()  # 返回 List[Document]
#
# 【网页类】
#   WebBaseLoader     : 在线网页，支持 bs_kwargs 过滤（本例用的）
#   AsyncHtmlLoader   : 异步批量抓多个网页
#   BSHTMLLoader      : 读取本地 .html 文件（需指定 open_encoding='utf-8'）
#
# 【本地文本/文件】
#   TextLoader        : .txt 文件
#   CSVLoader         : CSV，每行一个 Document
#   JSONLoader        : JSON，可按路径提取字段
#   UnstructuredFileLoader : 自动识别格式（万能）
#
# 【PDF】
#   PyPDFLoader       : 普通文字 PDF（最常用）
#   PDFPlumberLoader  : 表格/复杂排版 PDF
#   UnstructuredPDFLoader : 扫描件、图文混排
#
# 【Office】
#   Docx2txtLoader    : Word .docx
#   UnstructuredExcelLoader : Excel .xlsx
#   UnstructuredPowerPointLoader : PPT .pptx
#
# 【代码】
#   PythonLoader      : .py 文件
#   NotebookLoader    : Jupyter .ipynb
#
# 【云端/第三方】
#   ArxivLoader       : arXiv 论文
#   YoutubeLoader     : YouTube 字幕
#   NotionDBLoader    : Notion 数据库

# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# ==================== 05. Text Splitters（文本切分器） ====================
# 
# 【RecursiveCharacterTextSplitter vs from_tiktoken_encoder 区别】
# 
# 1. RecursiveCharacterTextSplitter(chunk_size=1000)
#    - 按"字符数"切分（characters），不是 token
#    - 递归尝试分隔符：["\n\n", "\n", " ", ""]
#    - 先按段落切，段落太长按换行切，再按空格切，最后按字符切
#    - 简单通用，但和模型的 token 计费不完全对齐
# 
# 2. RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300)
#    - 按"token 数"切分，更精准
#    - 内部用 tiktoken 先编码，按 token 边界切，再解码回文本
#    - chunk_size=300 表示最多 300 个 token（而不是 300 个字符）
#    - 适合配合 OpenAI/GPT 类模型使用
#    - 注意：tiktoken 在国内首次使用需下载缓存文件，或改用其他 tokenizer
# 
# 【其他常见 Splitter】
# 
#   CharacterTextSplitter        : 最基础的固定字符切分，不递归，简单粗暴
#   TokenTextSplitter            : 直接按 token 切（需指定 tokenizer）
#   MarkdownHeaderTextSplitter   : 按 Markdown 标题（#, ##）切分，保留标题层级
#   HTMLHeaderTextSplitter       : 按 HTML 标签（h1, h2, p）切分
#   RecursiveJsonSplitter        : 按 JSON 结构层级切分
#   NLTKTextSplitter             : 按自然语言句子切分（需安装 NLTK）
#   SpacyTextSplitter            : 用 Spacy 做句子切分（更精准，需安装 Spacy）
#   PythonCodeTextSplitter       : 针对 Python 代码，按类/函数/方法切分
#   LanguageTextSplitter         : 支持多种编程语言（JS, Java, C++ 等）
#
# 选择建议：
#   - 普通文章/网页   → RecursiveCharacterTextSplitter（或 from_tiktoken_encoder）
#   - Markdown 文档   → MarkdownHeaderTextSplitter
#   - 代码仓库        → PythonCodeTextSplitter / LanguageTextSplitter
#   - 需要精准控 token → from_tiktoken_encoder 或 TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# ==================== 06. Vectorstores（向量存储） ====================
# 
# 作用：把切分好的文本片段 → 向量化 → 存入向量数据库 → 创建检索器
# 
# 流程：
#   1. Chroma.from_documents() 内部会：
#      - 调用 embedding.embed_documents(splits) 把所有片段转成向量
#      - 自动创建 collection（表）
#      - 把【原文 + 向量 + metadata】一起存入本地/内存
#   2. as_retriever() 包装成一个标准检索接口，方便后续 LCEL 链调用
# 
# 关键参数：
#   - persist_directory="./chroma_db" : 指定持久化路径（否则重启数据丢失）
#   - collection_name="xxx"           : 给表起名字（默认叫 "langchain"）
#   - embedding=OpenAIEmbeddings()    : 指定用哪个模型做向量化

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

# ==================== 07. Retrieval（检索） ====================
 
# retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
#  这是检索器的手动调用方式（旧版 API，新版推荐用 retriever.invoke()，效果一样）：
# [扣费] 调用 Embedding API（检索时把问题向量化）
docs = retriever.get_relevant_documents("What is Task Decomposition?")

print(len(docs))


# ==================== 08. Prompt Engineering（Prompt 工程） ====================
# 
# 作用：定义大模型的指令模板，告诉它"基于检索到的资料回答问题"
# 
# 模板里的两个关键占位符：
#   {context}  : 检索器返回的相关文档片段（检索后自动填入）
#   {question} : 用户的原始问题（透传进来）
# 
# 两种获取 Prompt 的方式：
#   1. 手写（如下）：灵活可控，推荐国内使用
#   2. hub.pull("rlm/rag-prompt")：从 LangChain Hub 拉取标准模板（需联网，国内不稳定）
# 
# 调试技巧：
#   下面 chain.invoke({"context": docs, ...}) 是"手动喂料"写法，
#   直接把你之前检索到的 docs 塞进去，绕过了 retriever，适合单独调试 Prompt。

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 手写 RAG Prompt 模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM：初始化大模型
# 教程用 gpt-3.5-turbo，国内建议换成阿里云百炼的 deepseek-v3.2 或 qwen-turbo
llm = ChatOpenAI(
    model='deepseek-v3.2',      # ← 国内可访问的模型
    api_key=os.getenv("API_KEY"),            # ← 你的阿里云 API Key
    base_url=os.getenv("API_BASE"),          # ← https://dashscope.aliyuncs.com/compatible-mode/v1
    temperature=0               # 0 表示最确定性输出（不发散）
)

# 简单链：Prompt → LLM
# 注意：这里没有检索器，是手动把 docs 和问题一起传进去的调试写法
chain = prompt | llm

# 手动调用：直接把检索好的文档和问题传给链
# 实际生产环境会用 rag_chain（见 09），这里只是演示 Prompt 填充效果
# [扣费] 调用 LLM API（手动喂料的调试写法）
chain.invoke({"context": docs, "question": "What is Task Decomposition?"})

# 【可选】从 LangChain Hub 拉取官方 RAG Prompt 模板
# 需要安装：pip install langchainhub
# 国内网络不稳定，建议直接手写模板代替（上面 template 已经是等价写法）
# from langchainhub import hub
# prompt_hub_rag = hub.pull("rlm/rag-prompt")
# print(prompt_hub_rag)
prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:""")

# ==================== 09. LLM Generation（大模型生成答案 - 完整 RAG 链） ====================
# 
# 作用：把"检索 → Prompt → 大模型 → 解析输出"串成一条自动化生产线（LCEL 链）
# 
# 数据流：
#   用户问题 "What is Task Decomposition?"
#     → RunnablePassthrough() 原样保留问题，作为 "question" 字段
#     → retriever 把问题向量化，去向量库搜回最相关的 K 个文档片段，作为 "context" 字段
#     → prompt 把 {context} 和 {question} 填入模板，拼成完整指令
#     → llm 大模型基于上下文生成回答（AIMessage 对象）
#     → StrOutputParser() 剥壳，只取 .content 里的纯文本字符串
# 
# 和第 08 节的区别：
#   08 是手动调试：chain.invoke({"context": docs, "question": "..."})  需要你先手动准备好 docs
#   09 是全自动化：rag_chain.invoke("...")  检索到回答一气呵成，只需传一个问题

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 完整 RAG 链（LCEL 语法）
rag_chain = (
    # 并行组装输入字典
    {
        # retriever 接收用户问题 → 向量检索 → 返回 List[Document]
        "context": retriever,
        
        # RunnablePassthrough 把用户问题原样保留，作为 question 字段
        "question": RunnablePassthrough()
    }
    # 把组装好的字典塞进 Prompt 模板（替换 {context} 和 {question}）
    | prompt
    # 发给大模型生成答案
    | llm
    # 把 AIMessage 对象解析成纯字符串（去掉 metadata 等包装）
    | StrOutputParser()
)

# ------------------------------------------------------------------
# Runnable 的三个通用调用接口（invoke / batch / stream）
# 所有 LCEL 链都支持这三个方法，就像 Python 列表都有 .append() / .pop()
# ------------------------------------------------------------------

# 1. invoke：单条同步调用，一问一答
#    传入一个输入，等全部生成完，一次性返回完整结果
#    适用：用户问了一个问题，等一个完整答案
# [扣费] 包含检索(embedding) + LLM 生成
result = rag_chain.invoke("What is Task Decomposition?")
print(result)

# 2. batch：批量并行调用，一次处理多个问题
#    传入列表，底层并行发送请求，比 for 循环快得多
#    适用：后台批量处理 100 条用户提问
questions = [
    "What is Task Decomposition?",
    "What is Agent System Overview?",
    "What is Self-Reflection?"
]
# [扣费] 批量调用 LLM API（包含检索+生成）
results = rag_chain.batch(questions)
print(results)  # ["Task decomposition is...", "Agent System is...", "Self-Reflection is..."]

# ------------------------------------------------------------------
# invoke vs stream 的本质区别（重点理解）
# ------------------------------------------------------------------
# invoke：你看到的是"一次性拿到完整答案"
#   - 底层网络可以是流式传输（LLM 初始化时 streaming=True），但 LangChain 
#     内部帮你把所有 chunk 收集起来、拼成完整消息再返回
#   - 返回的是一个完整的字符串 / AIMessage
#   - ReAct Agent 的节点内部必须用 invoke，因为需要完整的 tool_calls 来判断下一步
#
# stream：你看到的是"一个字一个字往外蹦"
#   - 每生成一个 token 就立即 yield 出来，前端能实时看到打字机效果
#   - 适合用户体验，但节点内部不能用于判断 tool_calls（还没生成完）

# 3. stream：流式调用，逐字吐出
#    模型每生成一个 token 就立即 yield 出来，适合前端打字机效果
#    适用：用户不用干等，实时看到答案一个一个跳出来
# [扣费] 流式调用 LLM API（包含检索+生成）
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="")  # 逐字打印，不换行

# ------------------------------------------------------------------
# 还有异步版本（需要 async/await 语法）：
#   ainvoke()  : 异步单条调用
#   abatch()   : 异步批量调用
#   astream()  : 异步流式调用
# ------------------------------------------------------------------
# import asyncio
# async def main():
#     result = await rag_chain.ainvoke("What is Task Decomposition?")
#     print(result)
# asyncio.run(main())