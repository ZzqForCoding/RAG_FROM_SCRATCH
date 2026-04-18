"""
================================================================================
主要流程（文件1：完整 RAG 应用 - 从建库到问答）
================================================================================
阶段1：定义 Embedding 模型（CustomEmbeddings / OpenAIEmbeddings）
阶段2：建库（Index）→ 加载文档 → 切分 → 向量化 → 存入 Chroma（首次运行）
阶段3：查询（Query）→ 连接已有向量库 → 创建检索器
阶段4：组装 RAG 链（LCEL）→ 检索 + Prompt + LLM + 输出解析
阶段5：执行问答 → rag_chain.invoke("用户问题") → 返回答案
================================================================================
"""

import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

from langchain_core.embeddings import Embeddings
from openai import OpenAI

class CustomEmbeddings(Embeddings):
    """
    手动封装阿里云 Embedding API。
    与 OpenAIEmbeddings 的区别：完全可控，不依赖 LangChain 内部逻辑（无 tiktoken 问题）。
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_BASE")
        )

    def embed_documents(self, texts):
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue
            # [扣费] 调用 Embedding API（批量文本向量化）
            response = self.client.embeddings.create(
                model="text-embedding-v4",
                input=batch,
                dimensions=1024
            )
            all_embeddings.extend([data.embedding for data in response.data])
        return all_embeddings

    def embed_query(self, text):
        # [扣费] 调用 Embedding API（单条查询向量化）
        return self.embed_documents([text])[0]


# ========== 阶段2：建库（自动检测：首次建库，后续复用）==========
PERSIST_DIR = "./chroma_storage"
COLLECTION_NAME = "my_knowledge_base"

# 统一初始化 Embedding（供建库和查询复用）

    # embed
    # 1. 自定义embedding
    # embeddings = CustomEmbeddings()
    # 2. 使用LangChain提供的封装，阿里百炼遵守openai接口规范
    
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
# 3. DashScopeEmbeddings 
# DashScopeEmbeddings 是 LangChain Community 包里的专用封装，但：
# 它的参数名是 dashscope_api_key 而不是 api_key
# 某些新参数（如 dimensions=1024）可能更新不及时，因为百炼 v4 支持 dimensions 是较新的特性
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v4",
#     dashscope_api_key=os.getenv("API_KEY")
# )

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # ✅ 已有库：直接加载，不调用 Embedding API（不扣费）
    print("[INFO] 检测到已有向量库，直接加载，跳过 Embedding 建库...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
else:
    # 💰 首次运行：建库，会对所有 splits 调用 Embedding API（扣费）
    print("[INFO] 首次运行，正在建库并持久化（此步骤产生 Embedding 费用）...")

    # 加载本地HTML文档
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
    """
    结构：
        Document(page_content='...', metadata={'source': '...', title=''})
    """
    docs = loader.load()

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    """
    结构：
    [
        Document(page_content='...', metadata={'source': '...', title=''}), 
        Document(page_content='...', metadata={'source': '...', title=''}),
        Document(page_content='...', metadata={'source': '...', title=''}),
        ...
    ]
    """
    splits = text_splitter.split_documents(docs)

    # 默认内存模式，除非指定persist_directory
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )
# ========== 建库阶段结束 ==========


# ========== 阶段3：查询（每次运行）==========
# 连接已有向量库（不再生成 embedding，不扣 embedding 费）
# 说明：上面 if/else 已经把 vectorstore 创建好了，这里直接复用

# 默认值k是4
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Prompt
prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:""")

llm = ChatOpenAI(
    model='deepseek-v3.2',
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL 链定义：用管道符号 "|" 把多个步骤串成一条生产线
rag_chain = (
    
    # ==================== 第 1 步：组装输入字典 ====================
    # 这是一个 RunnableParallel（并行字典），输入会同时分给两个字段处理
    {
        # "context" 字段：用户问题 → 向量检索 → 格式化文档
        "context": retriever | format_docs,
        #           ↑            ↑
        #           |            └─ 把检索到的 Document 列表拼成一个大字符串
        #           └─ 把用户问题转成向量，去向量库搜 Top-K 相似文档
        
        # "question" 字段：原样透传用户的问题（不做任何处理）
        "question": RunnablePassthrough()
        #           ↑
        #           └─ "通行证"：输入是什么，输出就是什么
    }
    #     原文 10000 字
    #   → 切成 20 段 → 全部存入向量库（这是建库阶段，只做一次）
    #   → 用户提问："什么是任务分解？"
    #   → 检索器去向量库搜："哪 3 段最相关？"
    #   → 只返回第 5 段、第 8 段、第 12 段（可能来自文章不同位置）
    #   → format_docs 只拼接这 3 段（约 1500 字）→ 塞进 Prompt
    # 输出：{"context": "文档A\n\n文档B...", "question": "What is Task Decomposition?"}

    
    # ==================== 第 2 步：填充 Prompt ====================
    | prompt
    # 把上一步的字典塞进 Prompt 模板
    # {context} 和 {question} 占位符被替换成实际内容
    # 输出：一个填充完整的字符串（或 ChatPromptValue 对象）
    
    # ==================== 第 3 步：调用大模型 ====================
    | llm
    # 把填充好的 Prompt 发给 LLM（如 deepseek-v3.2 / GPT-3.5）
    # 输出：AIMessage 对象（包含模型生成的原始响应）
    
    # ==================== 第 4 步：解析输出 ====================
    | StrOutputParser()
    # 把 AIMessage 对象剥壳，只保留 .content 里的纯文本字符串
    # 输出："Task decomposition is a process..."（最终给用户看的答案）
)


# ==================== 阶段5：执行问答 ====================
if __name__ == "__main__":
    # [扣费] 包含检索(embedding) + LLM 生成
    # [运行示例] rag_chain.invoke("What is Task Decomposition?")
    # 
    # 检索参数：k=2（从向量库召回 2 个最相关的文档片段）
    # 注意：k 控制"参考资料的数量"，不控制"回答的句数"
    output = rag_chain.invoke("What is Task Decomposition?")
    print(output)
    
    """
    预期输出示例：
    Task Decomposition is a planning technique where a complex task is broken down into smaller, simpler steps. 
    This is commonly done using methods like Chain of Thought prompting, which instructs the model to "think step by step," or Tree of Thoughts, which explores multiple reasoning paths. 
    It can also be achieved through specific prompts, task instructions, or by outsourcing to an external planner.
    """
