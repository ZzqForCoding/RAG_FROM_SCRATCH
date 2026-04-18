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

load_dotenv()

"""
================================================================================
查询翻译策略（Query Translation）概览
================================================================================
当用户问题模糊、表述单一，导致向量检索召回不足时，可以通过"改写问题"来提升检索效果。
以下是目前主流的几种查询翻译/分解策略：

【1】Multi Query（多角度改写）← 本文件实现
    思路：让 LLM 把同一个问题改写成多个不同措辞的版本，分别去检索。
    特点：不改变问题的粒度，只是"换种说法问"。同一问题、不同角度。
    合并方式：简单并集去重（get_unique_union），不保留排名信息。
    适用：问题表述不够精确，Embedding 语义匹配有偏差时。

【2】子问题分解（Sub-question Decomposition）
    思路：把复杂大问题拆成多个更小的子问题，每个子问题独立检索、独立回答，
          最后把所有子答案汇总成完整答案。
    特点：问题的粒度变小了，从"一个大问题"变成"多个小问题"。
    合并方式：每个子问题有独立的检索和生成链，最终合并的是"答案"而非"文档"。
    适用：问题本身很复杂，包含多个子任务（如"比较A和B的优缺点，再给出建议"）。

【3】逐步回溯法（Step-back Prompting）
    思路：从具体问题上升到更高层次的抽象问题，先回答这个宏观问题作为"背景知识"，
          再基于这个背景来回答用户的原始具体问题。
    特点：问题的抽象层级变了，从"具体"→"宏观"→"具体"。
    合并方式：先检索宏观问题的文档 → LLM生成宏观答案 → 把宏观答案作为
          额外上下文，再检索原始问题的文档 → 最终回答。
    适用：问题涉及大量背景知识，需要先建立共同理解（如"为什么Transformer有用"）。

【4】RAG-Fusion（互逆排序融合）← 下一文件（04）实现
    思路：与 Multi Query 类似，同样生成多个改写查询分别检索。
    核心差异：合并方式不是简单去重，而是用 RRF（Reciprocal Rank Fusion）
          算法对多组检索结果进行智能排序融合。
    特点：保留排名信息！多组查询都排在前列的文档会获得更高权重，最终有序。
    适用：检索结果质量参差不齐，需要更科学的文档排序。

================================================================================
"""

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
        collection_name="my_knowledge_base"  # 必须与 01 文件一致！
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
# 修复③：显式指定 k=2，避免用默认值 k=4（多检索不一定更好，Multi Query 本身已扩大覆盖面）
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ===========================
# 05. Multi Query：一个问题 → 五个改写版本
# ===========================
# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines.
Do not add numbers, bullet points, or any prefix to each question.
Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# 把"一个问题"变成"五个问题的数组"，交给下游批量检索。
generate_queries = (
    prompt_perspectives
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
# 06. 多路结果合并去重
# ===========================
# 为什么需要 dumps / loads？
#   LangChain 的 Document 是自定义对象，不能直接放进 Python 的 set() 做去重。
#   dumps() 把 Document 序列化成 JSON 字符串 → 字符串可以用 set 去重 → loads() 再还原成 Document。
from langchain_core.load import dumps, loads

def get_unique_union(documents: list[list]):
    """
    合并多个检索结果列表，按文档内容去重。
    
    输入：list[list[Document]]
      外层列表 = 每个改写问题的检索结果（5个问题就有5个子列表）
      内层列表 = 该问题检索到的 k 个文档片段
      
    示例输入：
      [
        [doc1, doc2],      # 第1个改写问题的检索结果
        [doc2, doc3],      # 第2个改写问题的检索结果
        [doc1, doc4],      # 第3个改写问题的检索结果
        ...
      ]
    
    输出：list[Document]（去重后的并集）
    """
    # ① 展平：把「列表的列表」压成一维列表，同时把每个 Document 转成字符串
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    # ② 去重：利用 set 的唯一性，自动去掉内容完全相同的文档
    unique_docs = list(set(flattened_docs))
    
    # ③ 还原：把 JSON 字符串重新变回 Document 对象
    return [loads(doc) for doc in unique_docs]


# ===========================
# 07. 组装 Multi Query 检索链
# ===========================
# 数据流示意：
#   "What is task decomposition for LLM agents?"
#     ↓ generate_queries
#   ["What is task decomposition for LLM agents?", 
#    "How do LLM agents break down complex tasks?",
#    "Explain task decomposition in LLM-based systems.", ...]   ← 5个改写版本
#     ↓ retriever.map()
#   [[doc1, doc2], [doc2, doc3], [doc1, doc4], [doc5, doc6], [doc2, doc7]]  ← 5×2=10个结果
#     ↓ get_unique_union
#   [doc1, doc2, doc3, doc4, doc5, doc6, doc7]  ← 去重后的并集（可能7个）
# 
# retriever.map() 的作用：
#   普通 retriever.invoke("问题") → 对一个字符串做检索 → 返回 list[Document]
#   retriever.map() 接受一个字符串列表 → 对列表中每个元素分别调用 retriever → 返回 list[list[Document]]

question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# [扣费] 执行 Multi Query 检索：1次LLM生成5个问题 + 5次Embedding检索
docs = retrieval_chain.invoke({"question": question})


# ===========================
# 08. 组装最终 RAG 链（Multi Query + Generation）
# ===========================
# itemgetter("question") 的作用：
#   从输入字典中提取 "question" 字段的值。
#   例如：输入 {"question": "What is task decomposition?"} → 输出 "What is task decomposition?"
#   它和 lambda x: x["question"] 等价，但写法更简洁。
# RAG Prompt：把检索到的上下文 + 用户问题 塞进模板
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
#   ├─ "context" 分支：retrieval_chain
#   │     接收 question → generate_queries（生成5个改写）
#   │                   → retriever.map()（5次检索）
#   │                   → get_unique_union（合并去重）
#   │     输出：去重后的 Document 列表（自动传给 format_docs 转成字符串）
#   │
#   └─ "question" 分支：itemgetter("question")
#         从输入字典中提取原问题
#         输出："What is task decomposition for LLM agents?"
#   
#   合并成：{"context": "...拼接的文档内容...", "question": "What is task decomposition..."}
#     ↓ prompt
#   填充好的完整提示词
#     ↓ llm
#   AIMessage
#     ↓ StrOutputParser()
#   "Task decomposition is..."（最终答案字符串）

final_rag_chain = (
    {
        # context：复用上面定义的 retrieval_chain（包含 Multi Query 检索 + 去重）
        # 注意：retrieval_chain 的输出是 list[Document]，prompt 中的 {context} 需要字符串。
        # 但这里 LangChain 会自动把 Document 列表转成字符串（通过内置的 format_docs 逻辑），
        # 和 01 文件里手动写 "retriever | format_docs" 效果相同。
        "context": retrieval_chain,
        
        # question：从输入字典里把原问题原样掏出来，透传给 prompt
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# [扣费] 执行完整 RAG：Multi Query 检索 + LLM 生成答案
answer = final_rag_chain.invoke({"question": question})
print(answer)