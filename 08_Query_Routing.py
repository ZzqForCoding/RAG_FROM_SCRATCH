"""
================================================================================
Part 10: Routing（查询路由 / 意图路由）
================================================================================
【核心问题】
当系统拥有多个知识库、多个 Prompt 模板或多个处理链时，
如何将用户问题准确地路由到最合适的那一个？

如果盲目地把所有问题丢进同一个知识库检索，结果往往是：
  - 技术问题检索到了 HR 政策文档
  - Python 问题检索到了 JavaScript 代码片段
  - 简单问题被用学术论文的方式回答，晦涩难懂

【Routing 的本质】
在"检索"之前，先做一个轻量级的"意图判断"，决定走哪条路。

【两种路由方式对比】
  ┌─────────────────┬──────────────────────────────┬─────────┬────────────────────┐
  │  路由方式        │  原理                         │  成本   │  适用场景           │
  ├─────────────────┼──────────────────────────────┼─────────┼────────────────────┤
  │  LLM Routing    │  用 LLM 判断意图，结构化输出   │  高     │  领域差异大、需要   │
  │  (结构化输出)    │  选择数据源/Prompt             │ (LLM)  │  语义理解才能区分   │
  ├─────────────────┼──────────────────────────────┼─────────┼────────────────────┤
  │  Embedding      │  用向量相似度匹配查询与预设     │  低     │  领域特征明显、可   │
  │  Routing        │  Prompt/描述文本               │(Embedding│ 用关键词/标签区分  │
  └─────────────────┴──────────────────────────────┴─────────┴────────────────────┘

【与前五种查询翻译策略的关系】
  前五种策略（03~07）解决的是"如何用一个问题更好地检索一个知识库"，
  Routing 解决的是"有多个知识库/处理链时，该用哪一个"。
  二者是【正交】的：先路由选库，再用查询翻译策略优化检索。

【完整流程示意】
  用户问题: "Python 的 asyncio 怎么用？"
      ↓
  [Router 判断意图] ──→ python_docs
      ↓
  [使用 python_docs 的 Retriever 检索]
      ↓
  [用 Python 专家的 Prompt 生成回答]

【六种查询处理策略总览】
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │   策略          │   抽象方向 / 操作方式                                 │
  ├─────────────────┼──────────────────────────────────────────────────────┤
  │  Multi Query    │   水平方向：同一粒度，多种表述（03）                  │
  │  RAG-Fusion     │   水平方向：同上 + RRF 重排序（04）                   │
  │  Decomposition  │   向下拆解：大问题 → 多个小问题（05）                 │
  │  Step Back      │   向上抽象：具体问题 → 更宏观的问题（06）             │
  │  HyDE           │   空间转换：问题 → 假设文档（07）                     │
  │  Routing        │   意图分流：多库/多链 → 选最合适的一个（本文件）      │
  └─────────────────┴──────────────────────────────────────────────────────┘

【适用场景】
  ✅ 多领域知识库：公司同时有技术文档、产品手册、HR 政策
  ✅ 多语言支持：同一系统需要处理中文/英文/日文查询，路由到对应语言库
  ✅ 多角色回答：同一问题需要"专家模式"或"通俗模式"不同回答风格
  ✅ 多数据源混合：向量库 + SQL 数据库 + API，根据问题类型选择数据源
  ✅ 多版本文档：v1/v2/v3 不同版本的产品文档

【不适用场景】
  ❌ 只有一个知识库（路由是多余的，增加延迟和成本）
  ❌ 领域边界非常模糊（LLM 也难以判断，路由准确率会下降）
  ❌ 对延迟极度敏感且领域可简单规则匹配（用 Embedding Routing 或规则路由）

【参考资料】
  - LangChain Routing to Multiple Indexes:
    https://python.langchain.com/docs/use_cases/query_analysis/techniques/routing
  - LangChain LCEL Routing:
    https://python.langchain.com/docs/expression_language/how_to/routing
  - LangChain Embedding Router:
    https://python.langchain.com/docs/expression_language/cookbook/embedding_router
================================================================================
"""

import os
import warnings
from typing import Literal

import numpy as np
from dotenv import load_dotenv

# 过滤掉 LangChain 的弃用警告和 beta 警告，终端更干净
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*", category=UserWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 .env 环境变量
load_dotenv()


# ==============================================================================
# 公共工具函数
# ==============================================================================
def format_docs(docs):
    """将 Document 列表格式化为 \n\n 分隔的字符串，用于 Prompt 中的 {context}。"""
    return "\n\n".join(doc.page_content for doc in docs)


def cosine_similarity(a, b):
    """
    计算两组向量之间的余弦相似度。
    a: (m, dim) 数组, b: (n, dim) 数组
    返回: (m, n) 相似度矩阵
    """
    a = np.array(a)
    b = np.array(b)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


# ==============================================================================
# 01. 配置 Embedding 和 LLM（公共部分）
# ==============================================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
    check_embedding_ctx_length=False,
)

llm = ChatOpenAI(
    model='deepseek-v3.2',
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
)


# ==============================================================================
# 02. 构建多领域知识库（用于演示 LLM-based Routing）
# ==============================================================================
"""
为了演示"根据问题路由到不同知识库"，我们模拟三个编程语言领域的文档：
- python_docs: Python 相关
- js_docs: JavaScript 相关
- golang_docs: Go 相关

实际项目中，这三个可能是：
  - 三个独立的 ChromaDB collection
  - 三个独立的向量库实例
  - 同一个 collection 中用 metadata filter 区分（本例采用这种方式）
"""

python_docs = [
    Document(
        page_content="""Python 的 asyncio 模块提供了编写并发代码的基础设施。
它使用 async/await 语法，使得异步编程看起来像同步代码一样直观。
事件循环（Event Loop）是 asyncio 的核心，负责调度和执行协程。""",
        metadata={"source": "python_docs"},
    ),
    Document(
        page_content="""Python 装饰器（Decorator）是一种高阶函数，
可以在不修改原函数源代码的情况下，为函数添加额外的功能。
常见的内置装饰器有 @staticmethod、@classmethod、@property。""",
        metadata={"source": "python_docs"},
    ),
]

js_docs = [
    Document(
        page_content="""JavaScript 的 Promise 对象代表了一个异步操作的最终完成（或失败）及其结果值。
它有三种状态：pending（进行中）、fulfilled（已成功）和 rejected（已失败）。
Promise.prototype.then() 和 Promise.prototype.catch() 是处理异步结果的主要方法。""",
        metadata={"source": "js_docs"},
    ),
    Document(
        page_content="""JavaScript 闭包（Closure）是指有权访问另一个函数作用域中变量的函数。
闭包使得函数可以"记住"并访问它被创建时的词法作用域，即使这个函数在当前作用域之外执行。
这是 JavaScript 中实现模块模式和数据私有化的重要机制。""",
        metadata={"source": "js_docs"},
    ),
]

golang_docs = [
    Document(
        page_content="""Go 语言的 goroutine 是一种轻量级线程，由 Go 运行时（runtime）管理。
与操作系统线程相比，goroutine 的创建和切换成本极低，一个 Go 程序可以轻松启动成千上万个 goroutine。
使用 go 关键字即可启动一个新的 goroutine。""",
        metadata={"source": "golang_docs"},
    ),
    Document(
        page_content="""Go 的 channel 是 goroutine 之间通信和同步的主要机制。
channel 遵循 CSP（Communicating Sequential Processes）模型，
通过 "不要通过共享内存来通信，而要通过通信来共享内存" 的理念实现并发安全。""",
        metadata={"source": "golang_docs"},
    ),
]

# 合并所有文档
all_docs = python_docs + js_docs + golang_docs

# 切分（本地操作，不扣费）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(all_docs)

# 建立向量库（Embedding 扣费，但文档很少，费用极低）
PERSIST_DIR = "./chroma_routing_storage"


if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="multi_language_docs",
        persist_directory=PERSIST_DIR,
    )
else:
    print("[INFO] 首次运行，正在建库并持久化（此步骤产生 Embedding 费用）...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="multi_language_docs",
        persist_directory=PERSIST_DIR
    )

# 创建三个带 filter 的 retriever，分别对应三个领域
python_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2, "filter": {"source": "python_docs"}}
)
js_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2, "filter": {"source": "js_docs"}}
)
golang_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2, "filter": {"source": "golang_docs"}}
)


# ==============================================================================
# Part A: LLM-based Routing（结构化输出路由）
# ==============================================================================
"""
【原理】
利用 LLM 的语义理解能力，让模型判断用户问题属于哪个领域。
通过 `with_structured_output()` 绑定 Pydantic 模型，强制 LLM 输出结构化的路由决策
（如 {"datasource": "python_docs"}），而不是自由文本。

【优点】
  - 灵活：可以处理模糊、复杂、需要语义理解才能区分的查询
  - 可扩展：增加新领域只需修改 Literal 枚举和分支逻辑
  - 准确：LLM 理解上下文，比关键词匹配更精准

【缺点】
  - 成本高：每次查询都需要一次额外的 LLM 调用
  - 延迟高：路由本身增加了响应时间
  - 依赖 LLM 能力：如果 LLM 判断错误，后续全错

【适用】领域边界模糊、查询表述多样、需要语义理解的场景。
"""

from pydantic import BaseModel, Field


# -----------------------------
# A1. 定义路由数据结构
# -----------------------------
class RouteQuery(BaseModel):
    """将用户查询路由到最合适的数据源。"""

    # 约束datasource类型只能是["python_docs", "js_docs", "golang_docs"]
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ..., # 表示必填，调用时必须传值
        description="根据用户问题内容，判断应该使用哪个数据源来回答最相关",
    )


# -----------------------------
# A2. 创建带结构化输出的 LLM
# -----------------------------
# `with_structured_output` 会告诉 LLM："请按照 RouteQuery 的 schema 输出 JSON"
# LLM 实际输出的是 JSON 字符串，LangChain 自动解析为 Pydantic 对象
structured_llm = llm.with_structured_output(RouteQuery)

# -----------------------------
# A3. 定义路由 Prompt
# -----------------------------
router_system = """你是一个查询路由专家。你的任务是根据用户的问题，判断应该使用哪个数据源来回答。

可选的数据源：
- python_docs: Python 编程语言相关的问题（如 asyncio、装饰器、列表推导式等）
- js_docs: JavaScript 编程语言相关的问题（如 Promise、闭包、箭头函数等）
- golang_docs: Go 编程语言相关的问题（如 goroutine、channel、接口等）

请仔细分析问题的语义，选择最相关的数据源。只输出选择结果，不要解释原因。"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system),
        ("human", "{question}"),
    ]
)

# -----------------------------
# A4. 构建路由链
# -----------------------------
# router_chain: 输入 {"question": ...} → 输出 RouteQuery(datasource="...")
router_chain = router_prompt | structured_llm


# -----------------------------
# A5. 测试路由判断（可先运行此部分验证路由准确性）
# -----------------------------
# if __name__ == "__main__":
#     print("=" * 60)
#     print("【LLM Router 测试】判断问题应该路由到哪个数据源")
#     print("=" * 60)

#     test_questions = [
#         "Python 的 asyncio 怎么用？",
#         "JavaScript 的 Promise 是什么？",
#         "Go 语言的 goroutine 和 channel 怎么配合？",
#         "如何用装饰器给 Python 函数添加日志？",
#         "JS 里的闭包会导致内存泄漏吗？",
#     ]

#     for q in test_questions:
#         result = router_chain.invoke({"question": q})
#         print(f"  问题: {q}")
#         print(f"  → 路由到: {result.datasource}")
#         print()


# -----------------------------
# A6. 定义分支选择函数 + 完整 RAG 链
# -----------------------------
"""
路由判断完成后，需要根据结果选择对应的 retriever 和 prompt，
然后走完整的检索 → 生成流程。

这里使用 RunnableLambda 将路由结果映射到不同的处理链。
"""

# 定义三个领域的回答 Prompt
python_prompt = ChatPromptTemplate.from_template("""你是一个 Python 专家。请根据以下检索到的文档，回答用户的问题。
如果文档中没有相关信息，请诚实说明你不知道。

检索到的文档：
{context}

用户问题：{question}

请用中文回答：""")

js_prompt = ChatPromptTemplate.from_template("""你是一个 JavaScript 专家。请根据以下检索到的文档，回答用户的问题。
如果文档中没有相关信息，请诚实说明你不知道。

检索到的文档：
{context}

用户问题：{question}

请用中文回答：""")

golang_prompt = ChatPromptTemplate.from_template("""你是一个 Go 语言专家。请根据以下检索到的文档，回答用户的问题。
如果文档中没有相关信息，请诚实说明你不知道。

检索到的文档：
{context}

用户问题：{question}

请用中文回答：""")


# 路由 → 分支映射
def choose_route(result: RouteQuery):
    """
    根据 LLM Router 返回的结构化结果，选择对应的 (retriever, prompt) 组合。
    返回一个 Runnable（链），可以直接在 LCEL 中拼接。
    """
    ds = result.datasource.lower()
    if "python_docs" in ds:
        return (
            {"context": itemgetter("question") | python_retriever | format_docs, "question": itemgetter("question")}
            | python_prompt | llm | StrOutputParser()
        )
    elif "js_docs" in ds:
        return (
            {"context": itemgetter("question") | js_retriever | format_docs, "question": itemgetter("question")}
            | js_prompt | llm | StrOutputParser()
        )
    else:  # golang_docs
        return (
            {"context": itemgetter("question") | golang_retriever | format_docs, "question": itemgetter("question")}
            | golang_prompt | llm | StrOutputParser()
        )


# 构建完整的 LLM Routing RAG 链
# 注意：choose_route 返回的是一个链（Runnable），需要用它来处理原始输入
# 这里我们手动包装一下，让链能正确传递 question

def route_and_answer(inputs: dict) -> str:
    """
    完整的路由 + 回答流程：
    1. 用 router_chain 判断领域
    2. 用 choose_route 选择对应的处理链
    3. 执行处理链，返回答案
    """
    # Step 1: 路由判断
    route_result = router_chain.invoke({"question": inputs["question"]})
    print(f"  [路由判断] 问题路由到: {route_result.datasource}")

    # Step 2: 选择对应的链
    selected_chain = choose_route(route_result)

    # Step 3: 执行检索 + 生成
    answer = selected_chain.invoke(inputs)
    return answer


# -----------------------------
# A7. 测试完整链路
# -----------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("【LLM-based Routing 完整链路测试】")
    print("=" * 60)

    test_q = "Python 的装饰器是什么，怎么使用？"
    print(f"问题: {test_q}\n")
    answer = route_and_answer({"question": test_q})
    print(f"\n回答:\n{answer}")

    print("\n" + "-" * 60)

    test_q2 = "Go 的 channel 怎么用？"
    print(f"问题: {test_q2}\n")
    answer2 = route_and_answer({"question": test_q2})
    print(f"\n回答:\n{answer2}")


# ==============================================================================
# Part B: Embedding-based Routing（语义相似度路由）
# ==============================================================================
"""
【原理】
不用 LLM 做判断，而是预先将每个领域的"代表文本"（Prompt Template 或领域描述）
转换为 Embedding，存为向量。用户查询到来时：
  1. 将查询也转为 Embedding
  2. 计算查询与所有领域代表文本的余弦相似度
  3. 选择相似度最高的领域

【优点】
  - 成本低：只需要 Embedding API 调用，不需要 LLM
  - 延迟低：向量计算很快
  - 稳定：不受 LLM 幻觉影响

【缺点】
  - 泛化能力弱：如果查询表述与预定义的领域描述差异大，可能匹配错误
  - 需要精心设计领域描述文本
  - 难以处理"跨领域"或"边界模糊"的查询

【适用】领域特征明显、可以用少量代表性文本概括、对成本和延迟敏感的场景。
"""

# -----------------------------
# B1. 定义多领域 Prompt Templates
# -----------------------------
"""
这里我们定义两个截然不同的回答风格：
- physics_template: 物理教授风格，简洁易懂
- math_template: 数学家风格，拆解问题、逻辑严谨

用户问题到来时，通过 Embedding 相似度判断问题更像"物理题"还是"数学题"，
然后选择对应的 Prompt Template 来回答。

实际项目中，这可以是：
  - 不同领域的专家 Prompt（法律 / 医学 / 工程）
  - 不同风格的回答 Prompt（严谨学术 / 通俗科普）
  - 不同语言的回答 Prompt（中文 / 英文 / 日文）
"""

physics_template = """你是一位非常聪明的物理学教授。
你擅长用简洁易懂的方式回答物理学问题。
当你不知道答案时，你会诚实承认。

问题：
{query}"""

math_template = """你是一位非常优秀的数学家。你擅长回答数学问题。
你的优势在于能够将复杂的问题拆解成组成部分，分别回答，
然后将它们组合起来回答更宏观的问题。

问题：
{query}"""

# 将所有模板收集起来
template_names = ["physics", "math"]
prompt_templates = [physics_template, math_template]

# -----------------------------
# B2. 预计算所有模板的 Embedding
# -----------------------------
# 这一步是"离线"的，在系统启动时执行一次即可
print("\n预计算 Prompt Templates 的 Embedding...")
prompt_embeddings = embeddings.embed_documents(prompt_templates)
print(f"  共 {len(prompt_embeddings)} 个模板，每个维度 {len(prompt_embeddings[0])}")


# -----------------------------
# B3. 定义 Prompt Router 函数
# -----------------------------
def prompt_router(inputs: dict):
    """
    根据用户查询的 Embedding 与预设模板 Embedding 的相似度，
    选择最匹配的 Prompt Template，返回对应的 PromptTemplate 对象。
    """
    query = inputs["query"]

    # 1. 将用户查询转为 Embedding
    query_embedding = embeddings.embed_query(query)

    # 2. 计算余弦相似度
    # cosine_similarity 返回 (1, n_templates) 矩阵，取第一个元素
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]

    # 3. 找到最相似的模板索引
    best_idx = int(similarity.argmax())
    best_template = prompt_templates[best_idx]
    best_name = template_names[best_idx]

    print(f"  [Embedding Router] 查询: '{query}'")
    print(f"  [Embedding Router] 相似度 — physics: {similarity[0]:.4f}, math: {similarity[1]:.4f}")
    print(f"  [Embedding Router] 选择模板: {best_name}")

    # 4. 返回对应的 PromptTemplate 对象（可在 LCEL 链中直接使用）
    return PromptTemplate.from_template(best_template)


# -----------------------------
# B4. 构建 Embedding Routing 完整链
# -----------------------------
"""
链的结构：
  用户查询 → embed_query → 余弦相似度匹配 → 选择 PromptTemplate → LLM → 输出

注意：prompt_router 返回的是 PromptTemplate 对象，
LCEL 的 | 运算符会自动调用它的 invoke 方法，将上游输出作为输入传入。
"""

embedding_rag_chain = (
    {"query": RunnablePassthrough()}           # 透传用户查询
    | RunnableLambda(prompt_router)             # 路由选择 PromptTemplate
    | llm                                       # 用选中的 Prompt + LLM 生成
    | StrOutputParser()                         # 解析为字符串
)


# -----------------------------
# B5. 测试 Embedding Routing
# -----------------------------
# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("【Embedding-based Routing 测试】")
#     print("=" * 60)

#     physics_question = "黑洞是什么？它是怎么形成的？"
#     print(f"\n问题（物理）: {physics_question}")
#     answer = embedding_rag_chain.invoke(physics_question)
#     print(f"\n回答:\n{answer}")

#     print("\n" + "-" * 60)

#     math_question = "如何证明费马大定理？"
#     print(f"\n问题（数学）: {math_question}")
#     answer = embedding_rag_chain.invoke(math_question)
#     print(f"\n回答:\n{answer}")


# ==============================================================================
# 两种路由方式对比总结
# ==============================================================================
"""
┌──────────────────────┬─────────────────────────────┬─────────────────────────────┐
│ 对比维度              │ LLM-based Routing           │ Embedding-based Routing     │
├──────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ 核心机制              │ LLM 语义理解 + 结构化输出    │ 向量相似度计算               │
│ 每次查询成本          │ 高（需要 LLM API 调用）      │ 低（仅需 Embedding API）     │
│ 响应延迟              │ 高（LLM 推理时间）           │ 低（向量计算毫秒级）         │
│ 准确率                │ 高（理解上下文和隐含意图）   │ 中（依赖预定义文本质量）     │
│ 可扩展性              │ 高（加领域只需改枚举）       │ 中（需要为新领域写描述文本） │
│ 处理模糊查询          │ 强                          │ 弱                          │
│ 处理跨领域查询        │ 可输出"最可能"的一个         │ 可能硬匹配到错误的领域       │
│ 实现复杂度            │ 低（代码简洁）               │ 低（代码简洁）               │
├──────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ 推荐场景              │ • 领域差异大、语义复杂        │ • 领域特征明显               │
│                      │ • 预算充足、准确率优先        │ • 高并发、低延迟要求         │
│                      │ • 领域频繁变化               │ • 预算有限                   │
│                      │ • 查询表述非常多样           │ • 查询与领域关键词高度相关   │
└──────────────────────┴─────────────────────────────┴─────────────────────────────┘

【混合策略】
实际生产中可以两者结合：
  1. 先用 Embedding Router 做快速初筛（成本低）
  2. 当初筛的置信度（相似度差距）不够大时，再用 LLM Router 做二次确认
  3. 或者：Embedding Router 用于"明显属于某领域"的查询，
     LLM Router 用于"边界模糊"的查询

【更进一步：语义路由器（Semantic Router）】
市面上已有专门的开源库如 `semantic-router`，它本质上就是 Embedding-based Routing
的工程化封装，支持：
  - 动态添加/删除路由
  - 阈值控制（相似度低于阈值时走默认路由）
  - 多路返回（一个问题可能匹配多个领域）
  - 本地 Embedding 模型（不依赖 API）
  
GitHub: https://github.com/aurelio-labs/semantic-router
"""


# ==============================================================================
# 思考题
# ==============================================================================
"""
【思考题】

1. LLM Router 的 Prompt 设计有什么讲究？
   → system prompt 中需要清晰描述每个领域的特征和边界。
   → 可以用 Few-Shot 示例（如 06_Step_Back 的做法）来提升判断准确率。
   → 如果领域很多（10+），Literal 枚举会变得很长，此时可以考虑分层路由：
     先粗分大类（技术 / 业务 / HR），再细分小类（Python / JS / Go）。

2. Embedding Router 的领域描述文本怎么写最好？
   → 要写得"有代表性"，能覆盖该领域的核心关键词和典型问题模式。
   → 可以用该领域真实用户查询的聚合（如 TF-IDF 关键词）来生成描述。
   → 可以维护多段描述文本，取平均 Embedding 作为该领域的代表向量。

3. 如果一个问题同时涉及多个领域怎么办？
   → LLM Router：可以扩展 schema，让 datasource 支持列表（多选）。
   → Embedding Router：可以返回 Top-K 个领域，然后并行检索多个库后融合。
   → 这实际上就演变成了"多路召回 + 融合"的架构，与 RAG-Fusion（04）思路相通。

4. Routing 和 Agent 的关系是什么？
   → Routing 是 Agent 的一个子功能：Agent 决定"做什么"，Routing 决定"去哪里做"。
   → 在 ReAct / Tool-using Agent 中，Routing 相当于"工具选择"（Tool Selection）。
   → 当路由逻辑变得复杂（需要考虑历史对话、用户画像、业务规则），
     简单的 Literal 枚举就不够用了，需要升级为真正的 Agent 决策。

5. 除了 LLM 和 Embedding，还有什么路由方式？
   → 规则路由：正则表达式 / 关键词匹配（最快最准，但维护成本高）。
   → 分类器路由：训练一个小型文本分类模型（BERT 等），本地推理。
   → 混合路由：规则 → Embedding → LLM，层层兜底，成本和准确率平衡。
"""
