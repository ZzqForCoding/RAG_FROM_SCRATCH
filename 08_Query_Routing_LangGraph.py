"""
================================================================================
Part 10b: Routing 的 LangGraph 实现（对比 LCEL 版本）
================================================================================
【为什么写这个文件？】
08_Query_Routing.py 用 LCEL 实现了路由，但 LCEL 的隐式传参很难调试。
这个文件用 LangGraph 重写同样的逻辑，展示另一种更可控的实现方式。

【核心区别】
  LCEL: 声明式管道，数据在 | 运算符之间隐式流转，出问题难定位
  LangGraph: 显式状态机，每个节点从 state 取数据、返回更新，流程清清楚楚

【LangGraph 核心概念（4个）】
  1. State（状态）: 一个 TypedDict，整个图的"共享数据池"
  2. Node（节点）: 一个 Python 函数，接收 state，做一件事，返回对 state 的更新
  3. Edge（边）: 连接节点，控制"下一步去哪"
  4. Conditional Edge（条件边）: 根据 state 的值动态决定下一步走哪个节点

【为什么 State 里没有 messages: Annotated[Sequence[BaseMessage], add_messages]？】
  因为你的生产项目（ChatGraph）是对话式 Agent，需要多轮对话历史，所以用 messages。
  但这个文件是单轮问答流水线：问题 → 路由 → 检索 → 生成 → 结束，
  不需要累积对话历史，所以直接用字段名（question/context/answer）更直观。

  如果以后需要多轮对话，再加 messages 也不迟。
================================================================================
"""

import os
import warnings
from typing import Literal, TypedDict

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*beta.*", category=UserWarning)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph 的导入
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

load_dotenv()


# ==============================================================================
# 公共工具函数
# ==============================================================================
def format_docs(docs):
    """将 Document 列表格式化为 \n\n 分隔的字符串。"""
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================================================================
# 配置 Embedding 和 LLM
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
# 构建多领域知识库（和 08 文件完全一致）
# ==============================================================================
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
闭包使得函数可以"记住"并访问它被创建时的词法作用域，即使这个函数在当前作用域之外执行。""",
        metadata={"source": "js_docs"},
    ),
]

golang_docs = [
    Document(
        page_content="""Go 语言的 goroutine 是一种轻量级线程，由 Go 运行时（runtime）管理。
与操作系统线程相比，goroutine 的创建和切换成本极低。使用 go 关键字即可启动一个新的 goroutine。""",
        metadata={"source": "golang_docs"},
    ),
    Document(
        page_content="""Go 的 channel 是 goroutine 之间通信和同步的主要机制。
channel 遵循 CSP（Communicating Sequential Processes）模型。""",
        metadata={"source": "golang_docs"},
    ),
]

all_docs = python_docs + js_docs + golang_docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(all_docs)

PERSIST_DIR = "./chroma_routing_storage"
import shutil
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="multi_language_docs",
    persist_directory=PERSIST_DIR,
)

python_retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"source": "python_docs"}})
js_retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"source": "js_docs"}})
golang_retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"source": "golang_docs"}})


# ==============================================================================
# LLM Router（和 08 文件完全一致）
# ==============================================================================
class RouteQuery(BaseModel):
    """将用户查询路由到最合适的数据源。"""
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="根据用户问题内容，判断应该使用哪个数据源来回答最相关",
    )


parser = JsonOutputParser(pydantic_object=RouteQuery)

router_system = """你是一个查询路由专家。你的任务是根据用户的问题，判断应该使用哪个数据源来回答。
可选的数据源：
- python_docs: Python 编程语言相关的问题
- js_docs: JavaScript 编程语言相关的问题
- golang_docs: Go 编程语言相关的问题
请仔细分析问题的语义，选择最相关的数据源。只输出 JSON，不要解释原因。"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system + "\n\n{format_instructions}"),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions()).partial(format_instructions=parser.get_format_instructions())

router_chain = router_prompt | llm | parser


# 三个领域的回答 Prompt
python_prompt = ChatPromptTemplate.from_template("""你是一个 Python 专家。请根据以下检索到的文档，回答用户的问题。
检索到的文档：
{context}
用户问题：{question}
请用中文回答：""")

js_prompt = ChatPromptTemplate.from_template("""你是一个 JavaScript 专家。请根据以下检索到的文档，回答用户的问题。
检索到的文档：
{context}
用户问题：{question}
请用中文回答：""")

golang_prompt = ChatPromptTemplate.from_template("""你是一个 Go 语言专家。请根据以下检索到的文档，回答用户的问题。
检索到的文档：
{context}
用户问题：{question}
请用中文回答：""")


# ==============================================================================
# LangGraph 实现开始
# ==============================================================================
"""
【LangGraph 核心思想】
把 pipeline 拆成多个独立的"节点函数"，每个函数只做一件事：
  - 从 state 里取需要的数据
  - 执行逻辑
  - 返回一个 dict，告诉 LangGraph "我要更新 state 里的哪些字段"

LangGraph 会自动把返回值 merge 到 state 里，然后沿着边走到下一个节点。
"""

# -----------------------------
# 1. 定义 State（共享数据池）
# -----------------------------
class RoutingState(TypedDict):
    """
    整个 Graph 的共享状态。每个节点都能读取这些字段，也能更新它们。
    
    为什么不加 messages: Annotated[...]？
      因为这是单轮问答，不是对话 Agent。
      你的 ChatGraph 有 messages，是因为要累积多轮对话历史（用户说→AI答→用户追问→...）。
      这里只需要一次流转：question → datasource → context → answer，一次 invoke 就出结果。
      所以用直观的字段名就够了。
    """
    question: str     # 用户问题（输入）
    datasource: str   # 路由判断结果（router_node 设置）
    context: str      # 检索到的文档内容（search 节点设置）
    answer: str       # 最终答案（generate 节点设置）


# -----------------------------
# 2. 定义节点函数
# -----------------------------
"""
每个节点函数的签名都是：def node_name(state: RoutingState) -> dict
返回值是一个 dict，key 是 state 中的字段名，value 是要更新的值。
LangGraph 会自动把这个 dict merge 到全局 state 中。
"""

def router_node(state: RoutingState) -> dict:
    """
    节点 1：路由判断。
    从 state 里取 question，调用 LLM 判断应该路由到哪个数据源。
    返回 {"datasource": "python_docs"} 之类的，更新 state。
    """
    print(f"\n[Node: router] 收到问题: {state['question']}")
    
    result = router_chain.invoke({"question": state["question"]})
    ds = result["datasource"]
    
    print(f"[Node: router] 路由判断: {ds}")
    return {"datasource": ds}


def python_search(state: RoutingState) -> dict:
    """节点 2a：Python 文档检索。"""
    print(f"[Node: python_search] 检索中...")
    docs = python_retriever.invoke(state["question"])
    context = format_docs(docs)
    print(f"[Node: python_search] 检索到 {len(docs)} 条文档")
    return {"context": context}


def js_search(state: RoutingState) -> dict:
    """节点 2b：JavaScript 文档检索。"""
    print(f"[Node: js_search] 检索中...")
    docs = js_retriever.invoke(state["question"])
    context = format_docs(docs)
    print(f"[Node: js_search] 检索到 {len(docs)} 条文档")
    return {"context": context}


def golang_search(state: RoutingState) -> dict:
    """节点 2c：Go 文档检索。"""
    print(f"[Node: golang_search] 检索中...")
    docs = golang_retriever.invoke(state["question"])
    context = format_docs(docs)
    print(f"[Node: golang_search] 检索到 {len(docs)} 条文档")
    return {"context": context}


def generate(state: RoutingState) -> dict:
    """
    节点 3：生成答案。
    根据 state["datasource"] 选择对应的 Prompt，结合 context 和 question 生成回答。
    """
    print(f"[Node: generate] 正在生成答案...")
    
    ds = state["datasource"]
    if ds == "python_docs":
        prompt = python_prompt
    elif ds == "js_docs":
        prompt = js_prompt
    else:
        prompt = golang_prompt
    
    messages = prompt.format_messages(
        context=state["context"],
        question=state["question"],
    )
    response = llm.invoke(messages)
    
    print(f"[Node: generate] 答案生成完成")
    return {"answer": response.content}


# -----------------------------
# 3. 定义条件边函数
# -----------------------------
"""
条件边和普通边的区别：
  - 普通边: graph.add_edge("A", "B")  → A 做完必走 B
  - 条件边: graph.add_conditional_edges("A", choose, {"x": "B", "y": "C"})
            → A 做完后，调用 choose(state) 函数，返回值决定走 B 还是 C
"""

def choose_route(state: RoutingState) -> str:
    """
    条件边函数。返回值必须是字符串，且要和 add_conditional_edges 里的 mapping key 匹配。
    
    比如返回 "python_docs"，Graph 就会走到 "python_search" 节点。
    """
    return state["datasource"]


# -----------------------------
# 4. 组装图
# -----------------------------
"""
构建流程：
  START → router → [条件分支] → python_search / js_search / golang_search → generate → END
"""

graph = StateGraph(RoutingState)

# 注册所有节点
graph.add_node("router", router_node)
graph.add_node("python_search", python_search)
graph.add_node("js_search", js_search)
graph.add_node("golang_search", golang_search)
graph.add_node("generate", generate)

# 普通边：START → router
graph.add_edge(START, "router")

# 条件边：router 做完后，根据 choose_route 的返回值分流
graph.add_conditional_edges(
    "router",           # 从哪个节点出发
    choose_route,       # 调用这个函数决定下一步
    {                   # 返回值 → 目标节点的映射
        "python_docs": "python_search",
        "js_docs": "js_search",
        "golang_docs": "golang_search",
    }
)

# 普通边：三个检索节点都指向 generate
graph.add_edge("python_search", "generate")
graph.add_edge("js_search", "generate")
graph.add_edge("golang_search", "generate")

# 普通边：generate → END
graph.add_edge("generate", END)

# 编译图
app = graph.compile()


# -----------------------------
# 5. 运行 + 可视化
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("【LangGraph Routing 测试】")
    print("=" * 60)
    
    # 可以打印 Mermaid 语法，粘贴到 https://mermaid.live 看流程图
    print("\n[流程图 Mermaid 语法]:")
    print(app.get_graph().draw_mermaid())
    print()
    
    test_questions = [
        "Python 的装饰器是什么，怎么使用？",
        "JavaScript 的 Promise 是什么？",
        "Go 的 channel 怎么用？",
    ]
    
    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"问题: {q}")
        print("=" * 60)
        
        # invoke 只需要传入初始 state，LangGraph 会自动走完整个流程
        result = app.invoke({"question": q})
        
        print(f"\n最终答案:\n{result['answer']}")
        print("-" * 60)
