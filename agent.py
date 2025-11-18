from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def init_tools():
    # client = MultiServerMCPClient(
    #     {
    #         "weather": {
    #             "url": "http://localhost:8888/mcp",
    #             "transport": "streamable_http",
    #         },
    #     }
    # )
    # mcp_tools = await client.get_tools()
    return [rag_query, web_search] #+ mcp_tools

@tool
def rag_query(query: str) -> str:
    """Query the HR manual database for relevant information."""
    from vector_store import query_vector_store
    
    results = query_vector_store(query, top_k=3)
    
    if not results:
        return "No relevant information found in the HR manual."
    
    formatted_results = []
    for i, (chunk, score) in enumerate(results, 1):
        formatted_results.append(f"Relevant Section {i} (Score: {score:.3f}):\n{chunk}")
    
    return "\n\n".join(formatted_results)

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query."""
    tool = TavilySearchResults(max_results=3)
    search_docs = tool.invoke({'query': query})
    formatted_search_docs = "\n\n-----------\n\n".join(
        [
            f'{doc["title"]}\n{doc["url"]}\n{doc["content"]}\n-----------\n'
            for doc in search_docs
        ]
    )
    return formatted_search_docs

async def main():
    tools = await init_tools()
    print(tools)
    def assistant(state: MessagesState):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
        print(response)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    agent = graph.compile()

    return agent