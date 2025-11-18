# from typing import Annotated, TypedDict
# from langgraph.graph import START, StateGraph, END
# from langgraph.graph.message import  add_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain_core.tools import tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langgraph.prebuilt import ToolNode, tools_condition

# load_dotenv()


# class AgentState(TypedDict):
#     messages: Annotated[list, add_messages]

# @tool
# def web_search(query: str) -> str:
#     """Search Tavily for a query and return maximum 3 results.


#     Args:
#         query: The search query."""
#     tool = TavilySearchResults(max_results=3)
#     search_docs = tool.invoke({'query': query})
#     formatted_search_docs = "\n\n-----------\n\n".join(
#         [
#             f'{doc["title"]}\n{doc["url"]}\n{doc["content"]}\n-----------\n'
#             for doc in search_docs
#         ]
#     )
#     return formatted_search_docs

# tools = [web_search]

# def assistant(state: AgentState):
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )

#     llm_with_tools = llm.bind_tools(tools)

#     response = llm_with_tools.invoke(state["messages"])

#     return {"messages": [AIMessage(role="assistant", content=response.content)]}


# graph = StateGraph(AgentState)


# graph.add_node("assistant", assistant)
# graph.add_node("tools", ToolNode(tools))
# graph.add_edge(START, "assistant")
# graph.add_conditional_edges("assistant", tools_condition)
# graph.add_edge("tools", "assistant")


# agent = graph.compile()


# def chatbot_response(message, history):
#     """Handles the chat interaction with the Gemini API."""
   
#     chats={"messages": []}
#     for msg in history:
#         if msg.get('role') == 'user':
#             chats["messages"].append(HumanMessage(content=msg.get('content')))
#         elif msg.get('role') == 'assistant':
#             chats["messages"].append(AIMessage(content=msg.get('content')))
#     chats["messages"].append(HumanMessage(content=message))
#     response=agent.invoke(chats)
#     for msg in response["messages"]:
#         msg.pretty_print()
#     yield response["messages"][-1].content


# if __name__ == "__main__":
#     response = (agent.invoke({"messages": [HumanMessage(role="user", content="who is the president of france?")]}))
#     print(response["messages"][-1].content)


from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import  add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


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


tools = [web_search]


def assistant(state: AgentState):
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


graph = StateGraph(AgentState)


graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "assistant")


agent = graph.compile()


def chatbot_response(message, history):
    """Handles the chat interaction with the Gemini API."""
    with open("system_prompt.txt", "r") as file:
        system_prompt = file.read()
    sys_msg= SystemMessage(content=system_prompt)
   
    chats={"messages": []}
    chats["messages"].append(sys_msg)
    chats["messages"].append(HumanMessage(content=system_prompt))
    for msg in history:
        if msg.get('role') == 'user':
            chats["messages"].append(HumanMessage(content=msg.get('content')))
        elif msg.get('role') == 'assistant':
            chats["messages"].append(AIMessage(content=msg.get('content')))
    chats["messages"].append(HumanMessage(content=message))
    response=agent.invoke(chats)
    for msg in response["messages"]:
        msg.pretty_print()
    yield response["messages"][-1].content


if __name__ == "__main__":
    response = (agent.invoke({"messages": [HumanMessage(content="who is the president of france?")]}))
    print(response["messages"][-1].content)