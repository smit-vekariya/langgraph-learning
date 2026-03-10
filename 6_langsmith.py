from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from IPython.display import Image, display
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable

import json


load_dotenv()

@tool
def get_stock_price(symbol:str)-> float:
    ''' return the current  price of stock given the stock symbol
    -param symbol: stock symbol
    -return: current price of stock
    '''
    data = {
        "MSFT":2340.40,
        "AAPL":23.4,
        "RIL":234.4
    }
    return data.get(symbol, 0.0)

tools = [get_stock_price]

llm = init_chat_model("google_genai:gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

@traceable
def chatbot(state:State) -> State:
    message = llm_with_tools.invoke(state["messages"])
    return {"messages":[message]}


builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_node("tool_node", ToolNode(tools))

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition, {"tools": "tool_node", END: END})
builder.add_edge("tool_node", "chatbot_node")
builder.add_edge("chatbot_node", END)

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

state = None

config = {"configurable": {"thread_id": "user1"}}

while True:
    human = input("Human: ")
    if human.lower() in ("exit", "quit"):
        break

    user_message = {"messages": [HumanMessage(content=human)]}

    state = graph.invoke(user_message, config=config)

    msg = state["messages"][-1]

    if isinstance(msg.content, list):
        text = "".join(block["text"] for block in msg.content if block["type"] == "text")
    else:
        text = msg.content

    print("Bot: ", text)