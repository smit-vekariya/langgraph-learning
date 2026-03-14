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
from langgraph.types import interrupt, Command
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

@tool
def buy_stock(symbol:str, quantity:int):
    """Buy a specific quantity of a stock.

    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT)
        quantity: Number of shares to buy

    Returns:
        Confirmation message for the purchase
    """
    return f"You {quantity} stock of {symbol} buy successfully."


tools = [get_stock_price, buy_stock]

llm = init_chat_model("google_genai:gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

@traceable # this decorator use of langsmith
def chatbot(state:State) -> State:
    message = llm_with_tools.invoke(state["messages"])
    return {"messages":[message]}


def route_tools(state: State):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return END

    tool_name = last_message.tool_calls[0]["name"]

    if tool_name == "buy_stock":
        return "human_approval"

    return "tool_node"


def human_approval(state: State):
    """
    Intercepts buy_stock tool calls and asks the user to approve or reject
    each one before the ToolNode executes them.
    """

    last_message = state["messages"][-1]

    tool_call = last_message.tool_calls[0]
    args = tool_call["args"]

    approval = interrupt({
        "question": f"Approve buying {args['quantity']} shares of {args['symbol']}? (Yes/No)"
    })

    if approval.lower() == "yes":
        return  Command(goto='tool_node')

    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content="Trade cancelled by user.")]
        }
    )

builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_node("tool_node", ToolNode(tools))
builder.add_node("human_approval", human_approval)

builder.add_edge(START, "chatbot_node")

builder.add_conditional_edges(
    "chatbot_node",
    route_tools,
    {
        "human_approval": "human_approval",
        "tool_node": "tool_node",
        END: END
    }
)

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

    if "__interrupt__" in state:
        approval = input(state["__interrupt__"][0].value["question"])
        state = graph.invoke(
            Command(resume=approval),
            config=config
        )

    msg = state["messages"][-1]
    print("msg.content", msg.content)

    if isinstance(msg.content, list):
        text = "".join(block["text"] for block in msg.content if block["type"] == "text")
    else:
        text = msg.content

    print("Bot: ", text)