from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state:State) -> State:
    return {"messages":[llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)

builder.add_edge(START, "chatbot_node")
builder.add_edge("chatbot_node", END)

graph = builder.compile()


state = None

while True:
    human = input("Human: ")
    if human.lower() in ("exit", "quit"):
        break

    if state is None:
        state: State = {"messages": [{"role":"user", "content":human}]}
    else:
        state["messages"].append({"role":"user", "content":human})

    state = graph.invoke(state)

    print("Bot: ", state["messages"][-1].content)