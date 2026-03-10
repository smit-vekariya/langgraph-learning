from typing import TypedDict
from langgraph.graph import END, StateGraph, START


class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inr: float


def cal_usd(state: PortfolioState)->PortfolioState:
    state["total_usd"] = state["amount_usd"] * 1.08
    return state

def cal_inr(state: PortfolioState)->PortfolioState:
    state["total_inr"] = state["total_usd"] * 85
    return state


builder = StateGraph(PortfolioState)

builder.add_node("cal_usd_node", cal_usd)
builder.add_node("cal_inr_node", cal_inr)

builder.add_edge(START, "cal_usd_node")
builder.add_edge("cal_usd_node", "cal_inr_node")
builder.add_edge("cal_inr_node", END)

graph = builder.compile()

result = graph.invoke({"amount_usd": 100})
print(result)