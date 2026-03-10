from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph, START


class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_amount: float
    target_currency: Literal["inr", "eur"]


def cal_usd(state: PortfolioState)->PortfolioState:
    state["total_usd"] = state["amount_usd"] * 1.08
    return state

def cal_inr(state: PortfolioState)->PortfolioState:
    state["total_amount"] = state["total_usd"] * 85
    return state

def cal_eur(state: PortfolioState)->PortfolioState:
    state["total_amount"] = state["total_usd"] * 0.85
    return state

def select_currency(state: PortfolioState)->str:
    return state["target_currency"]


builder = StateGraph(PortfolioState)

builder.add_node("cal_usd_node", cal_usd)
builder.add_node("cal_inr_node", cal_inr)
builder.add_node("cal_eur_node", cal_eur)

builder.add_edge(START, "cal_usd_node")
builder.add_conditional_edges("cal_usd_node", select_currency, {
    "inr": "cal_inr_node",
    "eur": "cal_eur_node",
})
builder.add_edge(["cal_inr_node", "cal_eur_node"], END)

graph = builder.compile()

result = graph.invoke({"amount_usd": 100, "target_currency": "eur"})
print(result)