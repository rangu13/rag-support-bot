from langgraph.graph import StateGraph
from retriever import get_retriever, generate_answer
from hitl import human_escalation

class State(dict):
    pass


def process_node(state):
    query = state["query"]

    retriever = get_retriever()
    docs = retriever.get_relevant_documents(query)

    if not docs:
        state["escalate"] = True
        return state

    answer = generate_answer(query, docs)

    # Simple confidence check
    if "I don't know" in answer or len(answer) < 20:
        state["escalate"] = True
    else:
        state["response"] = answer
        state["escalate"] = False

    return state


def hitl_node(state):
    response = human_escalation(state["query"])
    state["response"] = response
    return state


def build_graph():
    graph = StateGraph(State)

    graph.add_node("process", process_node)
    graph.add_node("hitl", hitl_node)

    graph.set_entry_point("process")

    graph.add_conditional_edges(
        "process",
        lambda state: "hitl" if state["escalate"] else "__end__"
    )

    graph.add_edge("hitl", "__end__")

    return graph.compile()