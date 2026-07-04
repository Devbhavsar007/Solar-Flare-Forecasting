"""
LangGraph pipeline construction for JWALA solar flare orchestration.

Decision D7: Two detector edges fan into one merge node using the
Annotated reducer pattern on SolarPipelineState.detected_events.
Simple add_edge() from both detectors to merge — the reducer handles
concatenation automatically.
"""
from langgraph.graph import StateGraph, END

from src.orchestration.state import SolarPipelineState
from src.orchestration.agents import (
    ingestion_agent,
    solexs_detect_agent,
    hel1os_detect_agent,
    merge_agent,
    preprocess_agent,
    moment_score_agent,
    nowcast_agent,
    forecast_agent,
    uncertainty_agent,
    shap_agent,
    llm_report_agent,
    alert_router,
)


def build_pipeline():
    """
    Construct and compile the LangGraph solar flare pipeline.

    Graph topology:
        ingestion ──┬──> detect_solexs ──┐
                    └──> detect_hel1os ──┤
                                        ▼
                                      merge
                                        │
                                    preprocess
                                        │
                                   moment_score
                                        │
                                     nowcast
                                        │
                                     forecast
                                        │
                                   uncertainty
                                        │
                                      shap
                                        │
                                  alert_router
                                   ╱          ╲
                          llm_report          END
                              │
                             END

    The fan-in from detect_solexs and detect_hel1os into merge is handled
    by the Annotated reducer on SolarPipelineState.detected_events.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke().
    """
    graph = StateGraph(SolarPipelineState)

    # --- Add nodes ---
    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("detect_solexs", solexs_detect_agent)
    graph.add_node("detect_hel1os", hel1os_detect_agent)
    graph.add_node("merge", merge_agent)
    graph.add_node("preprocess", preprocess_agent)
    graph.add_node("moment_score", moment_score_agent)
    graph.add_node("nowcast", nowcast_agent)
    graph.add_node("forecast", forecast_agent)
    graph.add_node("uncertainty", uncertainty_agent)
    graph.add_node("shap", shap_agent)
    graph.add_node("llm_report", llm_report_agent)

    # --- Set entry point ---
    graph.set_entry_point("ingestion")

    # --- Parallel fan-out from ingestion to both detectors ---
    graph.add_edge("ingestion", "detect_solexs")
    graph.add_edge("ingestion", "detect_hel1os")

    # --- Fan-in: both detectors merge into merge agent ---
    # The Annotated reducer on detected_events concatenates both lists
    graph.add_edge("detect_solexs", "merge")
    graph.add_edge("detect_hel1os", "merge")

    # --- Linear pipeline after merge ---
    graph.add_edge("merge", "preprocess")
    graph.add_edge("preprocess", "moment_score")
    graph.add_edge("moment_score", "nowcast")
    graph.add_edge("nowcast", "forecast")
    graph.add_edge("forecast", "uncertainty")
    graph.add_edge("uncertainty", "shap")

    # --- Conditional routing via alert_router ---
    graph.add_conditional_edges(
        "shap",
        alert_router,
        {
            "llm_report": "llm_report",
            "end": END,
        },
    )

    # --- LLM report terminates the pipeline ---
    graph.add_edge("llm_report", END)

    # --- Compile and return ---
    return graph.compile()
