from typing import Literal, Optional, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total = False):
    messages : Annotated[list[BaseMessage], add_messages]

    observations : list[dict[str, Any]]
    tool_results : dict[str, Any]
    agent_decisions : list[dict[str, Any]]

    current_agent : str
    next_agent : str

    monitoring_report : dict[str, Any]
    traffic_status : dict[str, Any]

    iteration_count : int
    max_iterations : int

    confidence : float
    requires_human_approval : bool

    final_summary : str
    errors : list[str]    