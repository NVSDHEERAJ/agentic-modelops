from typing import Literal
from pydantic import BaseModel, Field

AgentName = Literal[
    "monitoring_agent",
    "diagnostic_retraining_agent",
    "deployment_governance_agent",
    "end"
]

MonitoringDecision = Literal[
    "healthy",
    "watch",
    "diagnose",
    "escalate"
]

class AgentDecision(BaseModel):
    current_agent : AgentName = Field(description = "Name of the agent making the decision")
    decision : MonitoringDecision = Field(description = "Final decision made by the agent")
    confidence : float = Field(ge = 0.0, lt = 1.0)
    reason : str = Field(description = "Short explanation for the decision")
    next_agent : AgentName
    needs_more_evidence : bool = False
    human_approval_required : bool = False