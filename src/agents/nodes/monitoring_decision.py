from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.state import AgentState
from src.agents.schemas import AgentDecision

from typing import cast

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
structured_llm = llm.with_structured_output(AgentDecision)

DECISION_PROMPT = """
You are the final decision step for the Monitoring Agent.

Review the conversation, including tool results, and produce a structured decision.

Decision guidance:
- If model is healthy, decision = healthy, next_agent = end.
- If evidence is weak or traffic is low, decision = watch, next_agent = end.
- If drift/performance degradation needs investigation, decision = diagnose, next_agent = diagnostic_retraining_agent.
- If monitoring data is missing/corrupted or tool failures happened, decision = escalate, next_agent = end.

Return all fields required by the AgentDecision schema:
- current_agent: always "monitoring_agent"
- decision: one of healthy, watch, diagnose, escalate
- confidence: float from 0.0 to less than 1.0
- reason: short explanation
- next_agent: diagnostic_retraining_agent or end
- needs_more_evidence: true if monitoring evidence is incomplete
- human_approval_required: true only for escalation or unsafe automated action

Do not call tools. Only return the structured decision.
"""

def monitoring_decision_node(state : AgentState) -> AgentState:
    messages = state.get("messages", [])

    decision = cast(AgentDecision, structured_llm.invoke([
        SystemMessage(content = DECISION_PROMPT),
        HumanMessage(content = f"Conversation and evidence:\n\n{messages}")
    ]))

    existing_decision = state.get("agent_decision", [])

    return {
        "agent_decisions" : existing_decision + [decision.model_dump()],
        "current_agent" : decision.current_agent,
        "next_agent" : decision.next_agent,
        "confidence" : decision.confidence,
        "requires_human_approval" : decision.human_approval_required,
        "final_summary" : decision.reason
    }