from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.state import AgentState


load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


DIAGNOSTIC_RETRAINING_DECISION_PROMPT = """
You are the final decision step for the Diagnostic Retraining Agent.

Review the conversation and tool results from the diagnostic retraining workflow.

Your job:
Decide whether the candidate retraining workflow produced a model that should move to deployment governance, be rejected, or require escalation.

Decision guidance:
- If no retraining was needed, decision = no_retrain_needed, next_agent = end.
- If retraining failed, decision = retrain_failed, next_agent = end.
- If a candidate was trained but failed promotion criteria, decision = candidate_rejected, next_agent = end.
- If a candidate was trained and passed promotion criteria, decision = candidate_ready_for_governance, next_agent = deployment_governance_agent.
- If evidence is missing, corrupted, or contradictory, decision = escalate, next_agent = end.

Return a concise JSON object with exactly these fields:
{
  "agent_name": "diagnostic_retraining_agent",
  "decision": "no_retrain_needed | retrain_failed | candidate_rejected | candidate_ready_for_governance | escalate",
  "confidence": 0.0,
  "reason": "short explanation",
  "candidate_version": null,
  "candidate_promotable": false,
  "next_agent": "deployment_governance_agent | end",
  "needs_more_evidence": false,
  "human_approval_required": false
}

Do not call tools. Only return the JSON object.
"""


def diagnostic_retraining_decision_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    existing_decisions = state.get("agent_decisions", [])

    response = llm.invoke(
        [
            SystemMessage(content=DIAGNOSTIC_RETRAINING_DECISION_PROMPT),
            HumanMessage(content=f"Conversation and evidence:\n\n{messages}"),
        ]
    )

    decision_text = str(response.content)

    return {
        "agent_decisions": existing_decisions
        + [
            {
                "agent_name": "diagnostic_retraining_agent",
                "decision": "diagnostic_retraining_completed",
                "reason": decision_text,
            }
        ],
        "current_agent": "diagnostic_retraining_agent",
        "next_agent": "end",
        "final_summary": decision_text,
    }
