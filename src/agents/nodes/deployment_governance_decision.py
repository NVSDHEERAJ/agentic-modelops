from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.state import AgentState


load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


DEPLOYMENT_GOVERNANCE_DECISION_PROMPT = """
You are the final decision step for the Deployment Governance Agent.

Review the conversation and tool results from the deployment governance workflow.

Your job:
Decide whether the latest candidate should be rejected, sent for human approval, considered already promoted after explicit approval, rolled back, or escalated.

Important rules:
- Do not claim a candidate was promoted unless a tool result explicitly says promoted=true.
- If candidate passed promotion gates but no explicit approval was granted, decision = human_approval_requested.
- If candidate failed promotion gates, decision = candidate_rejected.
- If promotion failed due to missing evidence, decision = escalation_required.
- If rollback succeeded, decision = rollback_completed.
- Promotion requires explicit human approval outside the agent unless approval was already provided to a promotion tool.

Return a concise JSON object with exactly these fields:
{
  "agent_name": "deployment_governance_agent",
  "decision": "candidate_rejected | human_approval_requested | promoted_after_approval | rollback_completed | escalation_required",
  "confidence": 0.0,
  "reason": "short explanation",
  "candidate_version": null,
  "human_approval_required": true,
  "next_agent": "end"
}

Do not call tools. Only return the JSON object.
"""


def deployment_governance_decision_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    existing_decisions = state.get("agent_decisions", [])

    response = llm.invoke(
        [
            SystemMessage(content=DEPLOYMENT_GOVERNANCE_DECISION_PROMPT),
            HumanMessage(content=f"Conversation and evidence:\n\n{messages}"),
        ]
    )

    decision_text = str(response.content)

    return {
        "agent_decisions": existing_decisions
        + [
            {
                "agent_name": "deployment_governance_agent",
                "decision": "deployment_governance_completed",
                "reason": decision_text,
            }
        ],
        "current_agent": "deployment_governance_agent",
        "next_agent": "end",
        "requires_human_approval": "human_approval_requested" in decision_text,
        "final_summary": decision_text,
    }