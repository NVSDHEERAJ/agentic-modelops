from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.prompts import DEPLOYMENT_GOVERNANCE_AGENT_SYSTEM_PROMPT
from src.agents.state import AgentState
from src.agents.tools.deployment_governance_tools import (
    inspect_active_deployment,
    inspect_latest_candidate,
    promote_latest_candidate_after_approval,
    request_human_approval_for_promotion,
    rollback_to_previous_deployment,
    validate_candidate_for_promotion,
)


load_dotenv()


DEPLOYMENT_GOVERNANCE_TOOLS = [
    inspect_active_deployment,
    inspect_latest_candidate,
    validate_candidate_for_promotion,
    request_human_approval_for_promotion,
    promote_latest_candidate_after_approval,
    rollback_to_previous_deployment,
]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
deployment_governance_llm = llm.bind_tools(DEPLOYMENT_GOVERNANCE_TOOLS)


def deployment_governance_agent(state: AgentState) -> AgentState:
    messages = state.get("messages", [])

    if not messages:
        messages = [
            SystemMessage(content=DEPLOYMENT_GOVERNANCE_AGENT_SYSTEM_PROMPT)
        ]

    response = deployment_governance_llm.invoke(messages)

    return {
        "messages": [response],
        "current_agent": "deployment_governance_agent",
    }
