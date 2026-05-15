from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.prompts import DIAGNOSTIC_RETRAINING_AGENT_SYSTEM_PROMPT
from src.agents.state import AgentState
from src.agents.tools.diagnostic_retraining_tools import (
    evaluate_candidate_model,
    get_feature_drift_breakdown,
    retrain_candidate_model,
    run_data_quality_checks,
)


load_dotenv()


DIAGNOSTIC_RETRAINING_TOOLS = [
    get_feature_drift_breakdown,
    run_data_quality_checks,
    retrain_candidate_model,
    evaluate_candidate_model,
]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
diagnostic_retraining_llm = llm.bind_tools(DIAGNOSTIC_RETRAINING_TOOLS)


def diagnostic_retraining_agent(state: AgentState) -> AgentState:
    messages = state.get("messages", [])

    if not messages:
        messages = [
            SystemMessage(content=DIAGNOSTIC_RETRAINING_AGENT_SYSTEM_PROMPT)
        ]

    response = diagnostic_retraining_llm.invoke(messages)

    return {
        "messages": [response],
        "current_agent": "diagnostic_retraining_agent",
    }
