from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from src.agents.state import AgentState
from src.agents.prompts import MONITORING_AGENT_SYSTEM_PROMPT
from src.agents.tools.monitoring_tools import (
    generate_monitoring_report,
    generate_traffic_stats,
    check_last_retrain_date
)

load_dotenv()

MONITORING_TOOLS = [
    generate_monitoring_report,
    generate_traffic_stats,
    check_last_retrain_date
]

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

monitoring_llm = llm.bind_tools(MONITORING_TOOLS)

def monitoring_agent(state: AgentState) -> AgentState:
    messages = state.get("messages", [])

    if not messages:
        messages = [
            SystemMessage(content = MONITORING_AGENT_SYSTEM_PROMPT)
        ]
    
    response = monitoring_llm.invoke(messages)

    return {
        "messages": [response],
        "current_agent" : "monitoring_agent"
    }