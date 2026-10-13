from ai_assistant.adapters.http.agent_routes import AgentExecuteResponse


def test_contract():
    assert AgentExecuteResponse.schema()["title"] == "AgentExecuteResponse"
