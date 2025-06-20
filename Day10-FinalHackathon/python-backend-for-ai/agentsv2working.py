import os
import json
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv()
gemini = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Shared in-memory session state
session_state = {
    "current": None,
    "history": []
}

# JSON safety parser
def extract_json_safe(text):
    try:
        text = text.strip().replace("```json", "").replace("```", "")
        start = text.find("{")
        if start > 0:
            text = text[start:]
        return json.loads(text)
    except:
        return {"action": "unknown", "target": "", "props": {}}

# Parse natural language commands into structured format
def parse_nl_command(command: str):
    prompt = PromptTemplate.from_template("""
Convert user instruction into JSON:
- action: [create_layout, update_css, undo]
- target: HTML selector (body, header, footer, etc.)
- props: CSS style object

Respond only with valid JSON.

Instruction: {command}
""")
    try:
        response = gemini.invoke(prompt.format(command=command))
        return extract_json_safe(response.content)
    except:
        return {"action": "unknown", "target": "", "props": {}}

# Tool 1: Generate Layout
def generate_layout(command: str):
    prompt = PromptTemplate.from_template("""
Generate a basic HTML layout and CSS props based on user instruction.

Instruction: {command}

Return JSON:
{{
  "layout": "<html>...</html>",
  "props": {{ "backgroundColor": "#xxxxxx", ... }}
}}
""")
    try:
        response = gemini.invoke(prompt.format(command=command))
        result = extract_json_safe(response.content)
    except:
        result = {
            "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
            "props": {"backgroundColor": "#000000", "color": "#ffffff"}
        }

    session_state["history"].append(session_state["current"])
    session_state["current"] = result
    return result

# Tool 2: Apply CSS
def apply_css(command: str):
    parsed = parse_nl_command(command)
    props = parsed.get("props", {})
    if not props:
        return {"error": "No CSS props found"}

    if session_state["current"] is None:
        session_state["current"] = {
            "layout": "<html><body><h1>Empty Page</h1></body></html>",
            "props": {}
        }

    updated = session_state["current"].copy()
    updated["props"].update(props)
    session_state["history"].append(session_state["current"])
    session_state["current"] = updated
    return updated

# Tool 3: Undo
def undo_change(_: str):
    if session_state["history"]:
        session_state["current"] = session_state["history"].pop()
        return session_state["current"]
    return {"error": "No previous state"}

# Fallback parser
def parse_fallback(command: str):
    return parse_nl_command(command)

# Register tools
tools = [
    Tool(name="generate_layout", func=generate_layout, description="Generate a new HTML layout"),
    Tool(name="apply_css", func=apply_css, description="Apply CSS to layout"),
    Tool(name="undo_change", func=undo_change, description="Undo last change"),
    Tool(name="parse_fallback", func=parse_fallback, description="Fallback command parser")
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=gemini,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Main agent execution entry
def run_agentic_editor(command: str):
    parsed = parse_nl_command(command)
    result = agent.run(command)
    return {
        "parsed": parsed,
        "tool_used": "LLM-decided",
        "current_state": session_state["current"],
        "result": result
    }

# Session utilities
def get_session_info():
    return {
        "current_state": session_state["current"],
        "history_length": len(session_state["history"])
    }

def reset_session():
    session_state["current"] = None
    session_state["history"].clear()
