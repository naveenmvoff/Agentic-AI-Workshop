import os
import json
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize LLM
gemini = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Session state (in-memory)
session_state = {
    "current": None,
    "history": []
}

# Safe JSON extractor from LLM output
def extract_json_safe(text):
    try:
        text = text.strip().replace("```json", "").replace("```", "")
        start = text.find("{")
        if start > 0:
            text = text[start:]
        return json.loads(text)
    except Exception:
        return {"error": "Could not parse JSON"}

# Tool 1: Generate new HTML layout
def generate_layout(command: str):
    prompt = PromptTemplate.from_template("""
Generate an HTML layout and basic CSS styles from this user instruction:

Instruction: {command}

Return a JSON with:
- layout: string (HTML content)
- props: dictionary of CSS styles
""")
    try:
        response = gemini.invoke(prompt.format(command=command))
        result = extract_json_safe(response.content)
    except Exception:
        result = {
            "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
            "props": {"backgroundColor": "#000000", "color": "#ffffff"}
        }

    session_state["history"].append(session_state["current"])
    session_state["current"] = result
    return result

# Tool 2: Apply CSS to existing layout
def apply_css(command: str):
    prompt = PromptTemplate.from_template("""
Extract a dictionary of CSS style changes from the user command:

Instruction: {command}

Return only the CSS props as JSON, like:
{{
  "color": "orange",
  "fontWeight": "bold"
}}
""")
    try:
        response = gemini.invoke(prompt.format(command=command))
        props = extract_json_safe(response.content)
    except:
        return {"error": "Failed to extract CSS properties"}

    if not isinstance(props, dict) or not props:
        return {"error": "No valid CSS props found"}

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

# Tool 3: Undo last change
def undo_change(_: str):
    if session_state["history"]:
        session_state["current"] = session_state["history"].pop()
        return session_state["current"]
    return {"error": "No previous state to undo"}

# Register tools with clear descriptions for the LLM
tools = [
    Tool(
        name="generate_layout",
        func=generate_layout,
        description="Generate a new HTML layout and initial styles from the user's instruction"
    ),
    Tool(
        name="apply_css",
        func=apply_css,
        description="Apply CSS changes to the current layout. Use when user wants to change colors, fonts, spacing, etc."
    ),
    Tool(
        name="undo_change",
        func=undo_change,
        description="Undo the last layout or CSS change. Use when the user wants to go back, revert, or undo."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=gemini,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Main function to run the agent
def run_agentic_editor(command: str):
    result = agent.run(command)
    return {
        "tool_used": "LLM-decided",
        "current_state": session_state["current"],
        "result": result
    }

# Utilities for session management
def get_session_info():
    return {
        "current_state": session_state["current"],
        "history_length": len(session_state["history"])
    }

def reset_session():
    session_state["current"] = None
    session_state["history"].clear()