import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

# Init Gemini
gemini = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Shared session state
session_state = {
    "current": None,
    "history": []
}

# ----------- RAG Context Helper ------------
# def get_rag_context(command: str) -> str:
#     knowledge_base = {
#         "color": "Use maroon (#800000) for luxury, blue (#007bff) for trust, and green (#28a745) for eco-friendly products.",
#         "water": "Emphasize purity and health with clean UI, white space, and soft blues or greens.",
#         "ecommerce": "Include clear product cards, pricing, add-to-cart buttons, and trust badges.",
#         "layout": "Use responsive sections, hero images, clean navigation, and CTAs.",
#         "typography": "Readable sans-serif fonts. Emphasize headers, clear paragraph spacing.",
#         "bottle": "High-quality images, product specs, and eco-benefits help bottle sales."
#     }
#     context_parts = [v for k, v in knowledge_base.items() if k in command.lower()]
#     if not context_parts:
#         context_parts.append("Use modern clean UI with good contrast and accessible design.")
#     return "\n".join(context_parts)

def get_rag_context(command: str) -> str:
    knowledge_base = {
        "color": "Use maroon (#800000) for luxury, blue (#007bff) for trust, and green (#28a745) for eco-friendly products.",
        "water": "Emphasize purity and health with clean UI, white space, and soft blues or greens.",
        "ecommerce": "Include clear product cards, pricing, add-to-cart buttons, and trust badges.",
        "layout": "Use responsive sections, hero images, clean navigation, and CTAs.",
        "typography": "Readable sans-serif fonts. Emphasize headers, clear paragraph spacing.",
        "bottle": "High-quality images, product specs, and eco-benefits help bottle sales."
    }
    
    context_parts = []
    query_lower = command.lower()
    
    for keyword, content in knowledge_base.items():
        if keyword in query_lower:
            context_parts.append(content)
    
    # Limit to maximum 2 context items
    if not context_parts:
        context_parts = ["Use modern clean UI with good contrast and accessible design."]
    
    return " ".join(context_parts[:2])  # Keep context very short

# ----------- Robust JSON Extractor ----------
def extract_json_safe(text: str) -> Dict[str, Any]:
    try:
        text = text.strip().replace("```json", "").replace("```", "")
        start = text.find("{")
        if start > 0:
            text = text[start:]
        return json.loads(text)
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {"error": "Could not parse JSON"}

# ----------- Tool 1: Generate Layout ---------
def generate_layout(command: str) -> Dict[str, Any]:
    context = get_rag_context(command)
    prompt = PromptTemplate.from_template("""
You are a layout generator. Use this context:

CONTEXT:
{context}

Generate a full HTML layout for: "{command}"

Return this JSON:
{{
  "layout": "<html>...</html>",
  "props": {{"backgroundColor": "#fff", "color": "#000"}}
}}
""")
    try:
        response = gemini.invoke(prompt.format(command=command, context=context))
        result = extract_json_safe(response.content)
    except Exception as e:
        logger.error(f"Layout generation failed: {e}")
        result = {
            "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
            "props": {"backgroundColor": "#000", "color": "#fff"}
        }

    session_state["history"].append(session_state["current"])
    session_state["current"] = result
    return result

# ----------- Tool 2: Apply CSS --------------
def apply_css(command: str) -> Dict[str, Any]:
    context = get_rag_context(command)
    layout = session_state["current"]["layout"] if session_state["current"] else "<html><body></body></html>"

    prompt = PromptTemplate.from_template("""
You are a CSS assistant. Based on the following:

Context:
{context}
Layout snippet:
{layout}

Apply this instruction: "{command}"

Return only:
{{
  "layout": "updated HTML with new CSS",
  "props": {{"color": "orange", "fontWeight": "bold"}}
}}
""")

    try:
        response = gemini.invoke(prompt.format(command=command, context=context, layout=layout[:800]))
        result = extract_json_safe(response.content)
    except Exception as e:
        logger.error(f"CSS update failed: {e}")
        result = {"error": "Failed to apply CSS"}

    if isinstance(result, dict) and "layout" in result:
        session_state["history"].append(session_state["current"])
        session_state["current"] = result
    return result

# ----------- Tool 3: Undo -------------------
def undo_change(_: str) -> Dict[str, Any]:
    if session_state["history"]:
        session_state["current"] = session_state["history"].pop()
        return session_state["current"]
    return {"error": "No previous state to undo"}

# ----------- Tools Registration -------------
tools = [
    Tool(
        name="generate_layout",
        func=generate_layout,
        description="Generate HTML layout and initial styles based on the user's instruction"
    ),
    Tool(
        name="apply_css",
        func=apply_css,
        description="Update CSS based on user styling instructions (color, padding, font, etc.)"
    ),
    Tool(
        name="undo_change",
        func=undo_change,
        description="Undo the last change (layout or CSS)"
    )
]

# ----------- Agent Initialization ----------
agent = initialize_agent(
    tools=tools,
    llm=gemini,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

# ----------- Main Function -----------------
# def run_agentic_editor(command: str) -> Dict[str, Any]:
#     try:
#         logger.info(f"Running agent for command: {command}")
#         agent.run(command)  # LLM decides tool
#         return {
#             "status": "success",
#             "tool_used": "auto",
#             "current_state": session_state["current"],
#             "history_length": len(session_state["history"])
#         }
#     except Exception as e:
#         logger.error(f"Agent failed: {e}")
#         return {
#             "status": "error",
#             "error": str(e),
#             "current_state": session_state["current"]
#         }

def run_agentic_editor(command: str) -> Dict[str, Any]:
        logger.info(f"Running agent for command: {command}")
        
        # Avoid calling Gemini for 'undo'
        if command.strip().lower() == "undo":
            result = undo_change(command)
        else:
            agent.run(command)
            result = session_state["current"]
        
        return {
            "status": "success",
            "tool_used": "auto",
            "current_state": result,
            "history_length": len(session_state["history"])
        }


# ----------- Utility: Session Info ----------
def get_session_info():
    return {
        "current_state": session_state["current"],
        "history_length": len(session_state["history"])
    }

def reset_session():
    session_state["current"] = None
    session_state["history"].clear()
    logger.info("Session reset")
