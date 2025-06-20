import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from itertools import cycle
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load and rotate API keys
GOOGLE_API_KEYS = [
    os.getenv("GOOGLE_API_KEY1"),
    os.getenv("GOOGLE_API_KEY2"),
    os.getenv("GOOGLE_API_KEY3")
]
GOOGLE_API_KEYS = [key for key in GOOGLE_API_KEYS if key]
if not GOOGLE_API_KEYS:
    raise ValueError("No valid GOOGLE_API_KEYs found")
key_cycle = cycle(GOOGLE_API_KEYS)

# Gemini client rotation
def get_gemini_client():
    current_key = next(key_cycle)
    logger.info(f"Using API key: {current_key[:6]}****")
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=current_key,
        temperature=0.3
    )

# Gemini invoke wrapper with retry + key rotation
def try_invoke_with_rotation(prompt, retries=3):
    for _ in range(retries):
        try:
            llm = get_gemini_client()
            return llm.invoke(prompt)
        except Exception as e:
            logger.warning(f"Key failed, rotating: {e}")
    return {"content": '{"error": "All keys failed or quota exhausted"}'}

# Session state (in-memory)
session_state = {
    "current": None,
    "history": []
}

# Contextual RAG helper
def get_rag_context(command: str) -> str:
    knowledge_base = {
        "color": "Use maroon (#800000) for luxury, blue (#007bff) for trust, and green (#28a745) for eco-friendly products.",
        "water": "Emphasize purity and health with clean UI, white space, and soft blues or greens.",
        "ecommerce": "Include clear product cards, pricing, add-to-cart buttons, and trust badges.",
        "layout": "Use responsive sections, hero images, clean navigation, and CTAs.",
        "typography": "Readable sans-serif fonts. Emphasize headers, clear paragraph spacing.",
        "bottle": "High-quality images, product specs, and eco-benefits help bottle sales."
    }
    context_parts = [v for k, v in knowledge_base.items() if k in command.lower()]
    if not context_parts:
        context_parts = ["Use modern clean UI with good contrast and accessible design."]
    return " ".join(context_parts[:2])

# Safer JSON extractor with logging
def extract_json_safe(text: str) -> Dict[str, Any]:
    try:
        text = text.strip().replace("```json", "").replace("```", "")
        start = text.find("{")
        if start > 0:
            text = text[start:]
        logger.info(f"Raw Gemini output:\n{text}")
        return json.loads(text)
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {"error": "Could not parse JSON"}

# Tool: Generate Layout
def generate_layout(command: str) -> Dict[str, Any]:
    context = get_rag_context(command)
    prompt = PromptTemplate.from_template("""
You are a layout generator. Use this context:

CONTEXT:
{context}

Generate a full HTML layout for: "{command}"

Return ONLY this JSON:
{{
  "layout": "<html>...</html>",
  "props": {{"backgroundColor": "#fff", "color": "#000"}}
}}
""")
    response = try_invoke_with_rotation(prompt.format(command=command, context=context))
    result = extract_json_safe(response.content)

    session_state["history"].append(session_state["current"])
    session_state["current"] = result
    return result

# Tool: Apply CSS
def apply_css(command: str) -> Dict[str, Any]:
    context = get_rag_context(command)
    layout = session_state["current"]["layout"] if session_state["current"] else "<html><body></body></html>"

    prompt = PromptTemplate.from_template("""
You are a strict JSON responder. Use the given context and layout to apply styling instructions.

CONTEXT:
{context}

LAYOUT:
{layout}

INSTRUCTION:
"{command}"

Now return ONLY valid JSON with this exact format:
{{
  "layout": "<updated HTML as a string>",
  "props": {{
    "color": "orange",
    "fontWeight": "bold"
  }}
}}

IMPORTANT:
- "layout" must be a single-line escaped HTML string
- Do NOT return code blocks or markdown
- Do NOT return any extra text
""")

    response = try_invoke_with_rotation(prompt.format(command=command, context=context, layout=layout[:800]))
    result = extract_json_safe(response.content)

    if isinstance(result, dict) and "layout" in result:
        session_state["history"].append(session_state["current"])
        session_state["current"] = result
    return result

# Tool: Undo
def undo_change(_: str) -> Dict[str, Any]:
    if session_state["history"]:
        session_state["current"] = session_state["history"].pop()
        return session_state["current"]
    return {"error": "No previous state to undo"}

# Register tools
tools = [
    Tool(name="generate_layout", func=generate_layout, description="Generate HTML layout"),
    Tool(name="apply_css", func=apply_css, description="Update CSS styles"),
    Tool(name="undo_change", func=undo_change, description="Undo the last change")
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=get_gemini_client(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

# Main command handler
def run_agentic_editor(command: str) -> Dict[str, Any]:
    logger.info(f"Running agent for command: {command}")
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

# Session utils
def get_session_info():
    return {
        "current_state": session_state["current"],
        "history_length": len(session_state["history"])
    }

def reset_session():
    session_state["current"] = None
    session_state["history"].clear()
    logger.info("Session reset")



# import os
# import json
# import logging
# from typing import Dict, Any
# from dotenv import load_dotenv
# from itertools import cycle
# from langchain.agents import Tool, initialize_agent, AgentType
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Load and rotate API keys
# GOOGLE_API_KEYS = [
#     os.getenv("GOOGLE_API_KEY1"),
#     os.getenv("GOOGLE_API_KEY2"),
#     os.getenv("GOOGLE_API_KEY3")
# ]
# GOOGLE_API_KEYS = [key for key in GOOGLE_API_KEYS if key]
# if not GOOGLE_API_KEYS:
#     raise ValueError("No valid GOOGLE_API_KEYs found")
# key_cycle = cycle(GOOGLE_API_KEYS)

# # Get Gemini Client from current key
# def get_gemini_client():
#     current_key = next(key_cycle)
#     logger.info(f"Using API key: {current_key[:6]}****")
#     return ChatGoogleGenerativeAI(
#         model="models/gemini-1.5-flash",
#         google_api_key=current_key,
#         temperature=0.3
#     )

# def try_invoke_with_rotation(prompt, retries=3):
#     for _ in range(retries):
#         try:
#             llm = get_gemini_client()
#             return llm.invoke(prompt)
#         except Exception as e:
#             logger.warning(f"Key failed, rotating: {e}")
#     return {"content": '{"error": "All keys failed or quota exhausted"}'}

# # Shared session state
# session_state = {
#     "current": None,
#     "history": []
# }

# # RAG Context Helper
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
#         context_parts = ["Use modern clean UI with good contrast and accessible design."]
#     return " ".join(context_parts[:2])

# # JSON Extractor
# def extract_json_safe(text: str) -> Dict[str, Any]:
#     try:
#         text = text.strip().replace("```json", "").replace("```", "")
#         start = text.find("{")
#         if start > 0:
#             text = text[start:]
#         return json.loads(text)
#     except Exception as e:
#         logger.error(f"Failed to parse JSON: {e}")
#         return {"error": "Could not parse JSON"}

# # Tool: Generate Layout
# def generate_layout(command: str) -> Dict[str, Any]:
#     context = get_rag_context(command)
#     prompt = PromptTemplate.from_template("""
# You are a layout generator. Use this context:

# CONTEXT:
# {context}

# Generate a full HTML layout for: "{command}"

# Return this JSON:
# {{
#   "layout": "<html>...</html>",
#   "props": {{"backgroundColor": "#fff", "color": "#000"}}
# }}
# """)
#     response = try_invoke_with_rotation(prompt.format(command=command, context=context))
#     result = extract_json_safe(response.content)

#     session_state["history"].append(session_state["current"])
#     session_state["current"] = result
#     return result

# # Tool: Apply CSS
# def apply_css(command: str) -> Dict[str, Any]:
#     context = get_rag_context(command)
#     layout = session_state["current"]["layout"] if session_state["current"] else "<html><body></body></html>"

#     prompt = PromptTemplate.from_template("""
# You are a CSS assistant. Based on the following:

# Context:
# {context}
# Layout snippet:
# {layout}

# Apply this instruction: "{command}"

# Return only:
# {{
#   "layout": "updated HTML with new CSS",
#   "props": {{"color": "orange", "fontWeight": "bold"}}
# }}
# """)
#     response = try_invoke_with_rotation(prompt.format(command=command, context=context, layout=layout[:800]))
#     result = extract_json_safe(response.content)

#     if isinstance(result, dict) and "layout" in result:
#         session_state["history"].append(session_state["current"])
#         session_state["current"] = result
#     return result

# # Tool: Undo
# def undo_change(_: str) -> Dict[str, Any]:
#     if session_state["history"]:
#         session_state["current"] = session_state["history"].pop()
#         return session_state["current"]
#     return {"error": "No previous state to undo"}

# # Tools Registration
# tools = [
#     Tool(name="generate_layout", func=generate_layout, description="Generate HTML layout"),
#     Tool(name="apply_css", func=apply_css, description="Update CSS styles"),
#     Tool(name="undo_change", func=undo_change, description="Undo the last change")
# ]

# # Agent Initialization
# agent = initialize_agent(
#     tools=tools,
#     llm=get_gemini_client(),
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=3
# )

# # Main Agent Handler
# def run_agentic_editor(command: str) -> Dict[str, Any]:
#     logger.info(f"Running agent for command: {command}")
#     if command.strip().lower() == "undo":
#         result = undo_change(command)
#     else:
#         agent.run(command)
#         result = session_state["current"]
#     return {
#         "status": "success",
#         "tool_used": "auto",
#         "current_state": result,
#         "history_length": len(session_state["history"])
#     }

# # Utility Functions
# def get_session_info():
#     return {
#         "current_state": session_state["current"],
#         "history_length": len(session_state["history"])
#     }

# def reset_session():
#     session_state["current"] = None
#     session_state["history"].clear()
#     logger.info("Session reset")