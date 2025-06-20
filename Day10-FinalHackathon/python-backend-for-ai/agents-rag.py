import os
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize LLM
try:
    gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    raise

# Simple RAG context (without FAISS)
def get_rag_context(query: str) -> str:
    """Get relevant context without vectorstore - using simple keyword matching"""
    
    knowledge_base = {
        "color": "Brand guidelines: Use professional colors. Maroon (#800000) represents elegance and sophistication. Blue (#007bff) for trust. Green (#28a745) for nature/eco-friendly.",
        "water": "Water bottle websites should emphasize purity, health, and freshness. Use clean, minimal designs with plenty of white space.",
        "ecommerce": "E-commerce sites need clear product displays, pricing, shopping cart, and contact information. Include testimonials and trust badges.",
        "layout": "Modern layouts use responsive design, clear navigation, hero sections, and call-to-action buttons.",
        "typography": "Use clean, readable fonts. Headers should be bold and attention-grabbing.",
        "bottle": "Product showcase should include high-quality images, specifications, benefits, and customer reviews."
    }
    
    context_parts = []
    query_lower = query.lower()
    
    for keyword, info in knowledge_base.items():
        if keyword in query_lower:
            context_parts.append(info)
    
    # Add default context if no matches
    if not context_parts:
        context_parts.append("Use modern, clean design principles with good color contrast and readability.")
    
    return "\n".join(context_parts)

# Shared session state
session_state = {
    "current": {
        "layout": "<html><body><h1>Welcome</h1><p>Start editing your website!</p></body></html>",
        "props": {"backgroundColor": "#ffffff", "color": "#333333"}
    },
    "history": []
}

# Tool 1: NL Command Parser Agent
# def parse_nl_command(command: str) -> Dict[str, Any]:
#     """Parse natural language command into structured format"""
#     context = get_rag_context(command)
    
#     prompt = PromptTemplate.from_template("""
# You are an intelligent website editing command parser. Use this context for guidance:

# CONTEXT:
# {context}

# Parse the following natural language command into a structured JSON format.

# Supported actions:
# - create_layout: Generate new HTML layout
# - update_css: Modify styling/appearance
# - update_content: Change text content
# - undo: Revert last change

# Common targets: body, header, footer, nav, main, section, div, h1, h2, p, button
# Common CSS properties: color, backgroundColor, fontSize, fontWeight, padding, margin, textAlign

# Examples:
# "Create a homepage with navigation" → {{"action": "create_layout", "target": "body", "props": {{"type": "homepage"}}}}
# "Make the header bold and blue" → {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "blue"}}}}
# "Change the title to Welcome" → {{"action": "update_content", "target": "h1", "props": {{"text": "Welcome"}}}}

# Command: {command}

# Return only valid JSON:
# """)
    
#     try:
#         response = gemini.invoke(prompt.format(command=command, context=context))
#         result = json.loads(response.content.strip())
#         logger.info(f"Parsed command: {result}")
#         return result
#     except Exception as e:
#         logger.error(f"Parsing failed: {e}")
#         return {"action": "create_layout", "target": "body", "props": {}}

def parse_nl_command(command: str) -> Dict[str, Any]:
    context = get_rag_context(command)
    prompt = PromptTemplate.from_template("""
You are an intelligent website editing command parser. Use this context for guidance:

CONTEXT:
{context}

Parse the following natural language command into a structured JSON format.

Supported actions:
- create_layout: Generate new HTML layout
- update_css: Modify styling/appearance
- update_content: Change text content
- undo: Revert last change

Common targets: body, header, footer, nav, main, section, div, h1, h2, p, button
Common CSS properties: color, backgroundColor, fontSize, fontWeight, padding, margin, textAlign

Examples:
"Create a homepage with navigation" → {{"action": "create_layout", "target": "body", "props": {{"type": "homepage"}}}}
"Make the header bold and blue" → {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "blue"}}}}
"Change the title to Welcome" → {{"action": "update_content", "target": "h1", "props": {{"text": "Welcome"}}}}

Command: {command}

Return only valid JSON:
""")
    
    try:
        response = gemini.invoke(prompt.format(command=command, context=context))
        logger.debug(f"LLM raw response: {response}")
        result = json.loads(response.strip())
        logger.info(f"Parsed command: {result}")
        return result
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return {"action": "create_layout", "target": "body", "props": {}}

# Tool 2: Layout Generator Agent
def generate_layout_agent(command: str) -> Dict[str, Any]:
    """Generate HTML layout based on command"""
    context = get_rag_context(command)
    
    prompt = PromptTemplate.from_template("""
You are a web layout generator. Use this context for styling guidance:

CONTEXT:
{context}

Generate a complete HTML layout for: "{command}"

Requirements:
- Create semantic HTML structure
- Include inline CSS styles
- Make it responsive and accessible
- Focus on the specific request (water bottle selling website)
- Use the specified colors (maroon: #800000)

Return JSON format:
{{
  "layout": "<html><head><title>Water Bottle Store</title><style>/* CSS here */</style></head><body>/* HTML content here */</body></html>",
  "props": {{"backgroundColor": "#ffffff", "color": "#333333", "theme": "water-bottle-store"}}
}}

Make it a complete, professional website layout.
""")
    
    try:
        response = gemini.invoke(prompt.format(command=command, context=context))
        result = json.loads(response.content.strip())
        
        # Store previous state
        session_state["history"].append(session_state["current"].copy())
        session_state["current"] = result
        
        logger.info("Generated new layout")
        return result
    except Exception as e:
        logger.error(f"Layout generation failed: {e}")
        fallback_layout = {
            "layout": f"""
<html>
<head>
    <title>Water Bottle Store</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
        .header {{ background-color: #800000; color: white; padding: 20px; text-align: center; }}
        .nav {{ background-color: #600000; padding: 10px; }}
        .nav a {{ color: white; text-decoration: none; margin: 0 15px; }}
        .hero {{ padding: 50px 20px; text-align: center; background: linear-gradient(135deg, #800000, #a00000); color: white; }}
        .products {{ padding: 40px 20px; max-width: 1200px; margin: 0 auto; }}
        .footer {{ background-color: #800000; color: white; padding: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Premium Water Bottles</h1>
        <p>Pure. Fresh. Premium Quality.</p>
    </div>
    <nav class="nav">
        <a href="#home">Home</a>
        <a href="#products">Products</a>
        <a href="#about">About</a>
        <a href="#contact">Contact</a>
    </nav>
    <div class="hero">
        <h2>Discover Our Premium Water Collection</h2>
        <p>Sustainable. Pure. Refreshing.</p>
        <button style="background-color: white; color: #800000; padding: 12px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">Shop Now</button>
    </div>
    <div class="products">
        <h2 style="text-align: center; color: #800000;">Our Products</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px;">
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center;">
                <h3>Premium Glass Bottle</h3>
                <p>Eco-friendly glass bottles for pure taste</p>
                <p style="color: #800000; font-size: 20px; font-weight: bold;">$25.99</p>
            </div>
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center;">
                <h3>Stainless Steel Bottle</h3>
                <p>Insulated bottles that keep water cold for 24 hours</p>
                <p style="color: #800000; font-size: 20px; font-weight: bold;">$35.99</p>
            </div>
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center;">
                <h3>Sports Water Bottle</h3>
                <p>Perfect for workouts and outdoor activities</p>
                <p style="color: #800000; font-size: 20px; font-weight: bold;">$19.99</p>
            </div>
        </div>
    </div>
    <footer class="footer">
        <p>&copy; 2024 Premium Water Bottles. All rights reserved.</p>
        <p>Contact: info@waterbottles.com | (555) 123-4567</p>
    </footer>
</body>
</html>
            """,
            "props": {"backgroundColor": "#f8f9fa", "color": "#333333", "theme": "maroon", "primaryColor": "#800000"}
        }
        
        session_state["history"].append(session_state["current"].copy())
        session_state["current"] = fallback_layout
        return fallback_layout

# Tool 3: CSS Update Agent
def update_css_agent(command: str) -> Dict[str, Any]:
    """Update CSS styling based on command"""
    context = get_rag_context(command)
    
    prompt = PromptTemplate.from_template("""
You are a CSS styling agent. Update the website styling based on: "{command}"

Context: {context}
Current layout: {layout}

Modify the CSS in the layout to match the request. Return the updated layout with new styling.

Return JSON format:
{{
  "layout": "updated_html_with_new_css",
  "props": {{"updated": "properties"}}
}}
""")
    
    try:
        current_layout = session_state["current"]["layout"]
        
        response = gemini.invoke(prompt.format(
            command=command, 
            context=context,
            layout=current_layout[:1000] + "..." if len(current_layout) > 1000 else current_layout
        ))
        
        result = json.loads(response.content.strip())
        
        # Store previous state
        session_state["history"].append(session_state["current"].copy())
        session_state["current"] = result
        
        logger.info("Updated CSS")
        return result
    except Exception as e:
        logger.error(f"CSS update failed: {e}")
        return session_state["current"]

# Tool 4: Undo Agent
def undo_agent(command: str) -> Dict[str, Any]:
    """Undo last change"""
    if session_state["history"]:
        session_state["current"] = session_state["history"].pop()
        logger.info("Undid last change")
        return session_state["current"]
    else:
        logger.warning("No history to undo")
        return {"error": "No previous state to undo"}

# Tool 5: Get Current State
def get_current_state(command: str) -> Dict[str, Any]:
    """Get current state of the website"""
    return {
        "current_state": session_state["current"],
        "history_length": len(session_state["history"])
    }

# Register tools for the agent
tools = [
    Tool(
        name="parse_command",
        func=parse_nl_command,
        description="Parse natural language editing commands into structured format"
    ),
    Tool(
        name="generate_layout",
        func=generate_layout_agent,
        description="Generate new HTML layout based on user instructions"
    ),
    Tool(
        name="update_css",
        func=update_css_agent,
        description="Update CSS styling and appearance of website elements"
    ),
    Tool(
        name="undo_change",
        func=undo_agent,
        description="Undo the last change made to the website"
    ),
    Tool(
        name="get_state",
        func=get_current_state,
        description="Get current state of the website"
    )
]

# Initialize the main agent
try:
    agent = initialize_agent(
        tools=tools,
        llm=gemini,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Agent initialization failed: {e}")
    raise

# Main entry point
def run_agentic_editor(command: str) -> Dict[str, Any]:
    """Main function to process user commands through the agentic system"""
    try:
        logger.info(f"Processing command: {command}")
        
        # Parse the command first
        parsed = parse_nl_command(command)
        action = parsed.get("action", "create_layout")
        
        # Route to appropriate agent based on action
        if action == "create_layout":
            result = generate_layout_agent(command)
        elif action == "update_css":
            result = update_css_agent(command)
        elif action == "undo":
            result = undo_agent(command)
        else:
            # Default to layout generation for unknown actions
            result = generate_layout_agent(command)
        
        return {
            "status": "success",
            "parsed_command": parsed,
            "action_taken": action,
            "result": result,
            "current_state": session_state["current"],
            "history_length": len(session_state["history"])
        }
        
    except Exception as e:
        logger.error(f"Command processing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "current_state": session_state["current"]
        }

# Utility functions
def reset_session():
    """Reset the session state"""
    global session_state
    session_state = {
        "current": {
            "layout": "<html><body><h1>Welcome</h1><p>Start editing your website!</p></body></html>",
            "props": {"backgroundColor": "#ffffff", "color": "#333333"}
        },
        "history": []
    }
    logger.info("Session reset")

def get_session_info():
    """Get current session information"""
    return {
        "current_state": session_state["current"],
        "history_length": len(session_state["history"]),
        "rag_enabled": True
    }
