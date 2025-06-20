import os, json
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv()
gemini = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Shared in-memory session state
session_state = {
    "current": None,
    "history": []
}

# 1️⃣ Natural Language Parser Tool
def parse_nl_command(command: str):
    prompt = PromptTemplate.from_template("""
You are an intelligent parser. Convert natural language editing commands into structured JSON format.

Supported keys:
- action: one of [create_layout, update_css, undo]
- target: valid HTML selector like body/header/footer/div
- props: dictionary of CSS styles (color, backgroundColor, fontWeight, etc.)

Examples:
Input: "Create a new homepage layout"
Output: {{"action": "create_layout", "target": "body", "props": {}}}

Input: "Make the header bold and red"
Output: {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "red"}}}}

Input: "Undo that change"
Output: {{"action": "undo", "target": "", "props": {}}}

Now parse:
Instruction: {command}

Respond with only valid JSON.
    """)
    try:
        response = gemini.invoke(prompt.format(command=command))
        return json.loads(response.content.strip())
    except:
        return {"action": "unknown", "target": "", "props": {}}

# 2️⃣ Tool - Layout Generator (LLM Decides HTML + CSS)
def generate_layout(command: str):
    layout_prompt = PromptTemplate.from_template("""
Generate a basic website HTML layout with optional inline styling based on the user's instruction.

Instruction: {command}

Return only valid JSON:
{{
  "layout": "<html>...</html>",
  "props": {{ CSS key-value pairs like color, backgroundColor, etc }}
}}
    """)
    try:
        response = gemini.invoke(layout_prompt.format(command=command))
        result = json.loads(response.content.strip())
    except:
        result = {
            "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
            "props": {"backgroundColor": "#000000", "color": "#ffffff"}
        }

    session_state["history"].append(session_state["current"])
    session_state["current"] = result
    return result

# 3️⃣ Tool - CSS Editor
def apply_css(command: str):
    parsed = parse_nl_command(command)
    props = parsed.get("props", {})
    selector = parsed.get("target", "body")

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

# 4️⃣ Tool - Undo Agent
def undo_change(_: str):
    if session_state["history"]:
        session_state["current"] = session_state["history"].pop()
        return session_state["current"]
    return {"error": "No previous state"}

# 5️⃣ Tool - Fallback Parser
def parse_fallback(command: str):
    return parse_nl_command(command)

# 🔧 Register Tools
tools = [
    Tool(name="generate_layout", func=generate_layout, description="Generate layout using user instruction"),
    Tool(name="apply_css", func=apply_css, description="Apply CSS changes to layout"),
    Tool(name="undo_change", func=undo_change, description="Undo the last change"),
    Tool(name="parse_fallback", func=parse_fallback, description="Fallback parser to structure unclear commands"),
]

# 🤖 Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=gemini,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 🎯 Main Entry: Agentic Editor Controller
def run_agentic_editor(command: str):
    parsed = parse_nl_command(command)
    result = agent.run(command)
    return {
        "parsed": parsed,
        "tool_used": "LLM-decided",
        "current_state": session_state["current"],
        "result": result
    }






# import os, json
# from dotenv import load_dotenv
# from langchain.agents import Tool, initialize_agent, AgentType
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Load environment
# load_dotenv()
# gemini = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# # Shared in-memory session state
# session_state = {
#     "current": None,
#     "history": []
# }

# # 1️⃣ Natural Language Parser Tool
# def parse_nl_command(command: str):
#     prompt = PromptTemplate.from_template("""
# You are an intelligent parser. Convert natural language editing commands into structured JSON format.

# Supported keys:
# - action: one of [create_layout, update_css, undo]
# - target: valid HTML selector like body/header/footer/div
# - props: dictionary of CSS styles (color, backgroundColor, fontWeight, etc.)

# Examples:
# Input: "Create a new homepage layout"
# Output: {{"action": "create_layout", "target": "body", "props": {}}}

# Input: "Make the header bold and red"
# Output: {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "red"}}}}

# Input: "Undo that change"
# Output: {{"action": "undo", "target": "", "props": {}}}

# Now parse:
# Instruction: {command}

# Respond with only valid JSON.
#     """)
#     try:
#         response = gemini.invoke(prompt.format(command=command))
#         return json.loads(response.content.strip())
#     except:
#         return {"action": "unknown", "target": "", "props": {}}

# # 2️⃣ Tool - Layout Generator (LLM Decides HTML + CSS)
# def generate_layout(command: str):
#     layout_prompt = PromptTemplate.from_template("""
# Generate a basic website HTML layout with optional inline styling based on the user's instruction.

# Instruction: {command}

# Return only valid JSON:
# {{
#   "layout": "<html>...</html>",
#   "props": {{ CSS key-value pairs like color, backgroundColor, etc }}
# }}
#     """)
#     try:
#         response = gemini.invoke(layout_prompt.format(command=command))
#         result = json.loads(response.content.strip())
#     except:
#         result = {
#             "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
#             "props": {"backgroundColor": "#000000", "color": "#ffffff"}
#         }

#     session_state["history"].append(session_state["current"])
#     session_state["current"] = result
#     return result

# # 3️⃣ Tool - CSS Editor
# def apply_css(command: str):
#     parsed = parse_nl_command(command)
#     props = parsed.get("props", {})
#     selector = parsed.get("target", "body")

#     if session_state["current"] is None:
#         session_state["current"] = {
#             "layout": "<html><body><h1>Empty Page</h1></body></html>",
#             "props": {}
#         }

#     updated = session_state["current"].copy()
#     updated["props"].update(props)
#     session_state["history"].append(session_state["current"])
#     session_state["current"] = updated
#     return updated

# # 4️⃣ Tool - Undo Agent
# def undo_change(_: str):
#     if session_state["history"]:
#         session_state["current"] = session_state["history"].pop()
#         return session_state["current"]
#     return {"error": "No previous state"}

# # 5️⃣ Tool - Fallback Parser
# def parse_fallback(command: str):
#     return parse_nl_command(command)

# # 🔧 Register Tools
# tools = [
#     Tool(name="generate_layout", func=generate_layout, description="Generate layout using user instruction"),
#     Tool(name="apply_css", func=apply_css, description="Apply CSS changes to layout"),
#     Tool(name="undo_change", func=undo_change, description="Undo the last change"),
#     Tool(name="parse_fallback", func=parse_fallback, description="Fallback parser to structure unclear commands"),
# ]

# # 🤖 Initialize Agent
# agent = initialize_agent(
#     tools=tools,
#     llm=gemini,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True
# )

# # 🎯 Main Entry: Agentic Editor Controller
# def run_agentic_editor(command: str):
#     parsed = parse_nl_command(command)
#     result = agent.run(command)
#     return {
#         "parsed": parsed,
#         "tool_used": "LLM-decided",
#         "current_state": session_state["current"],
#         "result": result
#     }



# # # ###### D:\Learnig\AI\IHUB Course\Agentic AI Task submission\Day10-FinalHackathon\python-backend-for-ai\agents.py
# # import os
# # import json
# # import logging
# # from typing import Dict, Any
# # from dotenv import load_dotenv
# # from langchain.agents import Tool, initialize_agent, AgentType
# # from langchain_core.prompts import PromptTemplate
# # from langchain_google_genai import ChatGoogleGenerativeAI

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Load environment variables
# # load_dotenv()
# # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # if not GOOGLE_API_KEY:
# #     raise ValueError("GOOGLE_API_KEY not found in environment variables")

# # # Initialize LLM
# # try:
# #     gemini = ChatGoogleGenerativeAI(
# #         model="gemini-1.5-flash", 
# #         google_api_key=GOOGLE_API_KEY,
# #         temperature=0.3
# #     )
# #     logger.info("Gemini model initialized successfully")
# # except Exception as e:
# #     logger.error(f"Failed to initialize Gemini: {e}")
# #     raise

# # # Simple RAG context (without FAISS)
# # def get_rag_context(query: str) -> str:
# #     """Get relevant context without vectorstore - using simple keyword matching"""
    
# #     knowledge_base = {
# #         "color": "Brand guidelines: Use professional colors. Maroon (#800000) represents elegance and sophistication. Blue (#007bff) for trust. Green (#28a745) for nature/eco-friendly.",
# #         "water": "Water bottle websites should emphasize purity, health, and freshness. Use clean, minimal designs with plenty of white space.",
# #         "ecommerce": "E-commerce sites need clear product displays, pricing, shopping cart, and contact information. Include testimonials and trust badges.",
# #         "layout": "Modern layouts use responsive design, clear navigation, hero sections, and call-to-action buttons.",
# #         "typography": "Use clean, readable fonts. Headers should be bold and attention-grabbing.",
# #         "bottle": "Product showcase should include high-quality images, specifications, benefits, and customer reviews."
# #     }
    
# #     context_parts = []
# #     query_lower = query.lower()
    
# #     for keyword, info in knowledge_base.items():
# #         if keyword in query_lower:
# #             context_parts.append(info)
    
# #     # Add default context if no matches
# #     if not context_parts:
# #         context_parts.append("Use modern, clean design principles with good color contrast and readability.")
    
# #     return "\n".join(context_parts)

# # # Shared session state
# # session_state = {
# #     "current": {
# #         "layout": "<html><body><h1>Welcome</h1><p>Start editing your website!</p></body></html>",
# #         "props": {"backgroundColor": "#ffffff", "color": "#333333"}
# #     },
# #     "history": []
# # }

# # # Tool 1: NL Command Parser Agent
# # def parse_nl_command(command: str) -> Dict[str, Any]:
# #     """Parse natural language command into structured format"""
# #     context = get_rag_context(command)
    
# #     prompt = PromptTemplate.from_template("""
# # You are an intelligent website editing command parser. Use this context for guidance:

# # CONTEXT:
# # {context}

# # Parse the following natural language command into a structured JSON format.

# # Supported actions:
# # - create_layout: Generate new HTML layout
# # - update_css: Modify styling/appearance
# # - update_content: Change text content
# # - undo: Revert last change

# # Common targets: body, header, footer, nav, main, section, div, h1, h2, p, button
# # Common CSS properties: color, backgroundColor, fontSize, fontWeight, padding, margin, textAlign

# # Examples:
# # "Create a homepage with navigation" → {{"action": "create_layout", "target": "body", "props": {{"type": "homepage"}}}}
# # "Make the header bold and blue" → {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "blue"}}}}
# # "Change the title to Welcome" → {{"action": "update_content", "target": "h1", "props": {{"text": "Welcome"}}}}

# # Command: {command}

# # Return only valid JSON:
# # """)
    
# #     try:
# #         response = gemini.invoke(prompt.format(command=command, context=context))
# #         result = json.loads(response.content.strip())
# #         logger.info(f"Parsed command: {result}")
# #         return result
# #     except Exception as e:
# #         logger.error(f"Parsing failed: {e}")
# #         return {"action": "create_layout", "target": "body", "props": {}}

# # # Tool 2: Layout Generator Agent
# # def generate_layout_agent(command: str) -> Dict[str, Any]:
# #     """Generate HTML layout based on command"""
# #     context = get_rag_context(command)
    
# #     prompt = PromptTemplate.from_template("""
# # You are a web layout generator. Use this context for styling guidance:

# # CONTEXT:
# # {context}

# # Generate a complete HTML layout for: "{command}"

# # Requirements:
# # - Create semantic HTML structure
# # - Include inline CSS styles
# # - Make it responsive and accessible
# # - Focus on the specific request (water bottle selling website)
# # - Use the specified colors (maroon: #800000)

# # Return JSON format:
# # {{
# #   "layout": "<html><head><title>Water Bottle Store</title><style>/* CSS here */</style></head><body>/* HTML content here */</body></html>",
# #   "props": {{"backgroundColor": "#ffffff", "color": "#333333", "theme": "water-bottle-store"}}
# # }}

# # Make it a complete, professional website layout.
# # """)
    
# #     try:
# #         response = gemini.invoke(prompt.format(command=command, context=context))
# #         result = json.loads(response.content.strip())
        
# #         # Store previous state
# #         session_state["history"].append(session_state["current"].copy())
# #         session_state["current"] = result
        
# #         logger.info("Generated new layout")
# #         return result
# #     except Exception as e:
# #         logger.error(f"Layout generation failed: {e}")
# #         fallback_layout = {
# #             "layout": f"""
# # <html>
# # <head>
# #     <title>Water Bottle Store</title>
# #     <style>
# #         body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
# #         .header {{ background-color: #800000; color: white; padding: 20px; text-align: center; }}
# #         .nav {{ background-color: #600000; padding: 10px; }}
# #         .nav a {{ color: white; text-decoration: none; margin: 0 15px; }}
# #         .hero {{ padding: 50px 20px; text-align: center; background: linear-gradient(135deg, #800000, #a00000); color: white; }}
# #         .products {{ padding: 40px 20px; max-width: 1200px; margin: 0 auto; }}
# #         .footer {{ background-color: #800000; color: white; padding: 20px; text-align: center; }}
# #     </style>
# # </head>
# # <body>
# #     <div class="header">
# #         <h1>Premium Water Bottles</h1>
# #         <p>Pure. Fresh. Premium Quality.</p>
# #     </div>
# #     <nav class="nav">
# #         <a href="#home">Home</a>
# #         <a href="#products">Products</a>
# #         <a href="#about">About</a>
# #         <a href="#contact">Contact</a>
# #     </nav>
# #     <div class="hero">
# #         <h2>Discover Our Premium Water Collection</h2>
# #         <p>Sustainable. Pure. Refreshing.</p>
# #         <button style="background-color: white; color: #800000; padding: 12px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">Shop Now</button>
# #     </div>
# #     <div class="products">
# #         <h2 style="text-align: center; color: #800000;">Our Products</h2>
# #         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px;">
# #             <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center;">
# #                 <h3>Premium Glass Bottle</h3>
# #                 <p>Eco-friendly glass bottles for pure taste</p>
# #                 <p style="color: #800000; font-size: 20px; font-weight: bold;">$25.99</p>
# #             </div>
# #             <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center;">
# #                 <h3>Stainless Steel Bottle</h3>
# #                 <p>Insulated bottles that keep water cold for 24 hours</p>
# #                 <p style="color: #800000; font-size: 20px; font-weight: bold;">$35.99</p>
# #             </div>
# #             <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center;">
# #                 <h3>Sports Water Bottle</h3>
# #                 <p>Perfect for workouts and outdoor activities</p>
# #                 <p style="color: #800000; font-size: 20px; font-weight: bold;">$19.99</p>
# #             </div>
# #         </div>
# #     </div>
# #     <footer class="footer">
# #         <p>&copy; 2024 Premium Water Bottles. All rights reserved.</p>
# #         <p>Contact: info@waterbottles.com | (555) 123-4567</p>
# #     </footer>
# # </body>
# # </html>
# #             """,
# #             "props": {"backgroundColor": "#f8f9fa", "color": "#333333", "theme": "maroon", "primaryColor": "#800000"}
# #         }
        
# #         session_state["history"].append(session_state["current"].copy())
# #         session_state["current"] = fallback_layout
# #         return fallback_layout

# # # Tool 3: CSS Update Agent
# # def update_css_agent(command: str) -> Dict[str, Any]:
# #     """Update CSS styling based on command"""
# #     context = get_rag_context(command)
    
# #     prompt = PromptTemplate.from_template("""
# # You are a CSS styling agent. Update the website styling based on: "{command}"

# # Context: {context}
# # Current layout: {layout}

# # Modify the CSS in the layout to match the request. Return the updated layout with new styling.

# # Return JSON format:
# # {{
# #   "layout": "updated_html_with_new_css",
# #   "props": {{"updated": "properties"}}
# # }}
# # """)
    
# #     try:
# #         current_layout = session_state["current"]["layout"]
        
# #         response = gemini.invoke(prompt.format(
# #             command=command, 
# #             context=context,
# #             layout=current_layout[:1000] + "..." if len(current_layout) > 1000 else current_layout
# #         ))
        
# #         result = json.loads(response.content.strip())
        
# #         # Store previous state
# #         session_state["history"].append(session_state["current"].copy())
# #         session_state["current"] = result
        
# #         logger.info("Updated CSS")
# #         return result
# #     except Exception as e:
# #         logger.error(f"CSS update failed: {e}")
# #         return session_state["current"]

# # # Tool 4: Undo Agent
# # def undo_agent(command: str) -> Dict[str, Any]:
# #     """Undo last change"""
# #     if session_state["history"]:
# #         session_state["current"] = session_state["history"].pop()
# #         logger.info("Undid last change")
# #         return session_state["current"]
# #     else:
# #         logger.warning("No history to undo")
# #         return {"error": "No previous state to undo"}

# # # Tool 5: Get Current State
# # def get_current_state(command: str) -> Dict[str, Any]:
# #     """Get current state of the website"""
# #     return {
# #         "current_state": session_state["current"],
# #         "history_length": len(session_state["history"])
# #     }

# # # Register tools for the agent
# # tools = [
# #     Tool(
# #         name="parse_command",
# #         func=parse_nl_command,
# #         description="Parse natural language editing commands into structured format"
# #     ),
# #     Tool(
# #         name="generate_layout",
# #         func=generate_layout_agent,
# #         description="Generate new HTML layout based on user instructions"
# #     ),
# #     Tool(
# #         name="update_css",
# #         func=update_css_agent,
# #         description="Update CSS styling and appearance of website elements"
# #     ),
# #     Tool(
# #         name="undo_change",
# #         func=undo_agent,
# #         description="Undo the last change made to the website"
# #     ),
# #     Tool(
# #         name="get_state",
# #         func=get_current_state,
# #         description="Get current state of the website"
# #     )
# # ]

# # # Initialize the main agent
# # try:
# #     agent = initialize_agent(
# #         tools=tools,
# #         llm=gemini,
# #         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# #         verbose=True,
# #         handle_parsing_errors=True,
# #         max_iterations=3
# #     )
# #     logger.info("Agent initialized successfully")
# # except Exception as e:
# #     logger.error(f"Agent initialization failed: {e}")
# #     raise

# # # Main entry point
# # def run_agentic_editor(command: str) -> Dict[str, Any]:
# #     """Main function to process user commands through the agentic system"""
# #     try:
# #         logger.info(f"Processing command: {command}")
        
# #         # Parse the command first
# #         parsed = parse_nl_command(command)
# #         action = parsed.get("action", "create_layout")
        
# #         # Route to appropriate agent based on action
# #         if action == "create_layout":
# #             result = generate_layout_agent(command)
# #         elif action == "update_css":
# #             result = update_css_agent(command)
# #         elif action == "undo":
# #             result = undo_agent(command)
# #         else:
# #             # Default to layout generation for unknown actions
# #             result = generate_layout_agent(command)
        
# #         return {
# #             "status": "success",
# #             "parsed_command": parsed,
# #             "action_taken": action,
# #             "result": result,
# #             "current_state": session_state["current"],
# #             "history_length": len(session_state["history"])
# #         }
        
# #     except Exception as e:
# #         logger.error(f"Command processing failed: {e}")
# #         return {
# #             "status": "error",
# #             "error": str(e),
# #             "current_state": session_state["current"]
# #         }

# # # Utility functions
# # def reset_session():
# #     """Reset the session state"""
# #     global session_state
# #     session_state = {
# #         "current": {
# #             "layout": "<html><body><h1>Welcome</h1><p>Start editing your website!</p></body></html>",
# #             "props": {"backgroundColor": "#ffffff", "color": "#333333"}
# #         },
# #         "history": []
# #     }
# #     logger.info("Session reset")

# # def get_session_info():
# #     """Get current session information"""
# #     return {
# #         "current_state": session_state["current"],
# #         "history_length": len(session_state["history"]),
# #         "rag_enabled": True
# #     }

# # # import os
# # # import json
# # # import logging
# # # from typing import Dict, Any, List
# # # from dotenv import load_dotenv
# # # from langchain.agents import Tool, initialize_agent, AgentType
# # # from langchain_core.prompts import PromptTemplate
# # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # from langchain.text_splitter import CharacterTextSplitter
# # # from langchain_community.vectorstores import FAISS
# # # from langchain.schema import Document

# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)

# # # # Load environment variables
# # # load_dotenv()
# # # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # # if not GOOGLE_API_KEY:
# # #     raise ValueError("GOOGLE_API_KEY not found in environment variables")

# # # # Initialize LLM and embedding models
# # # try:
# # #     gemini = ChatGoogleGenerativeAI(
# # #         model="gemini-1.5-flash", 
# # #         google_api_key=GOOGLE_API_KEY,
# # #         temperature=0.3
# # #     )
# # #     embeddings = GoogleGenerativeAIEmbeddings(
# # #         model="models/embedding-001", 
# # #         google_api_key=GOOGLE_API_KEY
# # #     )
# # # except Exception as e:
# # #     logger.error(f"Failed to initialize models: {e}")
# # #     raise

# # # # RAG Setup
# # # RAG_STORE_PATH = "rag_index_store"
# # # DOCS_FOLDER = "docs"

# # # def setup_rag_vectorstore():
# # #     """Setup or load RAG vectorstore"""
# # #     try:
# # #         if os.path.exists(RAG_STORE_PATH):
# # #             logger.info("Loading existing RAG vectorstore...")
# # #             return FAISS.load_local(RAG_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
# # #         logger.info("Creating new RAG vectorstore...")
# # #         documents = []
        
# # #         # Default documents if folder doesn't exist
# # #         default_docs = [
# # #             "Color Guidelines: Use brand colors - primary: #007bff, secondary: #6c757d. Avoid bright red for backgrounds.",
# # #             "Typography: Use clean sans-serif fonts. Headers should be bold. Body text should be readable.",
# # #             "Layout: Maintain consistent spacing. Use responsive design principles.",
# # #             "UX Guidelines: Keep navigation simple. Ensure accessibility standards.",
# # #             "Brand Voice: Professional yet approachable. Use clear, concise language."
# # #         ]
        
# # #         if os.path.exists(DOCS_FOLDER):
# # #             # Load from files
# # #             for filename in os.listdir(DOCS_FOLDER):
# # #                 if filename.endswith('.txt'):
# # #                     filepath = os.path.join(DOCS_FOLDER, filename)
# # #                     try:
# # #                         with open(filepath, 'r', encoding='utf-8') as f:
# # #                             content = f.read()
# # #                             documents.append(Document(page_content=content, metadata={"source": filename}))
# # #                     except Exception as e:
# # #                         logger.warning(f"Could not read {filepath}: {e}")
        
# # #         # Add default docs if no files found
# # #         if not documents:
# # #             for i, doc in enumerate(default_docs):
# # #                 documents.append(Document(page_content=doc, metadata={"source": f"default_{i}"}))
        
# # #         # Create vectorstore
# # #         text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# # #         splits = text_splitter.split_documents(documents)
        
# # #         vectorstore = FAISS.from_documents(splits, embeddings)
# # #         vectorstore.save_local(RAG_STORE_PATH)
# # #         logger.info(f"Created vectorstore with {len(splits)} documents")
# # #         return vectorstore
        
# # #     except Exception as e:
# # #         logger.error(f"RAG setup failed: {e}")
# # #         # Return None to disable RAG functionality
# # #         return None

# # # # Initialize vectorstore
# # # vectorstore = setup_rag_vectorstore()

# # # def get_rag_context(query: str, k: int = 3) -> str:
# # #     """Get relevant context from RAG vectorstore"""
# # #     if not vectorstore:
# # #         return "No additional context available."
    
# # #     try:
# # #         results = vectorstore.similarity_search(query, k=k)
# # #         context = "\n\n".join([doc.page_content for doc in results])
# # #         logger.info(f"Retrieved RAG context for: {query}")
# # #         return context
# # #     except Exception as e:
# # #         logger.error(f"RAG query failed: {e}")
# # #         return "No additional context available."

# # # # Shared session state with proper structure
# # # session_state = {
# # #     "current": {
# # #         "layout": "<html><body><h1>Welcome</h1><p>Start editing your website!</p></body></html>",
# # #         "props": {"backgroundColor": "#ffffff", "color": "#333333"}
# # #     },
# # #     "history": []
# # # }

# # # # Tool 1: NL Command Parser Agent
# # # def parse_nl_command(command: str) -> Dict[str, Any]:
# # #     """Parse natural language command into structured format"""
# # #     context = get_rag_context(command)
    
# # #     prompt = PromptTemplate.from_template("""
# # # You are an intelligent website editing command parser. Use this context for guidance:

# # # CONTEXT:
# # # {context}

# # # Parse the following natural language command into a structured JSON format.

# # # Supported actions:
# # # - create_layout: Generate new HTML layout
# # # - update_css: Modify styling/appearance
# # # - update_content: Change text content
# # # - undo: Revert last change

# # # Common targets: body, header, footer, nav, main, section, div, h1, h2, p, button
# # # Common CSS properties: color, backgroundColor, fontSize, fontWeight, padding, margin, textAlign

# # # Examples:
# # # "Create a homepage with navigation" → {{"action": "create_layout", "target": "body", "props": {{"type": "homepage"}}}}
# # # "Make the header bold and blue" → {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "blue"}}}}
# # # "Change the title to Welcome" → {{"action": "update_content", "target": "h1", "props": {{"text": "Welcome"}}}}

# # # Command: {command}

# # # Return only valid JSON:
# # # """)
    
# # #     try:
# # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # #         result = json.loads(response.content.strip())
# # #         logger.info(f"Parsed command: {result}")
# # #         return result
# # #     except Exception as e:
# # #         logger.error(f"Parsing failed: {e}")
# # #         return {"action": "unknown", "target": "body", "props": {}}

# # # # Tool 2: Layout Generator Agent
# # # def generate_layout_agent(command: str) -> Dict[str, Any]:
# # #     """Generate HTML layout based on command"""
# # #     context = get_rag_context(command)
    
# # #     prompt = PromptTemplate.from_template("""
# # # You are a web layout generator. Use this context for styling guidance:

# # # CONTEXT:
# # # {context}

# # # Generate a complete HTML layout based on this instruction: "{command}"

# # # Requirements:
# # # - Create semantic HTML structure
# # # - Include inline CSS styles
# # # - Make it responsive and accessible
# # # - Follow modern web design principles

# # # Return JSON format:
# # # {{
# # #   "layout": "<html><head><title>...</title></head><body>...</body></html>",
# # #   "props": {{"backgroundColor": "#ffffff", "color": "#333333", "fontFamily": "Arial, sans-serif"}}
# # # }}
# # # """)
    
# # #     try:
# # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # #         result = json.loads(response.content.strip())
        
# # #         # Store previous state
# # #         session_state["history"].append(session_state["current"].copy())
# # #         session_state["current"] = result
        
# # #         logger.info("Generated new layout")
# # #         return result
# # #     except Exception as e:
# # #         logger.error(f"Layout generation failed: {e}")
# # #         return {"error": f"Layout generation failed: {e}"}

# # # # Tool 3: CSS Update Agent
# # # def update_css_agent(command: str) -> Dict[str, Any]:
# # #     """Update CSS styling based on command"""
# # #     context = get_rag_context(command)
# # #     parsed = parse_nl_command(command)
    
# # #     prompt = PromptTemplate.from_template("""
# # # You are a CSS styling agent. Use this context for guidance:

# # # CONTEXT:
# # # {context}

# # # Apply CSS changes based on: "{command}"
# # # Current parsed intent: {parsed}

# # # Current layout: {layout}
# # # Current props: {props}

# # # Update the CSS properties appropriately. Return JSON:
# # # {{
# # #   "props": {{"color": "value", "backgroundColor": "value", ...}},
# # #   "target": "selector"
# # # }}
# # # """)
    
# # #     try:
# # #         current_layout = session_state["current"]["layout"]
# # #         current_props = session_state["current"]["props"]
        
# # #         response = gemini.invoke(prompt.format(
# # #             command=command, 
# # #             context=context,
# # #             parsed=parsed,
# # #             layout=current_layout,
# # #             props=current_props
# # #         ))
        
# # #         result = json.loads(response.content.strip())
        
# # #         # Store previous state
# # #         session_state["history"].append(session_state["current"].copy())
        
# # #         # Update current state
# # #         updated_props = session_state["current"]["props"].copy()
# # #         updated_props.update(result.get("props", {}))
# # #         session_state["current"]["props"] = updated_props
        
# # #         logger.info(f"Updated CSS: {result}")
# # #         return session_state["current"]
# # #     except Exception as e:
# # #         logger.error(f"CSS update failed: {e}")
# # #         return {"error": f"CSS update failed: {e}"}

# # # # Tool 4: Content Update Agent
# # # def update_content_agent(command: str) -> Dict[str, Any]:
# # #     """Update content based on command"""
# # #     context = get_rag_context(command)
    
# # #     prompt = PromptTemplate.from_template("""
# # # You are a content update agent. Use this context:

# # # CONTEXT:
# # # {context}

# # # Update content based on: "{command}"
# # # Current layout: {layout}

# # # Modify the HTML content appropriately. Return JSON:
# # # {{
# # #   "layout": "updated_html_here",
# # #   "props": {{"existing": "props"}}
# # # }}
# # # """)
    
# # #     try:
# # #         current_layout = session_state["current"]["layout"]
        
# # #         response = gemini.invoke(prompt.format(
# # #             command=command,
# # #             context=context,
# # #             layout=current_layout
# # #         ))
        
# # #         result = json.loads(response.content.strip())
        
# # #         # Store previous state
# # #         session_state["history"].append(session_state["current"].copy())
        
# # #         # Update current state
# # #         session_state["current"]["layout"] = result.get("layout", current_layout)
        
# # #         logger.info("Updated content")
# # #         return session_state["current"]
# # #     except Exception as e:
# # #         logger.error(f"Content update failed: {e}")
# # #         return {"error": f"Content update failed: {e}"}

# # # # Tool 5: Undo Agent
# # # def undo_agent(command: str) -> Dict[str, Any]:
# # #     """Undo last change"""
# # #     if session_state["history"]:
# # #         session_state["current"] = session_state["history"].pop()
# # #         logger.info("Undid last change")
# # #         return session_state["current"]
# # #     else:
# # #         logger.warning("No history to undo")
# # #         return {"error": "No previous state to undo"}

# # # # Tool 6: State Manager
# # # def get_current_state(command: str) -> Dict[str, Any]:
# # #     """Get current state of the website"""
# # #     return {
# # #         "current_state": session_state["current"],
# # #         "history_length": len(session_state["history"])
# # #     }

# # # # Register tools for the agent
# # # tools = [
# # #     Tool(
# # #         name="parse_command",
# # #         func=parse_nl_command,
# # #         description="Parse natural language editing commands into structured format"
# # #     ),
# # #     Tool(
# # #         name="generate_layout",
# # #         func=generate_layout_agent,
# # #         description="Generate new HTML layout based on user instructions"
# # #     ),
# # #     Tool(
# # #         name="update_css",
# # #         func=update_css_agent,
# # #         description="Update CSS styling and appearance of website elements"
# # #     ),
# # #     Tool(
# # #         name="update_content",
# # #         func=update_content_agent,
# # #         description="Update text content and HTML structure"
# # #     ),
# # #     Tool(
# # #         name="undo_change",
# # #         func=undo_agent,
# # #         description="Undo the last change made to the website"
# # #     ),
# # #     Tool(
# # #         name="get_state",
# # #         func=get_current_state,
# # #         description="Get current state of the website"
# # #     )
# # # ]

# # # # Initialize the main agent
# # # try:
# # #     agent = initialize_agent(
# # #         tools=tools,
# # #         llm=gemini,
# # #         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# # #         verbose=True,
# # #         handle_parsing_errors=True,
# # #         max_iterations=3
# # #     )
# # #     logger.info("Agent initialized successfully")
# # # except Exception as e:
# # #     logger.error(f"Agent initialization failed: {e}")
# # #     raise

# # # # Main entry point
# # # def run_agentic_editor(command: str) -> Dict[str, Any]:
# # #     """Main function to process user commands through the agentic system"""
# # #     try:
# # #         logger.info(f"Processing command: {command}")
        
# # #         # Parse the command first
# # #         parsed = parse_nl_command(command)
# # #         action = parsed.get("action", "unknown")
        
# # #         # Route to appropriate agent based on action
# # #         if action == "create_layout":
# # #             result = generate_layout_agent(command)
# # #         elif action == "update_css":
# # #             result = update_css_agent(command)
# # #         elif action == "update_content":
# # #             result = update_content_agent(command)
# # #         elif action == "undo":
# # #             result = undo_agent(command)
# # #         else:
# # #             # Let the main agent decide
# # #             result = agent.run(command)
        
# # #         return {
# # #             "status": "success",
# # #             "parsed_command": parsed,
# # #             "action_taken": action,
# # #             "result": result,
# # #             "current_state": session_state["current"],
# # #             "history_length": len(session_state["history"])
# # #         }
        
# # #     except Exception as e:
# # #         logger.error(f"Command processing failed: {e}")
# # #         return {
# # #             "status": "error",
# # #             "error": str(e),
# # #             "current_state": session_state["current"]
# # #         }

# # # # Additional utility functions
# # # def reset_session():
# # #     """Reset the session state"""
# # #     global session_state
# # #     session_state = {
# # #         "current": {
# # #             "layout": "<html><body><h1>Welcome</h1><p>Start editing your website!</p></body></html>",
# # #             "props": {"backgroundColor": "#ffffff", "color": "#333333"}
# # #         },
# # #         "history": []
# # #     }
# # #     logger.info("Session reset")

# # # def get_session_info():
# # #     """Get current session information"""
# # #     return {
# # #         "current_state": session_state["current"],
# # #         "history_length": len(session_state["history"]),
# # #         "rag_enabled": vectorstore is not None
# # #     }





# # # import os, json
# # # from dotenv import load_dotenv
# # # from langchain.agents import Tool, initialize_agent, AgentType
# # # from langchain_core.prompts import PromptTemplate
# # # from langchain_google_genai import ChatGoogleGenerativeAI

# # # # Load environment
# # # load_dotenv()
# # # gemini = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# # # # Shared in-memory session state
# # # session_state = {
# # #     "current": None,
# # #     "history": []
# # # }

# # # # 1️⃣ Natural Language Parser Tool
# # # def parse_nl_command(command: str):
# # #     prompt = PromptTemplate.from_template("""
# # # You are an intelligent parser. Convert natural language editing commands into structured JSON format.

# # # Supported keys:
# # # - action: one of [create_layout, update_css, undo]
# # # - target: valid HTML selector like body/header/footer/div
# # # - props: dictionary of CSS styles (color, backgroundColor, fontWeight, etc.)

# # # Examples:
# # # Input: "Create a new homepage layout"
# # # Output: {{"action": "create_layout", "target": "body", "props": {}}}

# # # Input: "Make the header bold and red"
# # # Output: {{"action": "update_css", "target": "header", "props": {{"fontWeight": "bold", "color": "red"}}}}

# # # Input: "Undo that change"
# # # Output: {{"action": "undo", "target": "", "props": {}}}

# # # Now parse:
# # # Instruction: {command}

# # # Respond with only valid JSON.
# # #     """)
# # #     try:
# # #         response = gemini.invoke(prompt.format(command=command))
# # #         return json.loads(response.content.strip())
# # #     except:
# # #         return {"action": "unknown", "target": "", "props": {}}

# # # # 2️⃣ Tool - Layout Generator (LLM Decides HTML + CSS)
# # # def generate_layout(command: str):
# # #     layout_prompt = PromptTemplate.from_template("""
# # # Generate a basic website HTML layout with optional inline styling based on the user's instruction.

# # # Instruction: {command}

# # # Return only valid JSON:
# # # {{
# # #   "layout": "<html>...</html>",
# # #   "props": {{ CSS key-value pairs like color, backgroundColor, etc }}
# # # }}
# # #     """)
# # #     try:
# # #         response = gemini.invoke(layout_prompt.format(command=command))
# # #         result = json.loads(response.content.strip())
# # #     except:
# # #         result = {
# # #             "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
# # #             "props": {"backgroundColor": "#000000", "color": "#ffffff"}
# # #         }

# # #     session_state["history"].append(session_state["current"])
# # #     session_state["current"] = result
# # #     return result

# # # # 3️⃣ Tool - CSS Editor
# # # def apply_css(command: str):
# # #     parsed = parse_nl_command(command)
# # #     props = parsed.get("props", {})
# # #     selector = parsed.get("target", "body")

# # #     if session_state["current"] is None:
# # #         session_state["current"] = {
# # #             "layout": "<html><body><h1>Empty Page</h1></body></html>",
# # #             "props": {}
# # #         }

# # #     updated = session_state["current"].copy()
# # #     updated["props"].update(props)
# # #     session_state["history"].append(session_state["current"])
# # #     session_state["current"] = updated
# # #     return updated

# # # # 4️⃣ Tool - Undo Agent
# # # def undo_change(_: str):
# # #     if session_state["history"]:
# # #         session_state["current"] = session_state["history"].pop()
# # #         return session_state["current"]
# # #     return {"error": "No previous state"}

# # # # 5️⃣ Tool - Fallback Parser
# # # def parse_fallback(command: str):
# # #     return parse_nl_command(command)

# # # # 🔧 Register Tools
# # # tools = [
# # #     Tool(name="generate_layout", func=generate_layout, description="Generate layout using user instruction"),
# # #     Tool(name="apply_css", func=apply_css, description="Apply CSS changes to layout"),
# # #     Tool(name="undo_change", func=undo_change, description="Undo the last change"),
# # #     Tool(name="parse_fallback", func=parse_fallback, description="Fallback parser to structure unclear commands"),
# # # ]

# # # # 🤖 Initialize Agent
# # # agent = initialize_agent(
# # #     tools=tools,
# # #     llm=gemini,
# # #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# # #     verbose=True,
# # #     handle_parsing_errors=True
# # # )

# # # # 🎯 Main Entry: Agentic Editor Controller
# # # def run_agentic_editor(command: str):
# # #     parsed = parse_nl_command(command)
# # #     result = agent.run(command)
# # #     return {
# # #         "parsed": parsed,
# # #         "tool_used": "LLM-decided",
# # #         "current_state": session_state["current"],
# # #         "result": result
# # #     }





# # # # import os, json
# # # # from dotenv import load_dotenv
# # # # from langchain.agents import Tool, initialize_agent, AgentType
# # # # from langchain_core.prompts import PromptTemplate
# # # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # # from langchain.text_splitter import CharacterTextSplitter
# # # # from langchain.docstore.document import Document
# # # # from langchain_community.vectorstores import FAISS

# # # # # Load environment variables
# # # # load_dotenv()
# # # # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # # # # Initialize LLM and embedding models
# # # # gemini = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
# # # # embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# # # # # Prepare or load FAISS vectorstore
# # # # RAG_STORE_PATH = "rag_index_store"
# # # # DOCS = [
# # # #     "docs/branding_tones.txt",
# # # #     "docs/ux_guidelines.txt",
# # # #     "docs/competitor_analysis.txt"
# # # # ]

# # # # if os.path.exists(RAG_STORE_PATH):
# # # #     vectorstore = FAISS.load_local(RAG_STORE_PATH, embeddings=embedding)
# # # # else:
# # # #     texts = []
# # # #     for file_path in DOCS:
# # # #         if os.path.exists(file_path):
# # # #             with open(file_path, "r", encoding="utf-8") as f:
# # # #                 texts.append(f.read())
# # # #     splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=40)
# # # #     docs = splitter.create_documents(texts)
# # # #     vectorstore = FAISS.from_documents(docs, embedding)
# # # #     vectorstore.save_local(RAG_STORE_PATH)

# # # # def get_rag_context(query: str, k: int = 3):
# # # #     results = vectorstore.similarity_search(query, k=k)
# # # #     context = "\n\n".join([doc.page_content for doc in results])
# # # #     print(f"\n[DEBUG] RAG context for '{query}':\n{context}\n")  # Debug output
# # # #     return context

# # # # # Shared session state
# # # # session_state = {
# # # #     "current": None,
# # # #     "history": []
# # # # }

# # # # # Tool 1: Natural Language Parser Tool
# # # # def parse_nl_command(command: str):
# # # #     context = get_rag_context(command)
# # # #     prompt = PromptTemplate.from_template("""
# # # # You are an intelligent parser. Use the following knowledge context:
# # # # {context}

# # # # Convert natural language editing commands into structured JSON format.
# # # # Supported keys:
# # # # - action: one of [create_layout, update_css, undo]
# # # # - target: valid HTML selector like body/header/footer/div
# # # # - props: dictionary of CSS styles (color, backgroundColor, fontWeight, etc.)
# # # # Now parse:
# # # # Instruction: {command}
# # # # Respond with only valid JSON.
# # # #     """)
# # # #     try:
# # # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # # #         return json.loads(response.content.strip())
# # # #     except Exception as e:
# # # #         print("[ERROR] parse_nl_command:", e)
# # # #         return {"action": "unknown", "target": "", "props": {}}

# # # # # Tool 2: Layout Generator Tool
# # # # def generate_layout(command: str):
# # # #     context = get_rag_context(command)
# # # #     prompt = PromptTemplate.from_template("""
# # # # Using the context:
# # # # {context}

# # # # Create an HTML layout with inline styles based on this instruction:
# # # # "{command}"
# # # # Respond in this JSON format:
# # # # {{
# # # #   "layout": "<html>...</html>",
# # # #   "props": {{ CSS styles like color, backgroundColor, etc }}
# # # # }}
# # # #     """)
# # # #     try:
# # # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # # #         result = json.loads(response.content.strip())
# # # #     except Exception as e:
# # # #         print("[ERROR] generate_layout:", e)
# # # #         result = {
# # # #             "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
# # # #             "props": {"backgroundColor": "#000000", "color": "#ffffff"}
# # # #         }
# # # #     session_state["history"].append(session_state["current"])
# # # #     session_state["current"] = result
# # # #     return result

# # # # # Tool 3: CSS Modifier Tool
# # # # def apply_css(command: str):
# # # #     context = get_rag_context(command)
# # # #     prompt = PromptTemplate.from_template("""
# # # # Using the context:
# # # # {context}

# # # # Extract CSS changes from the following instruction:
# # # # "{command}"
# # # # Respond only JSON like:
# # # # {{ "target": "body", "props": {{ ... }} }}
# # # #     """)
# # # #     try:
# # # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # # #         parsed = json.loads(response.content.strip())
# # # #     except Exception as e:
# # # #         print("[ERROR] apply_css:", e)
# # # #         parsed = {"target": "body", "props": {}}

# # # #     props = parsed.get("props", {})
# # # #     if session_state["current"] is None:
# # # #         session_state["current"] = {"layout": "<html><body><h1>Empty Page</h1></body></html>", "props": {}}

# # # #     updated = session_state["current"].copy()
# # # #     updated["props"].update(props)
# # # #     session_state["history"].append(session_state["current"])
# # # #     session_state["current"] = updated
# # # #     return updated

# # # # # Tool 4: Undo Tool
# # # # def undo_change(_: str):
# # # #     if session_state["history"]:
# # # #         session_state["current"] = session_state["history"].pop()
# # # #         return session_state["current"]
# # # #     return {"error": "No previous state"}

# # # # # Tool 5: Fallback Tool
# # # # def parse_fallback(command: str):
# # # #     return parse_nl_command(command)

# # # # # Register tools
# # # # TOOLS = [
# # # #     Tool(name="generate_layout", func=generate_layout, description="Generate HTML layout from instruction"),
# # # #     Tool(name="apply_css", func=apply_css, description="Apply CSS styling changes to current layout"),
# # # #     Tool(name="undo_change", func=undo_change, description="Undo last change"),
# # # #     Tool(name="parse_fallback", func=parse_fallback, description="Parse fallback for unknown instruction")
# # # # ]

# # # # # Initialize the agent
# # # # agent = initialize_agent(
# # # #     tools=TOOLS,
# # # #     llm=gemini,
# # # #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# # # #     verbose=True,
# # # #     handle_parsing_errors=True
# # # # )

# # # # # Main controller
# # # # def run_agentic_editor(command: str):
# # # #     parsed = parse_nl_command(command)
# # # #     action = parsed.get("action", "")

# # # #     if action == "create_layout":
# # # #         result = generate_layout(command)
# # # #     elif action == "update_css":
# # # #         result = apply_css(command)
# # # #     elif action == "undo":
# # # #         result = undo_change(command)
# # # #     else:
# # # #         result = agent.run(command)

# # # #     return {
# # # #         "parsed": parsed,
# # # #         "tool_used": action or "LLM-decided",
# # # #         "current_state": session_state["current"],
# # # #         "result": result
# # # #     }





# # # # # # import os, json
# # # # # # from dotenv import load_dotenv
# # # # # # from langchain.agents import Tool, initialize_agent, AgentType
# # # # # # from langchain_core.prompts import PromptTemplate
# # # # # # from langchain_google_genai import ChatGoogleGenerativeAI
# # # # # # from langchain.text_splitter import CharacterTextSplitter
# # # # # # from langchain.docstore.document import Document
# # # # # # from langchain.vectorstores import FAISS
# # # # # # from langchain.embeddings import GoogleGenerativeAIEmbeddings

# # # # # # # Load environment variables
# # # # # # load_dotenv()
# # # # # # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # # # # # # Initialize LLM and embedding models
# # # # # # gemini = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
# # # # # # embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# # # # # # # Prepare or load FAISS vectorstore
# # # # # # RAG_STORE_PATH = "rag_index_store"
# # # # # # DOCS = [
# # # # # #     "docs/branding_tones.txt",
# # # # # #     "docs/ux_guidelines.txt",
# # # # # #     "docs/competitor_analysis.txt"
# # # # # # ]

# # # # # # if os.path.exists(RAG_STORE_PATH):
# # # # # #     vectorstore = FAISS.load_local(RAG_STORE_PATH, embedding)
# # # # # # else:
# # # # # #     texts = []
# # # # # #     for file_path in DOCS:
# # # # # #         if os.path.exists(file_path):
# # # # # #             with open(file_path, "r", encoding="utf-8") as f:
# # # # # #                 texts.append(f.read())
# # # # # #     splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=40)
# # # # # #     docs = splitter.create_documents(texts)
# # # # # #     vectorstore = FAISS.from_documents(docs, embedding)
# # # # # #     vectorstore.save_local(RAG_STORE_PATH)

# # # # # # def get_rag_context(query: str, k: int = 3):
# # # # # #     results = vectorstore.similarity_search(query, k=k)
# # # # # #     context = "\n\n".join([doc.page_content for doc in results])
# # # # # #     print(f"\n[DEBUG] RAG context for '{query}':\n{context}\n")  # Debug output
# # # # # #     return context

# # # # # # # Shared session state
# # # # # # session_state = {
# # # # # #     "current": None,
# # # # # #     "history": []
# # # # # # }

# # # # # # # Tool 1: Natural Language Parser Tool
# # # # # # def parse_nl_command(command: str):
# # # # # #     context = get_rag_context(command)
# # # # # #     prompt = PromptTemplate.from_template("""
# # # # # # You are an intelligent parser. Use the following knowledge context:
# # # # # # {context}

# # # # # # Convert natural language editing commands into structured JSON format.
# # # # # # Supported keys:
# # # # # # - action: one of [create_layout, update_css, undo]
# # # # # # - target: valid HTML selector like body/header/footer/div
# # # # # # - props: dictionary of CSS styles (color, backgroundColor, fontWeight, etc.)
# # # # # # Now parse:
# # # # # # Instruction: {command}
# # # # # # Respond with only valid JSON.
# # # # # #     """)
# # # # # #     try:
# # # # # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # # # # #         return json.loads(response.content.strip())
# # # # # #     except:
# # # # # #         return {"action": "unknown", "target": "", "props": {}}

# # # # # # # Tool 2: Layout Generator Tool
# # # # # # def generate_layout(command: str):
# # # # # #     context = get_rag_context(command)
# # # # # #     prompt = PromptTemplate.from_template("""
# # # # # # Using the context:
# # # # # # {context}

# # # # # # Create an HTML layout with inline styles based on this instruction:
# # # # # # "{command}"
# # # # # # Respond in this JSON format:
# # # # # # {{
# # # # # #   "layout": "<html>...</html>",
# # # # # #   "props": {{ CSS styles like color, backgroundColor, etc }}
# # # # # # }}
# # # # # #     """)
# # # # # #     try:
# # # # # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # # # # #         result = json.loads(response.content.strip())
# # # # # #     except:
# # # # # #         result = {
# # # # # #             "layout": "<html><body><h1>Fallback Layout</h1></body></html>",
# # # # # #             "props": {"backgroundColor": "#000000", "color": "#ffffff"}
# # # # # #         }
# # # # # #     session_state["history"].append(session_state["current"])
# # # # # #     session_state["current"] = result
# # # # # #     return result

# # # # # # # Tool 3: CSS Modifier Tool
# # # # # # def apply_css(command: str):
# # # # # #     context = get_rag_context(command)
# # # # # #     prompt = PromptTemplate.from_template("""
# # # # # # Using the context:
# # # # # # {context}

# # # # # # Extract CSS changes from the following instruction:
# # # # # # "{command}"
# # # # # # Respond only JSON like:
# # # # # # {{ "target": "body", "props": {{ ... }} }}
# # # # # #     """)
# # # # # #     try:
# # # # # #         response = gemini.invoke(prompt.format(command=command, context=context))
# # # # # #         parsed = json.loads(response.content.strip())
# # # # # #     except:
# # # # # #         parsed = {"target": "body", "props": {}}

# # # # # #     props = parsed.get("props", {})
# # # # # #     if session_state["current"] is None:
# # # # # #         session_state["current"] = {"layout": "<html><body><h1>Empty Page</h1></body></html>", "props": {}}

# # # # # #     updated = session_state["current"].copy()
# # # # # #     updated["props"].update(props)
# # # # # #     session_state["history"].append(session_state["current"])
# # # # # #     session_state["current"] = updated
# # # # # #     return updated

# # # # # # # Tool 4: Undo Tool
# # # # # # def undo_change(_: str):
# # # # # #     if session_state["history"]:
# # # # # #         session_state["current"] = session_state["history"].pop()
# # # # # #         return session_state["current"]
# # # # # #     return {"error": "No previous state"}

# # # # # # # Tool 5: Fallback Tool
# # # # # # def parse_fallback(command: str):
# # # # # #     return parse_nl_command(command)

# # # # # # # Register tools
# # # # # # TOOLS = [
# # # # # #     Tool(name="generate_layout", func=generate_layout, description="Generate HTML layout from instruction"),
# # # # # #     Tool(name="apply_css", func=apply_css, description="Apply CSS styling changes to current layout"),
# # # # # #     Tool(name="undo_change", func=undo_change, description="Undo last change"),
# # # # # #     Tool(name="parse_fallback", func=parse_fallback, description="Parse fallback for unknown instruction")
# # # # # # ]

# # # # # # # Initialize the agent
# # # # # # agent = initialize_agent(
# # # # # #     tools=TOOLS,
# # # # # #     llm=gemini,
# # # # # #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# # # # # #     verbose=True,
# # # # # #     handle_parsing_errors=True
# # # # # # )

# # # # # # # Main controller
# # # # # # def run_agentic_editor(command: str):
# # # # # #     parsed = parse_nl_command(command)
# # # # # #     action = parsed.get("action", "")

# # # # # #     if action == "create_layout":
# # # # # #         result = generate_layout(command)
# # # # # #     elif action == "update_css":
# # # # # #         result = apply_css(command)
# # # # # #     elif action == "undo":
# # # # # #         result = undo_change(command)
# # # # # #     else:
# # # # # #         result = agent.run(command)

# # # # # #     return {
# # # # # #         "parsed": parsed,
# # # # # #         "tool_used": action or "LLM-decided",
# # # # # #         "current_state": session_state["current"],
# # # # # #         "result": result
# # # # # #     }
