# app.py

import time
import google.generativeai as genai
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import display, HTML
from rag_retriever import get_rag_context

# ====== Gemini Setup ======
api_key = "AIzaSyDNOy_19avY-3W0ZA6C407mUUI-GM3ICXA"
genai.configure(api_key=api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.7,
    convert_system_message_to_human=True,
    cache=True  # Caches repeated prompts
)

# ====== Prompt Template ======
template = """
You are a professional UI/UX designer and frontend developer specializing in high-conversion landing pages.

Your job is to design and generate a clean, conversion-optimized, fully mobile-responsive HTML5 landing page with the following inputs:

- Product: {product_description}
- Target Audience: {target_audience}
- Brand Tone: {brand_tone}
- Key Features: {key_features}
- Call-to-Action Preferences: {cta_preferences}
- Persona Focus: {persona_focus}

Requirements:
- Use semantic HTML5 and inline CSS
- Responsive layout (mobile-friendly)
- <header> with logo or product name
- Hero section with heading, subheading, and CTA button
- Three-column or card-style layout for features
- Footer with contact or tagline
- Use soft colors and clear typography
- Add image placeholders with alt text

⚠️ Output ONLY raw HTML.
"""

prompt_template = PromptTemplate.from_template(template)

# ====== Tool 1: HTML Generator ======
def generate_html_with_persona(input_dict):
    time.sleep(2)  # prevent API rate limit
    rag_context = get_rag_context("landing page design for " + input_dict['product_description'])
    full_prompt = (
        "Use the following design and branding context while generating HTML:\n"
        f"{rag_context}\n\n"
        "Now create the landing page as instructed below:\n\n"
        + prompt_template.format(**input_dict)
    )
    return llm.invoke(full_prompt).content

html_tool = Tool(
    name="generate_html_variant",
    func=generate_html_with_persona,
    description="Generates landing page HTML based on persona and product info"
)

# ====== Tool 2: A/B Hypothesis Generator ======
def generate_hypotheses(_):
    time.sleep(2)  # prevent API rate limit
    prompt = """
You're a CRO expert. Generate 3 A/B test hypotheses for a landing page. 
Cover: emotional vs logical appeal, urgency vs trust in CTA, layout changes.
"""
    return llm.invoke(prompt).content

hypothesis_tool = Tool(
    name="generate_hypotheses",
    func=generate_hypotheses,
    description="Suggests A/B test hypotheses for landing page optimization"
)

# ====== Agent Setup ======
agent = initialize_agent(
    tools=[html_tool, hypothesis_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ====== Inputs ======
product_description = "An AI-powered note-taking tool designed to transcribe lectures and generate smart summaries for students."
target_audience = "University students, especially those in fast-paced courses"
brand_tone = "Friendly, helpful, tech-savvy"
key_features = "Voice transcription, automatic summarization, sync to cloud, distraction-free UI"
cta_preferences = "Try it Free, Start 7-Day Trial"

# ====== Agent Instruction ======
instruction = f"""
1. First, call the `generate_hypotheses` tool.

2. Then call `generate_html_variant` tool three times:
   - First with persona_focus="Emotional"
   - Second with persona_focus="Logical"
   - Third with persona_focus="Value-Driven"

Use these inputs for all three:
- product_description: {product_description}
- target_audience: {target_audience}
- brand_tone: {brand_tone}
- key_features: {key_features}
- cta_preferences: {cta_preferences}

Return each HTML version with clear headings: "Emotional Variant", "Logical Variant", "Value-Driven Variant".
"""

# ====== Run Agent ======
agent_response = agent.run(instruction)

# ====== Display Output ======
display(HTML(agent_response))
