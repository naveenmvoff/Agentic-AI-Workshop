# # STEP 1: Install Dependencies
# pip install -q langchain langchain-google-genai google-generativeai langchain-core

# STEP 2: Imports
import google.generativeai as genai
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import display, HTML
import uuid

# STEP 3: Gemini Setup
api_key = "AIzaSyDNOy_19avY-3W0ZA6C407mUUI-GM3ICXA"
genai.configure(api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# STEP 4: Define a tool the agent can use to generate HTML
# template = """
# You are a conversion-focused web designer and developer.

# Your task is to generate a modern, mobile-responsive HTML5 landing page for:
# - Product: {product_description}
# - Target Audience: {target_audience}
# - Tone: {brand_tone}
# - Key Features: {key_features}
# - Call to Action: {cta_preferences}
# - Persona Focus: {persona_focus}

# Requirements:
# - Use clean, semantic HTML.
# - Include a responsive layout using inline CSS (no external files).
# - Add a <header> with the product name.
# - Include a visually appealing hero section with a heading, subheading, and CTA button.
# - Use cards or columns to highlight key features.
# - Add a footer with contact info or a short brand note.
# - Do not include any image URLs unless they are placeholders.
# - Make it beautiful, skimmable, and conversion-optimized.

# Only output the HTML. Do not include explanations or Markdown formatting.
# """


template = """
You are a professional UI/UX designer and frontend developer specializing in high-conversion landing pages.

Your job is to design and generate a clean, conversion-optimized, fully mobile-responsive **HTML5 landing page** with the following inputs:

- Product: {product_description}
- Target Audience: {target_audience}
- Brand Tone: {brand_tone}
- Key Features: {key_features}
- Call-to-Action Preferences: {cta_preferences}
- Persona Focus: {persona_focus}

Your output must:
- Use clean, semantic **HTML5**
- Include **inline CSS** for responsiveness (media queries for mobile)
- Add a `<header>` with product name/logo placeholder
- Include a visually engaging **hero section** with large heading, subheading, and CTA button
- Use a **three-column or card-style layout** for feature highlights
- Include a **footer** with contact info or short tagline
- Use **soft colors**, good font hierarchy, and spacing
- Include placeholder image containers with appropriate `alt` text
- Make sure layout scales well on both desktop and mobile

⚠️ Important:
- Output only raw HTML (no markdown, no explanations)
- Keep it well-structured and beautiful
- Avoid external CSS files or frameworks (inline only)
"""

prompt_template = PromptTemplate.from_template(template)

def generate_html_with_persona(input_dict):
    final_prompt = prompt_template.format(**input_dict)
    return llm.invoke(final_prompt).content

html_tool = Tool(
    name="generate_html_variant",
    func=generate_html_with_persona,
    description="Generates a landing page HTML given product info and persona"
)

# STEP 5: Initialize agent
agent = initialize_agent(
    tools=[html_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# STEP 6: Agent Input
product_description = "An AI-powered note-taking tool designed to transcribe lectures and generate smart summaries for students."
target_audience = "University students, especially those in fast-paced courses"
brand_tone = "Friendly, helpful, tech-savvy"
key_features = "Voice transcription, automatic summarization, sync to cloud, distraction-free UI"
cta_preferences = "Try it Free, Start 7-Day Trial"

# STEP 7: Ask the agent to decide personas and generate
instruction = f"""
You are a creative AI landing page generator.

You need to:
1. Decide 3 suitable persona styles for landing pages: Emotional, Logical, Value-Driven, etc.
2. For each one, call the generate_html_variant tool and pass the correct input.
3. Return all 3 variants with headings.

Product: {product_description}
Audience: {target_audience}
Tone: {brand_tone}
Features: {key_features}
CTA: {cta_preferences}
"""

agent_response = agent.run(instruction)

# STEP 8: Display in a simple viewer
display(HTML(agent_response))
