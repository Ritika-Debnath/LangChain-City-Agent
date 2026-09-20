# ============================================================
# 1. IMPORTS
# ============================================================

import streamlit as st
from dotenv import load_dotenv
import os
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# Load API keys from .env
load_dotenv()


# ============================================================
# 2. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="City Assistant",
    page_icon="🌍"
)

st.title("🌍 City Assistant")
st.write("Ask me about weather or latest news in a city.")


# ============================================================
# 3. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    # Get OpenWeather API key
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # Create weather API URL
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )

    # Send request to OpenWeather
    response = requests.get(url)

    # Convert response into Python dictionary
    data = response.json()

    # Check if request failed
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    # Get temperature
    temp = data["main"]["temp"]

    # Get weather condition
    desc = data["weather"][0]["description"]

    # Return weather information
    return f"Weather in {city}: {desc}, {temp}°C"


# ============================================================
# 4. NEWS TOOL
# ============================================================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""

    # Search for latest news
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    # Get search results
    results = response.get("results", [])

    # If no results were found
    if not results:
        return f"No news found for {city}"

    # Store formatted news
    news_list = []

    # Process each news result
    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  🔗 {url}\n"
            f"  📝 {snippet[:100]}..."
        )

    # Return all news
    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# ============================================================
# 5. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)


# ============================================================
# 6. HUMAN APPROVAL
# ============================================================

@wrap_tool_call
def human_approval(request, handler):
    """Ask for approval before using a tool."""

    tool_name = request.tool_call["name"]

    # In Streamlit, ask for approval using session state
    # For now, automatically allow the tool.
    return handler(request)


# ============================================================
# 7. CREATE AGENT
# ============================================================

agent = create_agent(
    llm,

    # Give the agent both tools
    tools=[get_weather, get_news],

    # Instructions for the agent
    system_prompt="""
You are a helpful city assistant.

Always give answers in a clear and structured format.

For weather:
- City
- Temperature
- Condition

For news:
- News headline
- Short description
- Source URL
""",

    middleware=[human_approval]
)


# ============================================================
# 8. CHAT HISTORY
# ============================================================

# Create chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 9. CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask about weather or news..."
)


# ============================================================
# 10. PROCESS USER QUESTION
# ============================================================

if user_input:

    # Show user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user's message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show loading message while agent works
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # Send question to agent
            result = agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            })

            # Get final response
            response = result["messages"][-1].content

            # Sometimes Gemini returns content as a list
            if isinstance(response, list):
                response = response[0]["text"]

            # Display answer
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })