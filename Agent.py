# ============================================================
# 1. IMPORTING THE REQUIRED LIBRARIES
# ============================================================

# Load variables from the .env file
from dotenv import load_dotenv

# Actually load the .env variables into the program
load_dotenv()

# Used to read environment variables such as API keys
import os

# Used to send HTTP requests to the OpenWeather API
import requests

# Import Google's Gemini chat model
from langchain_google_genai import ChatGoogleGenerativeAI

# Used to convert a Python function into a LangChain tool
from langchain.tools import tool

# HumanMessage = user's message
# ToolMessage = message containing the result of a tool
from langchain_core.messages import HumanMessage, ToolMessage

# TavilyClient is used to search the web and get news
from tavily import TavilyClient

# Used to create a LangChain agent
from langchain.agents import create_agent

# Used to add custom logic before/after a tool is called
from langchain.agents.middleware import wrap_tool_call


# ============================================================
# 2. CREATE THE TOOLS
# ============================================================


# ------------------------------------------------------------
# A. 🌦️ WEATHER TOOL (OPENWEATHER)
# ------------------------------------------------------------

# @tool tells LangChain that this function can be used by the AI
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    # Get the OpenWeather API key from the .env file
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # Create the URL used to request weather information
    # {city} will be replaced by the city given by the user
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    # Send a request to OpenWeather
    response = requests.get(url)

    # Convert the API response from JSON into a Python dictionary
    data = response.json()

    # Check whether the API request was successful
    # OpenWeather returns 200 when the request is successful
    if str(data.get("cod")) != "200":

        # If something went wrong, return an error message
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    # Get the temperature from the API response
    temp = data["main"]["temp"]

    # Get the weather description
    # Example: "clear sky", "light rain", etc.
    desc = data["weather"][0]["description"]

    # Return the final weather information as text
    return f"Weather in {city}: {desc}, {temp}°C"


# ------------------------------------------------------------
# B. 📰 NEWS TOOL (TAVILY)
# ------------------------------------------------------------

# Create a Tavily client using the API key from .env
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# Tell LangChain that this function is an AI tool
@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    # Ask Tavily to search for the latest news about the city
    response = tavily_client.search(
        query=f"latest news in {city}",

        # "basic" means use a basic search
        search_depth="basic",

        # Get a maximum of 3 search results
        max_results=3
    )

    # Get the list of search results
    # If there are no results, use an empty list
    results = response.get("results", [])

    # Check whether Tavily found any news
    if not results:

        # Return this message if no news was found
        return f"No news found for {city}"

    # Create an empty list to store formatted news
    news_list = []

    # Go through each news result one by one
    for r in results:

        # Get the news title
        # If there is no title, use "No title"
        title = r.get("title", "No title")

        # Get the URL of the news article
        url = r.get("url", "")

        # Get the short content/description of the article
        snippet = r.get("content", "")

        # Add the formatted news to the news_list
        # snippet[:100] keeps only the first 100 characters
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    # Join all news articles together and return them
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# ============================================================
# 3. LLM SETUP
# ============================================================

# Create the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


# ============================================================
# HUMAN APPROVAL MIDDLEWARE
# ============================================================

# @wrap_tool_call lets us add our own logic around tool calls
# Here, we use it to ask the user for permission
@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""

    # Get the name of the tool the agent wants to use
    # Example: "get_weather"
    tool_name = request.tool_call["name"]

    # Ask the user whether the tool should be executed
    confirm = input(
        f"Agent wants to call '{tool_name}'. Approve? (yes/no): "
    )

    # If the user does NOT type "yes"
    if confirm.lower() != "yes":

        # Do not execute the tool
        # Instead, send a message saying that the user denied it
        return ToolMessage(
            content="Tool call denied by user.",

            # Keep the same ID as the requested tool call
            tool_call_id=request.tool_call["id"]
        )

    # If the user typed "yes",
    # allow the tool to actually execute
    return handler(request)


# ============================================================
# 4. CREATE THE AGENT
# ============================================================

# Create an agent using Gemini and our two tools
agent = create_agent(

    # The LLM that controls the agent
    llm,

    # Give the agent access to these two tools
    tools=[get_weather, get_news],

    # Instructions that tell the agent how it should behave
    system_prompt="""
You are a helpful city assistant.

Always give answers in a clear and structured format.
Use headings, bullet points, and separate lines where appropriate.

For weather:
- City
- Temperature
- Condition

For news:
- News headline
- Short description
- Source URL
""",

    # Add our human approval middleware
    # It will ask permission before every tool call
    middleware=[human_approval]
)


# ============================================================
# 5. START THE CHAT
# ============================================================

# Display a message when the program starts
print("City Agent | type exit to quit")


# Keep asking the user for questions until they type "exit"
while True:

    # Take input from the user
    user_input = input("You : ")

    # If the user types "exit", stop the program
    if user_input.lower() == "exit":
        break

    # Send the user's question to the agent
    result = agent.invoke({

        # The agent receives the user's message
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })


    # Get the last message from the agent's response
    # Usually this contains the final answer
    response = result["messages"][-1].content


    # Sometimes the response can be a list instead of a string
    if isinstance(response, list):

        # Get the actual text from the first item
        response = response[0]["text"]


    # Print a blank line and "Bot:"
    print("\nBot:")

    # Print the final answer
    print(response)