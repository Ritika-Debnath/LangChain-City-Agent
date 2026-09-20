# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()


# Import Gemini chat model from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Import @tool decorator to convert a Python function into a LangChain tool
from langchain.tools import tool

# Import HumanMessage to represent the user's message
from langchain.messages import HumanMessage


# ============================================================
# 1. CREATING A TOOL
# ============================================================

# @tool tells LangChain that this function can be used as an AI tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    
    # Count the number of characters in the given text
    return len(text)


# Create a dictionary to store our tools
# The tool name is used to find the correct tool later
tools = {
    "get_text_length": get_text_length
}


# ============================================================
# 2. CREATING THE LLM
# ============================================================

# Create the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)


# ============================================================
# 3. BINDING THE TOOL WITH THE LLM
# ============================================================

# Tell the LLM that it has access to the get_text_length tool
# IMPORTANT: This does NOT execute the tool
# It only makes the tool available to the LLM
llm_with_tool = llm.bind_tools([get_text_length])


# Create an empty list to store the conversation messages
message = []


# Ask the user to enter a question
prompt = input("You: ")


# Convert the user's text into a HumanMessage
query = HumanMessage(prompt)


# Add the user's message to the conversation
message.append(query)


# ============================================================
# 4. FIRST LLM CALL
# ============================================================

# Send the user's question to the LLM
# The LLM decides whether it needs to use a tool
result = llm_with_tool.invoke(message)


# Add the LLM's response/tool request to the conversation
message.append(result)


# ============================================================
# 5. CHECK WHETHER THE LLM REQUESTED A TOOL
# ============================================================

# Check if the LLM made any tool call
if result.tool_calls:

    # Get the name of the tool requested by the LLM
    # Example: "get_text_length"
    tool_name = result.tool_calls[0]["name"]

    # Find that tool from our tools dictionary
    # Then EXECUTE the tool using .invoke()
    #
    # This is the actual TOOL EXECUTION step
    tool_message = tools[tool_name].invoke(
        result.tool_calls[0]
    )

    # Add the tool's result to the conversation
    # Now the LLM will be able to see the tool result
    message.append(tool_message)


# ============================================================
# 6. SECOND LLM CALL
# ============================================================

# Send the conversation to the LLM again
# Now the LLM can see the result returned by the tool
result = llm_with_tool.invoke(message)


# Print the LLM's final answer
print(result.content)