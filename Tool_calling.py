from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool


# 1. Creating A Tool:
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

# 2. Create LLM:
llm = ChatGoogleGenerativeAI(model= "gemini-3.1-flash-lite")    # HERE IN THIS LLM THERE IS NO TOOL BINDING. 

# 3. Tool-binding with LLM:
# IN tool-binding, we are binding the tool with the LLM model. This Tell to the LLM that "what tools are avilable" , "what they do", and "when to use them" .
llm_with_tool = llm.bind_tools([get_text_length])  # HERE IN THIS LLM THERE IS TOOL BINDING. The tool "get_text_length" is bound to the LLM model "llm_with_tool". This means that the LLM model will be able to use the tool to perform specific tasks. 


result = llm.invoke("hello")
result2 = llm_with_tool.invoke("hello")
print(result)
print(result2)



# "Tool-Calling" : the LLM model choose the tool to use based on the input prompt.
#  THE LLM DOSN'T EXECUTE THE TOOL, IT JUST SUGGESTS THE TOOL TO USE. 
