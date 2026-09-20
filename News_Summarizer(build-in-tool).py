# ============== IN THIS FILE WE LEARN ABOUT "BUILD-IN TOOLS" IN LANGCHAIN AND HOW TO USE THEM.  =================

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults  # Here we import the TavilySearchResults tool which is a search tool that can be used to fetch (current) news articles or other information from the web.

search_tool = TavilySearchResults(max_result = 5)       # In this line we are creating an instance of the TavilySearchResults tool with a maximum of 5 results to be fetched.

llm = ChatGoogleGenerativeAI(model= "gemini-3.1-flash-lite")  # Set up the LLM model to be used for summarization. 

prompt = ChatPromptTemplate.from_template(          # Here we are creating a prompt template that will be used to instruct the model on how to summarize the news articles fetched by the search tool.
    """
You are a helpful assistant

summarize the following news into clear bullet points       

{news}
"""
)

chain = prompt | llm | StrOutputParser()   # Here we are creating a chain that combines the prompt, the LLM model, and an output parser to process the news articles fetched by the search tool and generate a summarized output in bullet points.

news_results = search_tool.run("Latest AI news of 2026")  # Here we are using the search tool to fetch the latest news articles related to AI in 2026. The results will be stored in the variable "news_results".

result = chain.invoke({"news": news_results})       # Here we are invoking the chain with the fetched news articles as input, which will generate a summarized output in bullet points.

print(result)