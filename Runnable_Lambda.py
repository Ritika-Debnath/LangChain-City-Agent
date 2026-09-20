from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda      

# 1. Model
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
# 2. Output Parser
parser = StrOutputParser()


# 3. TWO DIFFERENT PROMPT TEMPLATES
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 sentences."
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail."
)

# 4. Input
topic = "Machine Learning"

# 5. RunnanbleLambda is used to extract the respective input for each prompt from the input dictionary.
#  It will run both prompts in parallel and return the results in a dictionary.
chain = RunnableParallel({
    "short" : RunnableLambda(lambda x: x ['short']) | short_prompt | model | parser,
    "detailed" : RunnableLambda(lambda x: x ['detailed']) | detailed_prompt | model | parser
})

# Here we are passing a dictionary with keys "short" and "detailed" containing the respective inputs for each prompt.
#  The RunnableLambda will extract the respective input for each prompt from the input dictionary.
result = chain.invoke({
    "short" : {"topic":"Machine Learning"},
    "detailed" : {"topic":"Deep Learning"}
})

print(result["short"])      # It will print the short explanation of the topic.
print(result["detailed"])   # It will print the detailed explanation of the topic.