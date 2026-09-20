from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel 

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

# 5. It will run both prompts in parallel and return the results in a dictionary.
parallel_chain = RunnableParallel({
    "short" : short_prompt | model | parser,
    "detailed" : detailed_prompt | model | parser
})

result = parallel_chain.invoke({"topic": "Machine Learning"})   # Here "result" will be a dictionary with keys "short" and "detailed" containing the respective outputs.  

print(result["short"])      # It will print the short explanation of the topic.
print(result["detailed"])   # It will print the detailed explanation of the topic.