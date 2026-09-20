from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

# 2. Model
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# 3. Output Parser
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Machine Learning"})

print(result)