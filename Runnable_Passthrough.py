# RunnablePassthrough() is a runnable that simply returns the input it receives.
# It is useful when you want to pass data through a chain of runnables without modifying it.

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. Model
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
# 2. Output Parser
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")
])

sequence = code_prompt | model | parser        # This line will generate the code for the given topic and parse it to string.

sequence2 = RunnableParallel(               
    {"code" :  RunnablePassthrough(),       
     "explanation" : explain_prompt | model | parser    # This line will take the code generated from the "sequence" and explain it in simple words using the model and parse it to string.
    }
)

chain = sequence | sequence2            # Here we are chaining the two sequences. First, it will generate the code and then it will run the second sequence in parallel to explain the code.

result = chain.invoke({"topic" : "Please write a code for a palindrome in python."})

print(result["code"])         # It will print the code for palindrome in python.
print(result["explanation"])  # It will print the explanation of the code in simple words.