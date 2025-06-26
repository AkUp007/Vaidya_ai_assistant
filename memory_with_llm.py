from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import os
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS


load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def load_llm():
    return ChatOpenAI(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        openai_api_key=TOGETHER_API_KEY,
        openai_api_base="https://api.together.xyz/v1",
        temperature=0.7,
        max_tokens=512
    )

raw_prompt = """
You are a highly knowledgeable and trustworthy medical assistant.

Your task is to provide **accurate**, **concise**, and **fact-based** answers based solely on the information provided in the context. Do not generate or assume facts beyond the given content. If the context does not contain enough information, respond with: `"I don't have sufficient information to answer that."`

— Avoid speculation, hallucination, or medical advice outside the scope.
— Do NOT add any introductory phrases like "Sure", "Of course", etc.
— Format the answer in simple and clear language.
— If the context includes sources or references, cite them briefly when relevant.

---

Context:
{context}

Question:
{question}

---

Final Answer:
"""


def set_prompt(custom_prompt):
    prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    return prompt

#load data

DB_FAISS_PATH = "vectorstore/db_faiss"
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
db = FAISS.load_local(DB_FAISS_PATH, embed_model, allow_dangerous_deserialization=True)
    
# create QA chain    

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {'k':3}),
    return_source_documents = True,
    chain_type_kwargs =  {'prompt':set_prompt(raw_prompt) }
)

user_query = input("How can i help you: ")
response = qa_chain.invoke({'query': user_query})
print("result: ", response["result"])
print("source doc: ", response["source_documents"])