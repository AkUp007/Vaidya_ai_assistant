import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import os
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Page configuration
st.set_page_config(page_title="Vaidy AI Assistant", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1, h4 {
            font-family: 'Segoe UI', sans-serif;
        }
        .stChatMessage {
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .stChatMessage.user {
            background-color: #e1f5fe;
            text-align: right;
        }
        .stChatMessage.assistant {
            background-color: #fff3e0;
        }
    </style>
""", unsafe_allow_html=True)

def load_llm():
    return ChatOpenAI(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        openai_api_key=TOGETHER_API_KEY,
        openai_api_base="https://api.together.xyz/v1",
        temperature=0.7,
        max_tokens=512
    )


DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    db = FAISS.load_local(DB_FAISS_PATH, embed_model, allow_dangerous_deserialization=True)
    return db

def set_prompt(custom_prompt):
    prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    return prompt

def main():
    st.markdown("<h1 style='text-align: center;'>🩺 Vaidy AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Your smart companion for instant medical insights and support</h4>", unsafe_allow_html=True)
    st.markdown("---")

    
    if 'messages' not in st.session_state:
        st.session_state.messages=[]
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])    
    prompt = st.chat_input("Ask anything medical...")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})
        
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

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vectorstore")
            qa_chain = RetrievalQA.from_chain_type(
                llm = load_llm(),
                chain_type = "stuff",
                retriever = vectorstore.as_retriever(search_kwargs = {'k':3}),
                return_source_documents = True,
                chain_type_kwargs =  {'prompt':set_prompt(raw_prompt) }
            )
        
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]
                source_docs = response["source_documents"]
            res_to_show = result+str(source_docs)
            st.chat_message('assistant').markdown(res_to_show)
            st.session_state.messages.append({'role':'assistant', 'content':res_to_show})
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
            
    # --- Footer section (Place this at the end) ---
        st.markdown("""
            <hr style="margin-top: 40px; margin-bottom: 10px;">
            <p style='text-align: center; font-size: 14px; color: #888;'>
                ⚙️ Powered by LangChain, TogetherAI & HuggingFace • Built with ❤️ by Akash
            </p>
        """, unsafe_allow_html=True)
        
if __name__ =="__main__":
    main()    