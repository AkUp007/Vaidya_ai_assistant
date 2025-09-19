# main.py
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import db
from db import conversations_collection, get_user_conversations
import auth
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from contextlib import asynccontextmanager
from bson.objectid import ObjectId
load_dotenv()

# --- Helper function to load LLM ---
def load_llm():
    """Attempt to load LLM with GROQ primary key, fallback if needed."""
    for key in [os.getenv("PRIMARY_GROQ_API_KEY"), os.getenv("FALLBACK_GROQ_API_KEY")]:
        if key:
            try:
                return ChatOpenAI(
                    model="llama-3.1-8b-instant",
                    openai_api_key=key,
                    openai_api_base="https://api.groq.com/openai/v1",
                    temperature=0.7,
                    max_tokens=512
                )
            except Exception as e:
                print(f"GROQ API key failed: {e}")
    raise ValueError("Both GROQ API keys failed or are missing. Cannot load LLM.")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class Conversation(BaseModel):
    id: str
    title: str
    created_at: datetime

# --- Authentication ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    username = auth.verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# --- AI Setup ---
qa_chain = None

@asynccontextmanager
# @app.on_event("startup")
async def lifespan(app: FastAPI):
    global qa_chain
    try:
        llm = load_llm()
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vectorstore/db_faiss", embed_model, allow_dangerous_deserialization=True)

        # 🟢 Rich, compassionate doctor-style prompt
        raw_prompt = """
            You are a compassionate, knowledgeable, and trustworthy medical assistant, like a kind doctor speaking directly to the patient.
            Your role is to give **accurate**, **polite**, and **helpful** medical answers based on the provided context, and to communicate in a way the patient feels understood and cared for.

            — Always respond in a warm, polite, and respectful tone, similar to how a doctor calmly explains to a patient.
            - Your task is to provide **accurate**, **concise**, and **fact-based** answers based solely on the information provided in the context.
            — If the patient's question is in Hindi, respond entirely in Hindi.
            — If it's in English, respond in English.
            — If the question is mixed (Hinglish), respond in **natural Hinglish** — use a friendly, simple mix of Hindi and English, like how people speak in daily conversation (e.g., "aapko rest lena chahiye and you should consult a doctor immediately", etc.).
            — Always answer every question to the best of your ability, using only the given context only.
            — Avoid robotic or overly formal tone — sound like a real, kind doctor.
            — Always try to help. If the context lacks full information, respond gently and share basic, widely accepted medical guidance.
            — Do NOT generate facts beyond the provided context unless they are basic, well-established medical facts.
            — Never hallucinate or speculate.
            — **Format your answers in short, clear, point-wise or numbered format**, so the patient can easily follow.
            — If sources are mentioned in the context, refer to them briefly and respectfully.

            ---

            Context:
            {context}

            Question:
            {question}

            ---

            Final Answer:
        """

        prompt_template = PromptTemplate(template=raw_prompt, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt_template}
        )
        print("✅ RAG chain with compassionate doctor prompt loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load RAG chain: {e}")
        
    yield  

app = FastAPI(title="VaidyAI API", lifespan=lifespan)


# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "🩺 VaidyAI API is running."}


@app.post("/register", status_code=201)
def register(user: auth.UserCreate):
    if db.get_user(user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = auth.get_password_hash(user.password)
    db.create_user(username=user.username, hashed_password=hashed_password)
    return {"message": "User registered successfully"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.get_user(form_data.username)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = auth.create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/conversations", response_model=List[Conversation])
def get_conversations(current_user: dict = Depends(get_current_user)):
    return db.get_user_conversations(current_user["username"])

@app.get("/conversations/{conversation_id}")
def get_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    conversation = db.get_conversation_by_id(conversation_id, current_user["username"])
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": conversation.get("messages", [])}

@app.post("/chat")
def chat_endpoint(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="AI service is not available")
    try:
        response = qa_chain.invoke({"query": request.prompt})
        ai_response = response["result"]
        username = current_user["username"]
        convo_id = request.conversation_id
        user_message = {"role": "user", "content": request.prompt, "timestamp": datetime.now(timezone.utc)}
        assistant_message = {"role": "assistant", "content": ai_response, "timestamp": datetime.now(timezone.utc)}
        if convo_id:
            db.add_message_to_conversation(convo_id, user_message)
            db.add_message_to_conversation(convo_id, assistant_message)
        else:
            new_convo_id = db.create_conversation(username, user_message)
            db.add_message_to_conversation(str(new_convo_id), assistant_message)
            convo_id = str(new_convo_id)
        return {"response": ai_response, "conversation_id": convo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- Delete all conversations ---
@app.delete("/conversations")
def delete_all_user_conversations(current_user: dict = Depends(get_current_user)):
    result = conversations_collection.delete_many({"username": current_user["username"]})
    return {"message": f"Deleted {result.deleted_count} conversations"}

# --- Delete a single conversation ---
@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    try:
        result = conversations_collection.delete_one({
            "_id": ObjectId(conversation_id),
            "username": current_user["username"]
        })
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid conversation ID: {e}")

# # --- Clear messages but keep conversation metadata ---
# @app.put("/conversations/{conversation_id}/clear")
# def clear_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
#     try:
#         result = conversations_collection.update_one(
#             {"_id": ObjectId(conversation_id), "username": current_user["username"]},
#             {"$set": {"messages": []}}
#         )
#         if result.matched_count == 0:
#             raise HTTPException(status_code=404, detail="Conversation not found")
#         return {"message": "Messages cleared successfully"}
#     except Exception as e:

#         raise HTTPException(status_code=400, detail=f"Invalid conversation ID: {e}")    
