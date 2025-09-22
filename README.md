# 🩺 **VaidyAI: Your Personal Medical AI Assistant**

## 📌 Overview

**VaidyAI** is a sophisticated, secure, and user-friendly medical AI assistant built using a **Retrieval-Augmented Generation (RAG)** architecture. It's designed to provide compassionate, accurate, and context-aware medical information. The system features a secure authentication layer, persistent chat history, and an intuitive web interface, making it a reliable tool for preliminary medical queries.

The application is split into two main components: a powerful **FastAPI** backend that handles AI logic and data management, and an interactive **Streamlit** frontend for a seamless user experience.

---

## ✨ Features

-   **Advanced RAG Pipeline**: Utilizes a FAISS vector store and a powerful LLM (Llama 3.1 via Groq) to provide answers based on a verified medical knowledge base.
-   **Secure User Authentication**: Full registration and login system using JWT (JSON Web Tokens) for secure, session-based access.
-   **Persistent Conversation History**: Chat histories are stored per-user in a MongoDB database, allowing users to resume or review past conversations.
-   **Multi-lingual & Empathetic Responses**: The AI is prompted to respond in English, Hindi, or Hinglish, adopting a compassionate and reassuring "doctor-like" tone.
-   **Full Chat Management**: Users can create new chats, switch between old conversations, and securely delete their chat history.
-   **Interactive Web Interface**: A clean, responsive, and easy-to-use chat interface built with Streamlit.
-   **Scalable Backend**: The FastAPI backend is asynchronous-ready, robust, and handles all core business logic efficiently.

---

## 🛠 Tech Stack

| **Component** | **Technology/Library** |
| ------------------- | -------------------------------------------------------- |
| **Frontend** | Streamlit                                                |
| **Backend API** | FastAPI, Uvicorn                                         |
| **Database** | MongoDB (with PyMongo)                                   |
| **AI Framework** | LangChain                                                |
| **LLM Provider** | Groq (Llama 3.1 8B Model)                                |
| **Embeddings** | HuggingFace Sentence-Transformers                        |
| **Vector Store** | FAISS (Facebook AI Similarity Search)                    |
| **Authentication** | python-jose (JWT), passlib (for password hashing)        |
| **Data Validation** | Pydantic                                                 |
| **Environment** | Python 3.9+, python-dotenv                               |

---

## 🚀 Getting Started

Follow these instructions to set up and run the VaidyAI assistant on your local machine.

### **🔹 Prerequisites**

-   **Python 3.9+**
-   A **MongoDB** instance (you can get a free one from [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register))
-   **Groq API Key** (from [GroqCloud](https://console.groq.com/keys))
-   A pre-built **FAISS vector store** in a `vectorstore/` directory.

### **📥 Installation & Configuration**

#### **1️⃣ Clone the Repository**

```bash
git clone [https://github.com/your-username/Vaidya_ai_assistant.git](https://github.com/your-username/Vaidya_ai_assistant.git)
cd Vaidya_ai_assistant
