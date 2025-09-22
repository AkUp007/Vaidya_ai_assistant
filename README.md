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
```

#### **2️⃣ Create a Virtual Environment**

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### **3️⃣ Install Dependencies**
Install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### **4️⃣ Set up Environment Variables**
Create a `.env` file in the root directory of the project and add the following keys. This file stores your secret credentials secure
```bash
# .env file

# MongoDB Connection
MONGO_URI="your_mongodb_connection_string"

# Groq LLM API Key
PRIMARY_GROQ_API_KEY="your_groq_api_key"

# JWT Authentication
JWT_SECRET_KEY="a_very_strong_and_secret_key_for_jwt"
JWT_ALGORITHM="HS256"
```
### **⚙️ How to Run the Application**
The application consists of two separate services that must be run concurrently: the backend and the frontend.

#### **1️⃣ Start the FastAPI Backend**
Open a terminal, activate your virtual environment, and run the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The `--reload` flag automatically restarts the server when you make changes to the code. The API will be accessible at `http://127.0.0.1:8000`.

#### **2️⃣ Start the Streamlit Frontend
**
Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run app.py
```
Your browser should automatically open a new tab with the VaidyAI chat interface. If not, you can access it at `http://localhost:8501`.
You can now register a new user, log in, and start chatting!

### **📂 Project Structure**

```graphql
Vaidya_ai_assistant/
├── main.py             # FastAPI backend: API endpoints, RAG chain setup
├── app.py              # Streamlit frontend: UI, chat interface, API calls
├── db.py               # MongoDB connection, CRUD functions for users/conversations
├── auth.py             # JWT token creation/verification, password hashing
├── requirements.txt    # List of all Python dependencies
├── .env                # Secret keys and configuration (user-created)
└── vectorstore/
    └── db_faiss/       # Directory containing the pre-built FAISS index
```

### **🤝 Contributing**

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.
- Fork the Project
- Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
- Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
- Push to the Branch (`git push origin feature/AmazingFeature`)
- Open a Pull Request

### **📜 License**
This project is licensed under the `MIT License`. See the `LICENSE` file for more information.

### **🙌 Acknowledgments**
- Thanks to the teams behind LangChain, FastAPI, and Streamlit for their incredible open-source tools.
- Powered by the high-speed inference of Groq.
