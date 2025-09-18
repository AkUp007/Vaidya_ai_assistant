# app.py
import os
import streamlit as st
import requests

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# --- State Management & Callbacks ---
def init_session_state():
    defaults = {"token": None, "username": None, "messages": [], "conversations": [], "current_conversation_id": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def new_chat_callback():
    st.session_state.messages = []
    st.session_state.current_conversation_id = None

def select_conversation_callback(conversation_id):
    try:
        res = requests.get(f"{API_URL}/conversations/{conversation_id}", headers=get_auth_headers())
        res.raise_for_status()
        st.session_state.messages = res.json().get("messages", [])
        st.session_state.current_conversation_id = conversation_id
    except requests.RequestException:
        st.error("Could not load chat history.")

# --- API Functions ---
def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

# --- UI Views ---
def show_auth_view():
    st.sidebar.title("Welcome")
    auth_choice = st.sidebar.radio("Get Started", ["Login", "Register"])
    with st.form("auth_form"):
        st.subheader(f"{auth_choice} to VaidyAI")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button(auth_choice):
            if auth_choice == "Login":
                try:
                    res = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
                    res.raise_for_status()
                    st.session_state.token = res.json()['access_token']
                    st.session_state.username = username
                    st.rerun()
                except requests.RequestException as e:
                    if e.response is not None:
                        try:
                            detail = e.response.json().get("detail", str(e))
                        except ValueError:
                            detail = e.response.text
                    else:
                        detail = "Connection Error"
                    st.error(f"Login failed: {detail}")
            else:  # Registration
                try:
                    res = requests.post(f"{API_URL}/register", json={"username": username, "password": password})
                    res.raise_for_status()
                    st.success("Registration successful! Please login.")
                except requests.RequestException as e:
                    if e.response is not None:
                        try:
                            detail = e.response.json().get("detail", str(e))
                        except ValueError:
                            detail = e.response.text
                    else:
                        detail = "Connection Error"
                    st.error(f"Registration failed: {detail}")


def show_chat_view():
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.sidebar.title("Chat History")
    st.sidebar.button("➕ New Chat", use_container_width=True, on_click=new_chat_callback)

    try:
        res = requests.get(f"{API_URL}/conversations", headers=get_auth_headers())
        res.raise_for_status()
        st.session_state.conversations = res.json()
    except requests.RequestException:
        st.session_state.conversations = []

    for convo in st.session_state.conversations:
        convo_id = convo.get("id") or convo.get("_id")  # fallback
        st.sidebar.button(convo['title'], key=convo_id,
                          use_container_width=True,
                          on_click=select_conversation_callback, args=(convo_id,))

    # --- Sidebar: Clear / Delete Chat History ---
    st.sidebar.title("Chat Management")

    # Clear all chat history
    if st.sidebar.button("🗑️ Clear Chat History"):
        if st.session_state.token:
            try:
                res = requests.delete(f"{API_URL}/conversations", headers=get_auth_headers())
                res.raise_for_status()
                st.success("Chat history deleted!")
                st.session_state.messages = []
                st.session_state.conversations = []
                st.session_state.current_conversation_id = None
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Failed to clear chat history: {e}")

    # Delete a single conversation
    st.sidebar.markdown("### Delete Specific Chat")
    for convo in st.session_state.conversations:
        convo_id = convo.get("id") or convo.get("_id")
        if st.sidebar.button(f"Delete: {convo['title']}", key=f"del_{convo_id}"):
            try:
                res = requests.delete(f"{API_URL}/conversations/{convo_id}", headers=get_auth_headers())
                res.raise_for_status()
                st.success(f"Deleted conversation: {convo['title']}")
                # Remove from session state
                st.session_state.conversations = [
                    c for c in st.session_state.conversations if c.get("id") != convo_id
                ]
                if st.session_state.current_conversation_id == convo_id:
                    st.session_state.current_conversation_id = None
                    st.session_state.messages = []
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Failed to delete conversation: {e}")

    # Optional: Clear messages but keep conversation
    # st.sidebar.markdown("### Clear Messages Only")
    # for convo in st.session_state.conversations:
    #     convo_id = convo.get("id") or convo.get("_id")
    #     if st.sidebar.button(f"Clear Messages: {convo['title']}", key=f"clear_{convo_id}"):
    #         try:
    #             res = requests.put(f"{API_URL}/conversations/{convo_id}/clear", headers=get_auth_headers())
    #             res.raise_for_status()
    #             st.success(f"Cleared messages in: {convo['title']}")
    #             if st.session_state.current_conversation_id == convo_id:
    #                 st.session_state.messages = []
    #             st.rerun()
    #         except requests.RequestException as e:
    #             st.error(f"Failed to clear messages: {e}")


    # st.markdown("### Conversation")
    # st.markdown("""<h4 style='text-align: center; margin-top: -10px;'>Your personal medical AI assistant</h4>""", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input("Ask anything medical..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            payload = {"prompt": prompt, "conversation_id": st.session_state.current_conversation_id}
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, headers=get_auth_headers())
                st.write("API status code:", res.status_code)
                st.write("API response:", res.text)
                res.raise_for_status()
                data = res.json()
                st.session_state.messages.append({"role": "assistant", "content": data['response']})
                if st.session_state.current_conversation_id is None:
                    st.session_state.current_conversation_id = data['conversation_id']
            except requests.RequestException as e:
                st.error(f"Error: {e}")
        st.rerun()

# --- Main App Controller ---
def main():
    init_session_state()
    st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1; font-size: 48px; margin-bottom: 5px;'>
        🩺 VaidyAI
    </h1>
    <h4 style='text-align: center; color: gray; font-size: 20px; margin-top: -10px;'>
        Your personal medical AI assistant
    </h4>
    """,
    unsafe_allow_html=True
)
    # st.markdown("---")
#     st.markdown(
#     """
#     <h1 style='text-align: center; color: #2E86C1; font-size: 48px; margin-bottom: 20px;'>
#         🩺 VaidyAI
#     </h1>
#     """,
#     unsafe_allow_html=True
# )
    
 # --- Footer section (Place this at the end) ---
    
    if not st.session_state.token:
        show_auth_view()
    else:
        show_chat_view()
        
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 46%;
            
        }
    </style>
    <div class="footer">
        <hr style="margin-top: 5px; margin-bottom: 5px;">
        <p style='text-align: center; font-size: 14px; color: #888;'>
            Built with ❤️ by Mr.Ak
        </p>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":

    main()

