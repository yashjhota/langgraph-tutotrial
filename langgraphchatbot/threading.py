import streamlit as st
from langchain_core.messages import HumanMessage
from langgrahbackend import chatbot  # import the chatbot from backend
import uuid # to generate unique thread ids

#***************************************** UTILITY FUNCTIONS ******************************************
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["messages"] = []
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    # Placeholder for loading conversation logic
    state = chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    return state.values.get('messages', []) or []


#*************************************************SESSION STATE HANDLING******************************************

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])

#*************************************************SIDEBAR UI LOGIC******************************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header("Conversatiosn Threads")

for thread in st.session_state["chat_threads"]:
    if st.sidebar.button(thread):
        st.session_state["thread_id"] = thread
        message = load_conversation(thread)

        # Compatability handling
        temp_messages = []
        for msg in message:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

        st.session_state["messages"] = temp_messages


#************************************************* MAIN CHAT INTERFACE LOGIC******************************************
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# accept user input

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['messages'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)
   
    CONFIG = {'configurable':{'thread_id':st.session_state["thread_id"]}}

    with st.chat_message('assistant'):
        
        ai_message=st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )
        st.session_state['messages'].append({'role': 'assistant', 'content': ai_message})