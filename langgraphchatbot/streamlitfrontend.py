import streamlit as st
from langchain_core.messages import HumanMessage
from langgrahbackend import chatbot  # import the chatbot from backend

# st.session_state  -> to store chat messages it is a dict like object -> it does not reset on rerun

st.title("LangGraph Chatbot")





# store the chat messages in the session state
CONFIG = {'configurable':{'thread_id':"thread-1"}}

if "messages" not in st.session_state:
    st.session_state["messages"] = []

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

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    
    ai_message = response['messages'][-1].content

    
    # Next add the AI message to message_history
    st.session_state['messages'].append({'role': 'assistant', 'content': ai_message})

    with st.chat_message('assistant'):
        st.text(ai_message)