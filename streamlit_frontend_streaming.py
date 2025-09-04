import streamlit as st
from langchain_core.messages import HumanMessage
from ui_12_backend import chatbot


# store the chat messages in the session state
CONFIG = {'configurable':{'thread_id':"thread-1"}}

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)



    
    # first add the message to message_history
 

    with st.chat_message('assistant'):
        
        ai_message =st.write_stream(
            message.content for message , metdata in chatbot.stream(
                {"messages":[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

    st.session_state['messages'].append({'role': 'assistant', 'content': ai_message})