# ...existing code...
import streamlit as st
from ui_12_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    return uuid.uuid4()

def get_default_name():
    return f"Chat {len(st.session_state.get('chat_threads', [])) + 1}"

def add_thread(thread_id, name=None):
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if 'thread_names' not in st.session_state:
        st.session_state['thread_names'] = {}
    # store names keyed by stringified id for stability in streamlit keys
    st.session_state['thread_names'][str(thread_id)] = name or get_default_name()

def reset_chat(name=None):
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, name=name)
    st.session_state['message_history'] = []

def get_thread_name(thread_id):
    return st.session_state.get('thread_names', {}).get(str(thread_id), str(thread_id))

def rename_thread(thread_id, new_name):
    if 'thread_names' not in st.session_state:
        st.session_state['thread_names'] = {}
    st.session_state['thread_names'][str(thread_id)] = new_name

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        if not state:
            return []
        return state.values.get('messages', []) or []
    except Exception:
        return []


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

# initialize new_chat_name BEFORE widget creation
if 'new_chat_name' not in st.session_state:
    st.session_state['new_chat_name'] = ''

# ensure current thread is registered
add_thread(st.session_state['thread_id'], name=st.session_state['thread_names'].get(str(st.session_state['thread_id'])))

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot with Threading Support 🗨️')

# text input widget (uses session_state['new_chat_name'])
st.sidebar.text_input('New chat name (optional)', key='new_chat_name')

# use a callback to modify session_state safely (avoids modifying a widget-owned key at top level)
def _new_chat_callback():
    name = st.session_state.get('new_chat_name') or None
    reset_chat(name=name)
    # clear the input via session_state inside the callback
    st.session_state['new_chat_name'] = ''

st.sidebar.button('New Chat ➕', key='new_chat_btn', on_click=_new_chat_callback)

st.sidebar.header('My Conversations 💬')

for thread_id in st.session_state['chat_threads'][::-1]:
    name = get_thread_name(thread_id)
    # use a stable key per thread to avoid duplicate button collisions
    if st.sidebar.button(name, key=f"open_{str(thread_id)}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

st.sidebar.markdown("---")
st.sidebar.write("Selected conversation:")
st.sidebar.write(get_thread_name(st.session_state['thread_id']))
new_name = st.sidebar.text_input('Rename selected conversation', value=get_thread_name(st.session_state['thread_id']), key='rename_name')
if st.sidebar.button('Rename ✏️'):
    rename_thread(st.session_state['thread_id'], new_name)


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # stream assistant response
    with st.chat_message("assistant"):
        def ai_only_stream():
            try:
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content
            except Exception as e:
                yield f"[Error streaming response: {e}]"

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
#