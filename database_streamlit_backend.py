from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect("chatbot.db", check_same_thread=False) # By deault true it gives error because of using mulitple threads

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

# for message_chunk , metadata in chatbot.stream(
#     {"messages":[HumanMessage(content="what is the receipe of panner 65?")]},
#     config={'configurable': {'thread_id': "thread-1"}},
#     stream_mode="messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end='', flush=True)

response=chatbot.invoke(
    {"messages":[HumanMessage(content="I love india write a poem?")]},
    config={'configurable': {'thread_id': "thread-3"}}
)

# # print(chatbot.get_state(config={'configurable': {'thread_id': "thread-1"}}).values['messages'])
print(response)

