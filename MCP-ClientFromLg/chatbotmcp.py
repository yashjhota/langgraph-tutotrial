# chatbotmcp.py
import asyncio
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient

# ----------------------------------------------------------------------
# 0️⃣  Load environment (e.g. GROQ_API_KEY)
# ----------------------------------------------------------------------
load_dotenv()

# ----------------------------------------------------------------------
# 1️⃣  LLM
# ----------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b")   # make sure GROQ_API_KEY is set

# ----------------------------------------------------------------------
# 2️⃣  MCP client – adjust the command if you are on Windows
# ----------------------------------------------------------------------
client = MultiServerMCPClient(
    {
       "Mathematical Computation Platform": {
        "transport":"stdio",
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "C:\\Users\\jhota\\OneDrive\\Apps\\Desktop\\file-mcp-server\\main.py"
      ],
    }
    }
)


# ----------------------------------------------------------------------
# 3️⃣  State definition
# ----------------------------------------------------------------------
class ChatState(TypedDict):
    """State carried through the graph."""
    messages: Annotated[list[BaseMessage], add_messages]

# ----------------------------------------------------------------------
# 4️⃣  Build the graph (async!)
# ----------------------------------------------------------------------
async def build_graph() -> StateGraph:
    """Create and compile the LangGraph graph."""
    # 4‑a. Pull the tool specs from the MCP server
    available_tools = await client.get_tools()
    print("🔧 Tools received from MCP server:", available_tools)

    # 4‑b. Bind the tools to the LLM so it can call them
    llm_with_tools = llm.bind_tools(available_tools)

    # ------------------------------------------------------------------
    # 4‑c. Nodes
    # ------------------------------------------------------------------
    async def chat_node(state: ChatState):
        """LLM node – either returns an answer or a tool request."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        # The LLM returns a LangChain message (AIMessage, ToolMessage, …)
        return {"messages": [response]}

    # ToolNode knows how to execute the tool calls returned by the LLM
    tool_node = ToolNode(available_tools)

    # ------------------------------------------------------------------
    # 4‑d. Checkpointer (SQLite)
    # ------------------------------------------------------------------
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    # ------------------------------------------------------------------
    # 4‑e. Assemble the graph
    # ------------------------------------------------------------------
    graph = StateGraph(ChatState, checkpointer=checkpointer)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Start → chat_node
    graph.add_edge(START, "chat_node")

    # After chat_node we may either end or go to the tool node
    graph.add_conditional_edges("chat_node", tools_condition)

    # After a tool call we go back to the chat node (to let the LLM respond)
    graph.add_edge("tools", "chat_node")

    # (Optional) you could add an explicit END edge if you want to stop
    # after a plain answer, but tools_condition already handles that.

    compiled = graph.compile()
    return compiled

# ----------------------------------------------------------------------
# 5️⃣  Main entry point
# ----------------------------------------------------------------------
async def main() -> None:
    chatbot = await build_graph()   # ← await the async builder

    # Example user query
    user_msg = HumanMessage(content="What is 5 plus 3?")

    # Run the graph – `ainvoke` returns the final state dict
    result_state = await chatbot.ainvoke({"messages": [user_msg]})

    # The last message in the state is the LLM’s answer (or a tool result)
    final_message = result_state["messages"][-1].content
    print("\n🤖 Final answer:")
    print(final_message)

if __name__ == "__main__":
    asyncio.run(main())