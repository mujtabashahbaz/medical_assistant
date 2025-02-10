import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Set up API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store in env variables or manually set it
if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in the environment.")
    OPENAI_API_KEY = st.text_input("Enter OpenAI API Key:", type="password")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define medical tools
def diagnose(symptoms: str) -> str:
    """Provide possible diagnoses based on symptoms."""
    return f"Based on the symptoms ({symptoms}), possible conditions include migraines, dehydration, or tension headaches. Please consult a doctor."

def treatment_advice(condition: str) -> str:
    """Provide general treatment advice."""
    return f"For {condition}, recommended approaches include hydration, rest, and pain relievers like ibuprofen. Always seek professional medical guidance."

def medication_info(medication: str) -> str:
    """Provide details on a medication."""
    return f"{medication} is typically used for pain management. Dosage and safety depend on patient-specific factors. Consult a healthcare provider."

tools = [diagnose, treatment_advice, medication_info]

# Set up LangChain
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()
sys_msg = SystemMessage(content="You are a knowledgeable medical assistant. Provide helpful, fact-based medical information.")

# Assistant node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build LangGraph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile with memory
react_graph_memory = builder.compile(checkpointer=memory)

# Streamlit UI
st.title("🩺 Medical Query Assistant")
st.write("Ask medical-related questions, and get AI-powered insights.")

# Initialize session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Enter your medical question:", "")

if user_input:
    # Append user input to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get response
    config = {"configurable": {"thread_id": "1"}}
    response = react_graph_memory.invoke({"messages": st.session_state.chat_history}, config)

    # Append assistant response to history
    for msg in response["messages"]:
        st.session_state.chat_history.append(msg)
        st.write(f"**🤖 AI:** {msg.content}")

# Display chat history
st.subheader("📜 Conversation History")
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.write(f"**🧑 You:** {msg.content}")
    else:
        st.write(f"**🤖 AI:** {msg.content}")

# Clear button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared!")
