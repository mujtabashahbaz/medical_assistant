import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Set API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Please set your OPENAI_API_KEY in the environment.")
    OPENAI_API_KEY = st.text_input("Enter OpenAI API Key:", type="password")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- 🚀 LANGGRAPH TOOLS ---

@tool
def check_symptoms(symptoms: str) -> str:
    """Analyze symptoms and suggest possible conditions."""
    conditions = {
        "headache": ["Migraine", "Tension headache", "Dehydration"],
        "fever": ["Flu", "Viral infection", "Dengue"],
        "chest pain": ["Heart attack", "Acid reflux", "Anxiety"],
        "stomach pain": ["Gastritis", "Appendicitis", "Food poisoning"]
    }
    matched_conditions = [cond for key, cond in conditions.items() if key in symptoms.lower()]
    return f"Possible conditions: {', '.join(sum(matched_conditions, []))}." if matched_conditions else "Consult a doctor."

@tool
def drug_interactions(medications: str) -> str:
    """Check for potential interactions between medications."""
    interactions = {
        ("aspirin", "ibuprofen"): "Increased risk of stomach bleeding.",
        ("warfarin", "aspirin"): "Increased risk of severe bleeding.",
        ("antibiotics", "antacids"): "Reduced antibiotic effectiveness."
    }
    meds = medications.lower().split(", ")
    found_interactions = [f"{pair[0].title()} and {pair[1].title()}: {desc}" for pair, desc in interactions.items() if pair[0] in meds and pair[1] in meds]
    return "\n".join(found_interactions) if found_interactions else "No known interactions found."

@tool
def treatment_advice(condition: str) -> str:
    """Provide treatment recommendations for medical conditions."""
    treatments = {
        "diabetes": "Monitor blood sugar levels, eat a balanced diet, and exercise regularly.",
        "hypertension": "Reduce salt intake, manage stress, and maintain a healthy weight.",
        "insomnia": "Maintain a consistent sleep schedule, limit screen time before bed, and avoid caffeine at night.",
        "obesity": "Increase physical activity, eat fiber-rich foods, and avoid processed sugars."
    }
    return treatments.get(condition.lower(), "Consult a medical professional for advice.")

# 🔍 Integrate DuckDuckGo Search for real-time medical info
duckduckgo_search = DuckDuckGoSearchRun(name="web_search", description="Search medical information on the web.")

# --- 🚀 BUILD LANGGRAPH AGENT ---
tools = [check_symptoms, drug_interactions, treatment_advice, duckduckgo_search]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()
sys_msg = SystemMessage(content="You are a medical assistant providing expert healthcare insights.")

# Define Assistant Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build Graph with In-Built Tool Execution
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile with memory
react_graph_memory = builder.compile(checkpointer=memory)

# --- 🚀 STREAMLIT UI ---
st.title("💊 AI Medical Assistant with Live Search")
st.write("Ask your medical questions and get AI-powered insights!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Enter your medical question:", "")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    config = {"configurable": {"thread_id": "1"}}
    response = react_graph_memory.invoke({"messages": st.session_state.chat_history}, config)

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

# Clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared!")
