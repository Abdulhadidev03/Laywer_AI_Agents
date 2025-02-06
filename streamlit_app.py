import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import PyPDF2
import io

load_dotenv()

def read_pdf(uploaded_file):
    """Read and extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def read_txt(uploaded_file):
    """Read uploaded text file"""
    try:
        text = uploaded_file.getvalue().decode('utf-8')
        return text
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def read_uploaded_file(uploaded_file):
    """Process uploaded file based on type"""
    if uploaded_file is None:
        return None
    
    if uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return read_txt(uploaded_file)
    else:
        return "Unsupported file type"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "file_content" not in st.session_state:
    st.session_state.file_content = None

os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ["NVIDIA_API_KEY"] = os.getenv('NVIDIA_API_KEY')

llm = LLM(
    model="groq/deepseek-r1-distill-llama-70b",
    temperature=0.7,
    max_tokens=1000,
    timeout=300,
)

llm1 = LLM(
    model="nvidia_nim/meta/llama-3.3-70b-instruct",
    temperature=0.7,
    max_tokens=50000,
    timeout=300,
    api_key=os.getenv("NVIDIA_API_KEY")
)


search_tool = SerperDevTool(n=1, filters={"site": ["legalsearch.pk", "pakistancode.gov.pk"]})

# Enhanced Case Research Analyst with file processing capabilities
case_research_analyst = Agent(
    role="Case Research Analyst",
    goal="Conduct thorough legal research on the specific query and analyze any provided documents",
    backstory="""You are an experienced legal researcher specializing in Pakistani law. 
    You analyze both user queries and provided documents, conducting comprehensive research 
    and providing detailed analysis. You process all information contextually and never 
    provide template responses.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[search_tool],
    llm=llm1
)

lawyer_agent = Agent(
    role="Legal Expert",
    goal="Analyze research findings and documents to provide detailed legal advice",
    backstory="""You are a senior Pakistani lawyer who provides comprehensive legal analysis 
    based on both research findings and document analysis. You consider all provided 
    information and context to deliver specific, actionable advice.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    llm=llm
)

def create_research_task(query, file_content):
    """Create research task with query and file content context"""
    context = f"Query: {query}\n"
    if file_content:
        context += f"\nDocument Content:\n{file_content}\n"
    
    return Task(
        description=f"""
        Analyze the following context:
        {context}
        
        Your tasks:
        1. If a document is provided, analyze its contents first
        2. Research the legal context of the query
        3. Find relevant Pakistani laws and precedents
        4. Synthesize document analysis with legal research
        5. Provide comprehensive findings
        
        Requirements:
        - Analyze any provided documents thoroughly
        - Find relevant laws and cases
        - Provide specific analysis for this case
        - No template responses
        """,
        expected_output="""
        A detailed research report including:
        - Document analysis (if provided)
        - Relevant laws and regulations
        - Case precedents
        - Specific analysis and findings
        """,
        agent=case_research_analyst
    )

def create_lawyer_task(query, file_content):
    """Create lawyer task with query and file content context"""
    context = f"Query: {query}\n"
    if file_content:
        context += f"\nDocument Content:\n{file_content}\n"
    
    return Task(
        description=f"""
        Review the following context and research findings:
        {context}
        
        Your tasks:
        1. Analyze all provided information
        2. Consider document contents (if any)
        3. Apply relevant laws to the specific case
        4. Provide practical recommendations
        5. Suggest concrete next steps
        
        Requirements:
        - Address specific details from documents
        - Provide contextual legal advice
        - Include actionable recommendations
        - No generic responses
        """,
        expected_output="""
        A comprehensive legal advisory including:
        - Analysis of all provided information
        - Specific legal implications
        - Practical recommendations
        - Clear next steps
        """,
        agent=lawyer_agent
    )

def process_query(query, file_content=None):
    """Process query and file content through AI crew"""
    try:
        # Create tasks with current context
        research_task = create_research_task(query, file_content)
        lawyer_task = create_lawyer_task(query, file_content)
        
        # Create crew with current tasks
        current_crew = Crew(
            agents=[case_research_analyst, lawyer_agent],
            tasks=[research_task, lawyer_task],
            llm=llm,
            verbose=True
        )
        
        # Process through crew
        result = current_crew.kickoff(inputs={
            "query": query,
            "file_content": file_content
        })
        
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="AI Lawyer Bot", layout="wide")
st.title("AI Lawyer Bot")

# Sidebar with file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt"])
    
    if uploaded_file:
        st.session_state.file_content = read_uploaded_file(uploaded_file)
        st.success("File uploaded and processed successfully!")
        
        # Display file content preview
        with st.expander("View Document Content"):
            st.text(st.session_state.file_content[:500] + "..." if len(st.session_state.file_content) > 500 else st.session_state.file_content)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.file_content = None
        st.rerun()

# Main chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    st.subheader("Chat with AI Lawyer")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your legal query:")

if user_input and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query and documents..."):
            response = process_query(user_input, st.session_state.file_content)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.is_processing = False
    st.rerun()