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

search_tool = SerperDevTool(n=1, filters={"site": ["legalsearch.pk", "pakistancode.gov.pk"]})


llm = LLM(
    model="groq/deepseek-r1-distill-llama-70b",
    temperature=0.7,
    max_tokens=3000,
    timeout=300,
)

search_tool = SerperDevTool(n=3, filters={"site": ["legalsearch.pk", "pakistancode.gov.pk"]})

# Document Analysis Agent - Only activated when there's a document
document_analyst = Agent(
    role="Legal Document Analyst",
    goal="Analyze legal documents and extract relevant information",
    backstory="""As a specialized document analyst at our law firm, you excel at reviewing 
    legal documents and identifying key information relevant to our clients' cases. You work 
    closely with our research team to ensure all document details are properly contextualized.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    llm=llm
)

# Legal Research Agent - Always active
legal_researcher = Agent(
    role="Senior Legal Researcher",
    goal="Conduct comprehensive legal research and provide expert analysis",
    backstory="""You are a distinguished legal researcher at our prestigious Pakistani law firm. 
    You specialize in finding relevant laws, precedents, and legal frameworks for our clients' cases. 
    You work closely with our partners to build strong legal foundations for every case.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[search_tool],
    llm=llm
)

# Senior Partner - Always active
senior_partner = Agent(
    role="Senior Lawyer",
    goal="Provide authoritative legal advice and strategic direction",
    backstory="""As a senior partner at our leading Pakistani law firm, you synthesize all available 
    information to provide strategic legal advice. You have decades of experience in Pakistani law 
    and are known for your practical, results-oriented approach to legal problems.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    llm=llm
)

def create_document_analysis_task(query, file_content):
    """Create task for document analysis"""
    return Task(
        description=f"""
        As our firm's document analyst, review this document in light of the client's query:
        
        Client Query: {query}
        
        Document Content:
        {file_content}
        
        Provide a focused analysis highlighting elements relevant to the client's specific concerns.
        """,
        expected_output="""
        A clear analysis of the document including:
        - Key legal points relevant to the query
        - Important details and context
        - Potential implications for the case
        """,
        agent=document_analyst
    )

def create_research_task(query, document_analysis=None):
    """Create research task with or without document context"""
    base_description = f"""
    As our firm's senior researcher, investigate this legal matter:
    
    Client Query: {query}
    """
    
    if document_analysis:
        base_description += f"""
        Document Analysis Findings:
        {document_analysis}
        """
    
    return Task(
        description=base_description + """
        Conduct thorough research focused on:
        1. Relevant Pakistani laws and regulations
        2. Applicable case precedents
        3. Current legal framework
        """,
        expected_output="""
        A comprehensive research brief including:
        - Applicable laws and regulations
        - Relevant precedents
        - Legal framework analysis
        """,
        agent=legal_researcher
    )

def create_partner_task(query, research_findings, document_analysis=None):
    """Create partner task for final advice"""
    base_description = f"""
    As senior partner, review this case and provide strategic advice:
    
    Client Query: {query}
    
    Research Findings:
    {research_findings}
    """
    
    if document_analysis:
        base_description += f"""
        Document Analysis:
        {document_analysis}
        """
    
    return Task(
        description=base_description + """
        Provide comprehensive legal advice including:
        1. Analysis of the situation
        2. Strategic recommendations
        3. Practical next steps
        """,
        expected_output="""
        Professional legal advice including:
        - Situation analysis
        - Strategic recommendations
        - Clear action items
        """,
        agent=senior_partner
    )

def process_query(query, file_content=None):
    """Process query with dynamic task creation based on available information"""
    try:
        tasks = []
        agents = [legal_researcher, senior_partner]
        
        # If there's a document, analyze it first
        if file_content:
            agents.append(document_analyst)
            doc_task = create_document_analysis_task(query, file_content)
            tasks.append(doc_task)
            
            # Create crew for document analysis
            doc_crew = Crew(
                agents=[document_analyst],
                tasks=[doc_task],
                llm=llm,
                verbose=True
            )
            
            # Get document analysis
            doc_analysis = doc_crew.kickoff()
        else:
            doc_analysis = None
        
        # Create research task with document context if available
        research_task = create_research_task(query, doc_analysis)
        tasks.append(research_task)
        
        # Run research task
        research_crew = Crew(
            agents=[legal_researcher],
            tasks=[research_task],
            llm=llm,
            verbose=True
        )
        research_findings = research_crew.kickoff()
        
        # Create partner task with all available information
        partner_task = create_partner_task(query, research_findings, doc_analysis)
        
        # Get final advice from partner
        final_crew = Crew(
            agents=[senior_partner],
            tasks=[partner_task],
            llm=llm,
            verbose=True
        )
        
        result = final_crew.kickoff()
        return result
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Pakistani Law Firm Assistant", layout="wide")
st.title("Pakistani Law Firm Assistant")

# Sidebar with file upload
with st.sidebar:
    st.header("Document Upload (Optional)")
    uploaded_file = st.file_uploader("Upload relevant documents", type=["pdf", "txt"])
    
    if uploaded_file:
        st.session_state.file_content = read_uploaded_file(uploaded_file)
        st.success("Document processed successfully!")
        
        with st.expander("Document Preview"):
            st.text(st.session_state.file_content[:500] + "..." if len(st.session_state.file_content) > 500 else st.session_state.file_content)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.file_content = None
        st.rerun()

# Main chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    st.subheader("Consult with Our Legal Team")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("What legal matter can we assist you with today?")

if user_input and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Our legal team is analyzing your case..."):
            response = process_query(user_input, st.session_state.file_content)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.is_processing = False
    st.rerun()