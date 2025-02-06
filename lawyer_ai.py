import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import PyPDF2
import io

def load_css():
    st.markdown("""
        <style>
        /* Center-align headers */
        h1, h2, h3, h4, h5, h6 {
            text-align: center;
        }
        
        /* Custom styling for the main title */
        .main-title {
            color: #1B3B6F;
            padding: 1.5rem 0;
            border-bottom: 2px solid #1B3B6F;
            margin-bottom: 2rem;
        }
        
        /* Styling for section headers */
        .section-header {
            background-color: #1B3B6F;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Custom container styling */
        .custom-container {
           
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
           
        }
        
        .user-message {
           
            border-left: 5px solid #1B3B6F;
        }
        
        .assistant-message {
          
            border-left: 5px solid #2E7D32;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1rem;
            background-color: #1B3B6F;
            color: white;
            position: fixed;
            bottom: 0;
            width: 100%;
            left: 0;
        }
        </style>
    """, unsafe_allow_html=True)

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

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

llm = LLM(
    model="groq/deepseek-r1-distill-llama-70b",
    temperature=0.7,
    max_tokens=1000,
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
    role="Senior Partner",
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
    
st.set_page_config(
        page_title="Pakistani Law Firm Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
def main():
    load_css()
    
    # Page config with custom theme
    # st.set_page_config(
    #     page_title="Pakistani Law Firm Assistant",
    #     page_icon="⚖️",
    #     layout="wide",
    #     initial_sidebar_state="expanded"
    # )
    
    # Main title with custom styling
    st.markdown('<h1 class="main-title">Pakistani Law Firm Assistant</h1>', unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 3])
    
    # Sidebar (col1) styling
    with col1:
        st.markdown('<div class="section-header">Document Management</div>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Legal Documents",
            type=["pdf", "txt"],
            help="Upload PDF or TXT files for analysis"
        )
        
        if uploaded_file:
            st.session_state.file_content = read_uploaded_file(uploaded_file)
            st.success("✅ Document processed successfully!")
            
            with st.expander("📄 Document Preview"):
                st.text(st.session_state.file_content[:500] + "..." if len(st.session_state.file_content) > 500 else st.session_state.file_content)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.file_content = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add information about the firm
        st.markdown('<div class="section-header">About Our Firm</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-container">
            <p>Our AI-powered legal assistant provides:</p>
            ✓ Expert Legal Analysis<br>
            ✓ Document Review<br>
            ✓ Case Research<br>
            ✓ Strategic Advice<br>
            </div>
        """, unsafe_allow_html=True)
    
    # Main chat area (col2)
    with col2:
        st.markdown('<div class="section-header">Legal Consultation</div>', unsafe_allow_html=True)
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                message_class = "user-message" if message["role"] == "user" else "assistant-message"
                st.markdown(
                    f'<div class="chat-message {message_class}">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
        
        # Chat input
        user_input = st.chat_input("How can we assist you with your legal matter today?")
        
        if user_input and not st.session_state.is_processing:
            st.session_state.is_processing = True
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("🤔 Our legal team is analyzing your case..."):
                response = process_query(user_input, st.session_state.file_content)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.is_processing = False
            st.rerun()
    
    # Footer
    st.markdown(
        '<div class="footer">© 2024 Pakistani Law Firm Assistant | Powered by AI</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()