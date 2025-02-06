import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import PyPDF2
import io

load_dotenv()

class DocumentProcessor:
    @staticmethod
    def extract_pdf_text(pdf_file):
        """Extract text from PDF with better error handling and logging"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
            text_content = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(f"--- Page {page_num + 1} ---\n{page.extract_text()}\n")
            
            return "\n".join(text_content)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

    @staticmethod
    def extract_txt_text(txt_file):
        """Extract text from TXT file with error handling"""
        try:
            return txt_file.getvalue().decode('utf-8')
        except Exception as e:
            st.error(f"Error processing TXT file: {str(e)}")
            return None

    @staticmethod
    def process_uploaded_file(uploaded_file):
        """Process uploaded file and return its content"""
        if uploaded_file is None:
            return None
        
        # Log file details for debugging
        st.sidebar.write(f"Processing file: {uploaded_file.name}")
        st.sidebar.write(f"File type: {uploaded_file.type}")
        
        if uploaded_file.type == "application/pdf":
            content = DocumentProcessor.extract_pdf_text(uploaded_file)
        elif uploaded_file.type == "text/plain":
            content = DocumentProcessor.extract_txt_text(uploaded_file)
        else:
            st.error("Unsupported file type")
            return None
            
        if content:
            # Log successful extraction
            st.sidebar.success(f"Successfully extracted content from {uploaded_file.name}")
            return content
        return None

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "current_file_content" not in st.session_state:
    st.session_state.current_file_content = None

os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

llm = LLM(
    model="groq/deepseek-r1-distill-llama-70b",
    temperature=0.7,
    max_tokens=2000,  # Increased for handling document content
    timeout=300,
)

search_tool = SerperDevTool(n=3, filters={"site": ["legalsearch.pk", "pakistancode.gov.pk"]})

case_research_analyst = Agent(
    role="Case Research Analyst",
    goal="Analyze user queries and document contents to provide comprehensive legal research",
    backstory="""You are an expert legal researcher who carefully analyzes both user queries 
    and uploaded documents. You pay special attention to document contents and integrate 
    them with your legal research. When a document is provided, you always reference 
    specific parts of it in your analysis.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    tools=[search_tool],
    llm=llm
)

lawyer_agent = Agent(
    role="Legal Expert",
    goal="Provide legal advice based on document analysis and research findings",
    backstory="""You are a senior lawyer who specializes in analyzing legal documents 
    and providing specific advice. When documents are provided, you carefully review 
    their contents and incorporate them into your legal analysis and recommendations.""",
    allow_delegation=False,
    verbose=True,
    memory=True,
    llm=llm
)

def create_document_analysis_task(query, doc_content):
    """Create a task specifically for document analysis"""
    return Task(
        description=f"""
        Analyze the following document and query:

        QUERY: {query}

        DOCUMENT CONTENT:
        {doc_content}

        Tasks:
        1. Carefully read and analyze the provided document
        2. Extract key information and legal points from the document
        3. Identify specific sections relevant to the query
        4. Connect document contents with relevant legal principles
        
        Requirements:
        - Reference specific parts of the document in your analysis
        - Quote relevant sections when appropriate
        - Explain how the document content relates to the query
        """,
        expected_output="""
        Detailed analysis including:
        - Key points from the document
        - Relevant quotes and references
        - Connection to the user's query
        - Initial legal implications
        """,
        agent=case_research_analyst
    )

def create_legal_analysis_task(query, doc_content):
    """Create a task for legal analysis incorporating document findings"""
    return Task(
        description=f"""
        Provide legal analysis based on:

        QUERY: {query}

        DOCUMENT CONTENT:
        {doc_content}

        Tasks:
        1. Review document analysis and content
        2. Apply relevant legal principles
        3. Provide specific advice based on document context
        4. Suggest practical next steps
        
        Requirements:
        - Reference specific document sections in your advice
        - Provide practical recommendations
        - Explain legal implications clearly
        """,
        expected_output="""
        Comprehensive legal advice including:
        - Analysis of document content
        - Specific legal implications
        - Practical recommendations
        - Next steps
        """,
        agent=lawyer_agent
    )

def process_query_and_document(query, doc_content=None):
    """Process both query and document content"""
    try:
        tasks = []
        
        # If document content exists, add document analysis task
        if doc_content:
            doc_task = create_document_analysis_task(query, doc_content)
            tasks.append(doc_task)
        
        # Add legal analysis task
        legal_task = create_legal_analysis_task(query, doc_content)
        tasks.append(legal_task)
        
        # Create and run crew
        crew = Crew(
            agents=[case_research_analyst, lawyer_agent],
            tasks=tasks,
            llm=llm,
            verbose=True
        )
        
        result = crew.kickoff(inputs={
            "query": query,
            "document_content": doc_content
        })
        
        return result
    except Exception as e:
        st.error(f"Error in processing: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="AI Lawyer Bot", layout="wide")
st.title("AI Lawyer Bot")

# Sidebar with enhanced file handling
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt"])
    
    if uploaded_file:
        # Process and store file content
        doc_content = DocumentProcessor.process_uploaded_file(uploaded_file)
        if doc_content:
            st.session_state.current_file_content = doc_content
            st.success(f"Successfully processed: {uploaded_file.name}")
            
            # Show preview of processed content
            with st.expander("Document Content Preview"):
                st.text(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
        else:
            st.error("Failed to process document")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.current_file_content = None
        st.rerun()

# Main chat interface
chat_container = st.container()

with chat_container:
    st.subheader("Chat with AI Lawyer")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input with document-aware processing
user_input = st.chat_input("Type your legal query:")

if user_input and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query and documents..."):
            # Process query with document content if available
            response = process_query_and_document(
                user_input, 
                st.session_state.current_file_content
            )
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.is_processing = False
    st.rerun()