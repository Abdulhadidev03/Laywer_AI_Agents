import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from litellm import completion


load_dotenv()
topic = "Lawyer firm using Generative AI"

os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
os.environ["HUGGINGFACE_API_KEY"] = os.getenv('HUGGINGFACE_API_KEY')
os.environ["NVIDIA_API_KEY"] = os.getenv('NVIDIA_API_KEY')

 
# llm = LLM(
#     model="gemini/gemini-1.5-pro-latest",
#     temperature=0.7
# )

llm = LLM(
    model="nvidia_nim/qwen/qwen2.5-7b-instruct",
    temperature=0.7,
    api_key=os.getenv("NVIDIA_API_KEY")
)





search_tool = SerperDevTool(n=1)


#Agent 1

legal_intake_specialist = Agent(
    role="Legal Intake Specialist",
    goal="Process client queries and uploaded documents, extracting key details to prepare structured data for legal analysis.",
    backstory="You are a skilled legal intake specialist with expertise in gathering and organizing client information. "
              "Your primary responsibility is to understand the context of each query or document, "
              "identify important details, and structure the data for the legal team. "
              "You excel at quickly processing complex inputs and ensuring accuracy for downstream agents.",
    allow_delegation=False,
    verbose=True,
    # tools=[file_parser_tool, text_processor_tool],
    llm=llm
)



#Agent 2

case_research_analyst = Agent(
    role="Case Research Analyst",
    goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources to support case preparation.",
    backstory="You are a highly skilled case research analyst with advanced web research skills. "
              "Your expertise lies in finding, analyzing, and synthesizing relevant legal information from reliable sources. "
              "You excel at distinguishing reliable sources, fact-checking, cross-referencing, and identifying key insights that support legal cases. "
              "You provide well-structured research briefs with proper citations, ensuring actionable insights for the legal team.",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

#Agent 3

lawyer_agent = Agent(
    role="Legal Expert",
    goal=f"Provide accurate legal advice and actionable suggestions based on the input provided by other agents.",
    backstory="You are a highly knowledgeable and empathetic lawyer with expertise in Pakistani laws. "
              "Your role is to analyze the processed data, identify relevant laws, and offer precise legal interpretations. "
              "You excel at structuring information into clear, concise, and actionable advice for users. "
              "You always maintain a professional and empathetic tone, ensuring your responses are helpful and understandable.",
    allow_delegation=False,
    verbose=True,
    # tools=[law_database_tool, legal_analysis_tool],  # Replace with actual tools/libraries for legal data processing
    llm=llm
)

legal_intake_task = Task(
    description="""
        1. Process client-provided input, including text queries and uploaded files.
        2. Extract key details and structure the information to prepare it for research.
        - Identify the legal context of the query (e.g., civil, criminal, corporate, etc.).
        - Summarize the client's concerns and categorize the main issues.
        3. Verify the completeness of the input and flag missing information.
        4. Pass the structured data to the Case Research Analyst.
    """,
    expected_output="""
        A structured summary of client input, containing:
        - Main legal context (e.g., civil, criminal, corporate, etc.)
        - Key details and facts extracted from the query or document
        - Any additional questions or clarifications needed for research
        - Well-organized sections for seamless handoff to the research agent
    """,
    agent=legal_intake_specialist
)

# Research Tasks

case_research_task = Task(
    description="""
            1. Conduct comprehensive research on {topic} including:
            - Relevant laws and regulations
            - Legal cases related to {topic}
            - Relevant articles and documents from reliable sources
            - Analyze and synthesize information from reliable sources
            - Identify key insights that support legal cases
            2. Evaluate source credibility and fact-check all information
            3. Organize finding into a structure research briekf
            4. Include all relavant citations and sources""",

    expected_output = """
            A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns

            Please format with clear sections and bullet points for easy reference.
            """,
    agent=case_research_analyst        

)

#lawyer tasks

lawyer_agent_task = Task(
    description="""
        1. Review the structured output from the Case Research Analyst.
        2. Analyze the provided information and generate a detailed legal opinion.
        - Identify the legal violations, relevant acts, or regulations.
        - Provide actionable legal advice based on the case details.
        3. Suggest steps the client should take, including potential legal actions.
        4. Format the response in a professional and concise legal style.
    """,
    expected_output="""
        A comprehensive legal response, containing:
        - Identification of legal violations or issues
        - References to specific laws, acts, or regulations
        - Clear and actionable advice for the client
        - Suggested steps or actions to proceed with their case
        - Professional tone with structured formatting for easy readability
    """,
    agent=lawyer_agent
)


crew = Crew(
    agents=[legal_intake_specialist, case_research_analyst, lawyer_agent],
    tasks=[legal_intake_task, case_research_task, lawyer_agent_task],
    llm=llm,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": topic})

print(result)