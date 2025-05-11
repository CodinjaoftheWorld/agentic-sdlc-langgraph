import os
from dotenv import load_dotenv
import re
load_dotenv() ## aloading all the environment variable

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# Importing Graph libaries
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import StrOutputParser


# Defining llm
from langchain_groq import ChatGroq
llm = ChatGroq(mode="gemma2-9b-it")


# Data modeling
# Define Graph State
from typing import List, Dict
from typing_extensions import TypedDict

class State(TypedDict):
    """
    Represents in the stage of our graph.

    Attributes:
        requirements: user requirement
        user_stories: user stories
        design_document: design document functional and technical
        code: code
        test_cases: test cases
        qa_testing: qa testing
        monitoring: monitoring
        feedback: feedback
        status: status
    """
    requirements: str
    user_stories: List[str]
    design_document: Dict[List[str], List[str]]
    code: str
    test_cases: List[str]
    qa_testing: str
    monitoring: str
    feedback: str
    status: str

class UserStories(BaseModel):
    stories: List[str]

# class Review(BaseModel):
#     status: Literal["Approved", "Not Approved"] = Field(
#         description="Overall decision on whether the input meets the expected quality and standards."
#     )
#     feedback: str = Field(
#         description="Detailed feedback that provides specific, actionable insights about strengths and weaknesses. Should include concrete suggestions for improvement with clear reasoning behind each point. For code reviews, include comments on quality, readability, performance, and adherence to best practices. For design documents, address completeness, clarity, and technical feasibility."
#     )

class DesignDocument(BaseModel):
    functional:List[str] = Field(description="Functional Design Document")
    technical:List[str] = Field(description="Technical Design Document")

class Review(BaseModel):
    review: str = Field(
        description="Detailed feedback that provides specific, actionable insights about strengths and weaknesses. Should include concrete suggestions for improvement with clear reasoning behind each point. For code reviews, include comments on quality, readability, performance, and adherence to best practices. For design documents, address completeness, clarity, and technical feasibility."
    )
    status: Literal["Approved", "Not Approved"]

class GenerateCode(BaseModel):
    generated_code: str = Field(
        description="Generated code in the format mentioned in the prompt."
    )

class TestCases(BaseModel):
    cases: List[str]




def user_input_requirements(state: State):
    return state


def auto_generate_user_stories(state:State):
    if state["requirements"] == "":
        return {"error": "Please enter requirement before generating user stories!!"}

    prompt_user_stories = PromptTemplate(

        template = """
        You are a seasoned Agile Product Manager with deep expertise in crafting user-centric user stories.

        Your task is to generate 6 well-defined user stories based on the following product requirement:

        "{requirements}"

        Guidelines for each user story:
        - Use this format: *As a <type of user>, I want to <goal or feature> so that <benefit or reason>.*
        - Focus on the user's perspective and their value—not technical implementation details.
        - Ensure each story is:
        - Concise but informative
        - Independent from the others
        - Testable, with implied or clear acceptance criteria
        - Representing a different functional aspect of the requirement

        Think broadly about user roles (e.g., admin, end user, guest, etc.) to provide well-rounded coverage of the application's core functionality.

        Return only the 6 user stories in a numbered list, with no additional explanation.
        """,
        input_variables=["requirements"]
    )
    
    chain_userstory = prompt_user_stories | llm.with_structured_output(UserStories)
    response = chain_userstory.invoke({'requirements': state['requirements']})
    state['user_stories'] = response.stories

    return state

def product_owner_review(state: State):
    
    prompt_review = PromptTemplate(
        template= """ You are a senior Product Owner with over 10 years of experience in Agile development and user story evaluation.

        Task: Review the following set of user stories using the **INVEST** criteria (Independent, Negotiable, Valuable, Estimable, Small, Testable):

        {user_stories}

        Your response should include:
        1. **Status**: Approved / Not Approved — based on whether the stories, as a whole, meet the INVEST standards.
        2. **Feedback**: Provide a clear, point-wise list with detailed observations and recommendations. Your feedback should cover:
        - Clarity and structure of the stories
        - Whether the stories provide clear and measurable user value
        - Whether acceptance criteria are present, implied, or missing
        - Any overlaps, redundancies, or dependencies
        - Suggestions for improvement, if any
        - Whether the stories collectively address the core functionality

        Format your response as follows:
        ---
        **Status**: <Approved / Not Approved>

        **Feedback**: List of 
        - Point 1: ...
        - Point 2: ...
        - Point 3: ...
        ...
        ---
        Only include the status and feedback list. Do not restate the original stories.
        """,
        input_variables= ['user_stories']
    )

    chain_review = prompt_review | llm.with_structured_output(Review)
    response = chain_review.invoke({"user_stories": "\n".join(state['user_stories'])})
    state['status'] = response.status
    state['feedback'] = response.review

    return state


def decision(state: State):
    if state['status'] == "Approved":
        return "Approved"
    else:
        return "Feedback"


def revise_user_stories(state: State):
    feedback_points = state['feedback']
    old_user_stories = state['user_stories']

    prompt_regeneration = PromptTemplate(
        template="""You are an expert in generating user stories. The Product Owner has provided feedback.

            Below is the feedback:
            {feedback_points}

            These are the previously generated user stories:
            {old_user_stories}

            Based on this feedback, regenerate exactly 5 user stories that incorporate these improvements.
            Format:
            As a <user>, I want to <action> so that <benefit>.""",
            input_variables=["feedback_points", "old_user_stories"]
    )

    chain_regeneration = prompt_regeneration | llm.with_structured_output(UserStories)
    response = chain_regeneration.invoke({
        "feedback_points": feedback_points,
        "old_user_stories": old_user_stories
    })

    state['user_stories'] = response.stories

    return state


def create_design_document(state: State):

    prompt_create_design_document = PromptTemplate(
        template="""
        You are a senior software architect with deep expertise in converting user stories into detailed functional and technical design documents.

        Task: Based on the following user stories, create two separate sections:
        1. **Functional Design Document (FDD)**
        2. **Technical Design Document (TDD)**

        User Stories:
        {user_stories}

        Guidelines:

        ### Functional Design Document (FDD)
        - Describe key features and functionality from a **user perspective**
        - Include:
        - Feature summaries
        - User flows and interactions
        - Functional requirements
        - Edge cases and validation rules
        - Assumptions and constraints
        - Use bullet points or subheadings for clarity

        ### Technical Design Document (TDD)
        - Describe how the system will be **built**
        - Include:
        - System architecture and components
        - APIs and data contracts
        - Database schema or data models
        - Technology stack and tools
        - Security and performance considerations
        - Integration points (internal or external systems)
        - Be concise but cover enough detail for implementation

        Ensure that the documents are:
        - Clear and actionable
        - Free of implementation ambiguity
        - Suitable for review by both technical and non-technical stakeholders

        Respond using the following structure:
        ---
        **Functional Design Document**

        <Your content here>

        **Technical Design Document**

        <Your content here>
        """,
        input_variables=["user_stories"]
    )

    chain_create_design_document = prompt_create_design_document | llm.with_structured_output(DesignDocument)
    response = chain_create_design_document.invoke({"user_stories": "\n".join(state['user_stories'])})
    
    state['design_document'] = {'design_document':{
                                        'functional': response.functional,
                                        'technical': response.technical}
                                        }
    
    return state


def design_review(state: State):
    prompt_design_review = PromptTemplate(
        template="""You are a senior technical architect reviewing functional and technical design documents.
            Below is the design documentation:
            {design_document}
                   
            Respond in the following format:
            - Status: Approved / Not Approved
            - Feedback: Provide feedback on Design Docs

            """,
            input_variables=["design_document"]
    )

    chain_design_review = prompt_design_review | llm.with_structured_output(Review)
    response = chain_design_review.invoke({'design_document': state['design_document']})

    state['status'] = response.status
    state['feedback'] = response.review

    return state


def revise_design_document(state: State):
    old_design_document = state['design_document']
    design_review_feedback = state['feedback']

    prompt_revise_design_document = PromptTemplate(
            template="""You are a senior technical architect correcting docs based on feedback.
            The following feedback was provided after deisgn review:

           {design_review_feedback}

           these are old docs

           {old_design_document}

            **Update the design documents to address the issues while maintaining clarity and structure. **""",
            input_variables=["design_review_feedback","old_design_document"]
        )

    chain_revise_design_document = prompt_revise_design_document | llm.with_structured_output(DesignDocument)
    response = chain_revise_design_document.invoke({"design_review_feedback": state['feedback'], "old_design_document": old_design_document})
    
    state['design_document'] = {'design_document':{
                                        'functional': response.functional,
                                        'technical': response.technical}
                                        }
    
    return state



def save_files(file_blocks, output_dir="generated_code"):
    os.makedirs(output_dir, exist_ok=True)
    for file in file_blocks:
        filepath = os.path.join(output_dir, file["filename"])
        with open(filepath, "w") as f:
            f.write(file["code"])
        print(f"✅ Saved: {file['filename']}")

def parse_files_from_response(response_text: str):
    pattern = r"Filename:\s*(?P<filename>[\w_]+\.py)\s*Code:\s*```python\n(?P<code>.*?)```"
    matches = re.finditer(pattern, response_text, re.DOTALL)

    files = []
    for match in matches:
        files.append({
            "filename": match.group("filename").strip(),
            "code": match.group("code").strip()
        })
    return files

def generate_code(state: State):
    prompt_generate_code = PromptTemplate(
        template = """
        You are a senior software architect and software engineer with deep expertise in designing modular, scalable, and production-ready systems.

        Task: Based solely on the following design documents, independently decide how the project should be split into multiple Python files, and generate full code for each file.

        Design Documents:
        {design_documents}

        Instructions:
        - Analyze the requirements carefully.
        - Based on the design, independently decide:
        - How many Python files are needed
        - What each file should be responsible for
        - Appropriate filenames (use snake_case, ending with `.py`)
        - Ensure each file:
        - Has a single clear responsibility (Single Responsibility Principle)
        - Contains clean, production-grade Python code
        - Includes necessary imports and handles module dependencies properly
        - Has comments explaining any complex logic

        Coding Guidelines:
        - Follow clean code and SOLID principles
        - Handle errors and edge cases where applicable
        - Secure the code against common vulnerabilities
        - Optimize performance wherever appropriate
        - Maintain modular, extensible structure
        - Use standard naming conventions
        - Include all necessary imports and dependencies
        - Make the code easy for another developer to understand and extend

        Output Format (strictly follow this for every file):
        ---
        Filename: <file_name.py>
        Code:
        ```python
        <Full Python code for this file>

        Important Rules:
        DO NOT include any explanations, introductions, or summaries.
        DO NOT add any text outside the specified format.
        Each file must have its own Filename and Code block as shown.
        Maintain proper Python indentation and formatting.
        Assume the generated files will be saved separately in a project folder.
        """,
            input_variables={"design_document"}
        )

    chain_code_generation = prompt_generate_code | llm.with_structured_output(GenerateCode)
    code_response = chain_code_generation.invoke({"design_documents":state['design_document']})
    
    # Parse
    files = parse_files_from_response(code_response.generated_code)
    # Save locally
    save_files(files, "generated_code")

    state['code']=code_response.generated_code

    return state

def code_review(state: State):
    """Reviews generated code based on design documents and provides feedback."""
    prompt_code_review = PromptTemplate(
        template= """You are a senior software engineer conducting a code review.
        Analyze the following code and provide feedback with Approved/Not Approved
        {generated_code}""",
        input_variables=["generated_code"]
    )

    chain_code_review = prompt_code_review | llm.with_structured_output(Review)
    response = chain_code_review.invoke(state['code'])

    state['status']=response.status
    state['feedback']=response.review

    return state


def fix_code_after_code_review(state: State):
    """Fixes code based on feedback from the Code Review process."""
    prompt_fix_code = PromptTemplate(
        template="""
        You are an expert senior software engineer responsible for fixing code quality issues.

        The following code was reviewed, and feedback was provided.

        Original Code:
        {generated_code}

        Code Review Feedback:
        {code_review_feedback}

        Your Task:
        - Correct the code to address **all feedback points** carefully.
        - Fix any security vulnerabilities, logic errors, performance issues, or code smells mentioned.
        - Improve code readability, maintainability, and adherence to best practices.
        - Ensure all imports, functions, and structures are complete and functional.

        Coding Guidelines:
            - Follow clean code and SOLID principles
            - Handle errors and edge cases where applicable
            - Secure the code against common vulnerabilities
            - Optimize performance wherever appropriate
            - Maintain modular, extensible structure
            - Use standard naming conventions
            - Include all necessary imports and dependencies
            - Make the code easy for another developer to understand and extend

        Output Format (strictly follow this for every file):
        ---
        Filename: <file_name.py>
        Code:
        ```python
        <Full Python code for this file>

        Important Rules:
        DO NOT include any explanations, introductions, or summaries.
        DO NOT add any text outside the specified format.
        Each file must have its own Filename and Code block as shown.
        Maintain proper Python indentation and formatting.
        Assume the generated files will be saved separately in a project folder.
        """,
        input_variables=["generated_code", "code_review_feedback"]
        )
    
    chain_code_fix = prompt_fix_code | llm.with_structured_output(GenerateCode)
    response_fix_code = chain_code_fix.invoke({
            "generated_code": state["code"],
            "code_review_feedback": state["feedback"]
        })
    
    # Parse
    files = parse_files_from_response(response_fix_code.generated_code)
    # Save locally
    save_files(files, "generated_code")

    state['code']=response_fix_code.generated_code

    return state


def security_review(state: State):
    """Conducts a security review of the code to check for vulnerabilities."""
    prompt_security = PromptTemplate(
        template="""You are a senior cybersecurity expert specializing in secure coding practices and vulnerability assessment.

        Task: Conduct a thorough security review of the following code:

        **Code:**
        {generated_code}

        Provide structured feedback, including detected issues and suggested fixes.
        Format:
        - Status: Approved / Not Approved
        - Feedback: (Explain security risks and provide recommended changes)
        
        """,
        input_variables="generated_code"
        )
    
    chain_security = prompt_security | llm.with_structured_output(Review)
    response_security = chain_security.invoke({
        "generated_code": state['code']
    })

    state['status'] = response_security.status
    state['feedback'] = response_security.review

    return state

def fix_code_after_security_review(state: State):
    """Fixes code based on security review feedback to eliminate vulnerabilities."""
    prompt_fix_after_security_review = PromptTemplate(
        template="""You are a cybersecurity expert and software engineer fixing security vulnerabilities.
            The following code was reviewed, and security concerns were identified:

            **Original Code:**
            {generated_code}

            **Security Review Feedback:**
            {security_review_feedback}

            Fix all security issues

            Return only the corrected code. 
            Do not include explanations.
            
            Coding Guidelines:
            - Follow clean code and SOLID principles
            - Handle errors and edge cases where applicable
            - Secure the code against common vulnerabilities
            - Optimize performance wherever appropriate
            - Maintain modular, extensible structure
            - Use standard naming conventions
            - Include all necessary imports and dependencies
            - Make the code easy for another developer to understand and extend

            Output Format (strictly follow this for every file):
            ---
            Filename: <file_name.py>
            Code:
            ```python
            <Full Python code for this file>

            Important Rules:
            DO NOT include any explanations, introductions, or summaries.
            DO NOT add any text outside the specified format.
            Each file must have its own Filename and Code block as shown.
            Maintain proper Python indentation and formatting.
            Assume the generated files will be saved separately in a project folder.

            """,
            input_variables=["generated_code", "security_review_feedback"]
        )
    chain_fix_after_security_review = prompt_fix_after_security_review | llm | StrOutputParser()
    response_fix_code = chain_fix_after_security_review.invoke({
            "generated_code": state["code"],
            "security_review_feedback": state["feedback"]
        })
    
    # Parse
    files = parse_files_from_response(response_fix_code)
    # Save locally
    save_files(files, "generated_code")

    state['code']=response_fix_code

    return state


def write_test_cases(state:State):
    """Generates test cases for the code based on functional and technical design documents."""
    prompt_test_case = PromptTemplate(
        template="""You are a senior QA engineer with expertise in comprehensive test coverage and test-driven development.

        Task: Create a comprehensive test suite for the following code and design specifications:

        **Code:**
        {generated_code}

        **Functional Design Document:**
        {functional_design}

        **Technical Design Document:**
        {technical_design}

        Generate a structured list of **unit tests, integration tests, and edge cases**.
        Use the following format:

        - **Test Case Name:** <Descriptive Name>
        - **Description:** <What the test validates>
        - **Test Steps:** <Step-by-step execution>
        - **Expected Result:** <Expected output>

        """,
        input_variables={"generated_code", "functional_design", "technical_design"}
    )

    chain_test_case = prompt_test_case | llm.with_structured_output(TestCases)
    test_cases = chain_test_case.invoke({
        "generated_code": state["code"],
        "functional_design": state["design_document"].get("functional", "No functional design available."),
        "technical_design": state["design_document"].get("technical", "No technical design available.")
    })

    state["test_cases"] = test_cases.cases
    return state


def test_cases_review(state: State):
    """Conducts a Testcase review of test cases."""
    prompt_review_testcase = PromptTemplate(
        template="""You are a senior test strategy expert reviewing the following test cases:

        **testcases:**
        {testcases}

        Provide structured feedback
        Format:
        - Status: Approved / Needs Fixes
        - Feedback: (Explain improvements)
        
        """,
        input_variables={"testcases"}
    )

    chain_review_testcase = prompt_review_testcase | llm.with_structured_output(Review)
    response = chain_review_testcase.invoke({
        "testcases": state["test_cases"]
    })
    state['status']=response.status
    state['feedback']=response.review
    
    return state


def fix_test_cases_after_review(state: State):
    """Fixes testcases based on review feedback """
    prompt_fix_testcases = PromptTemplate(
        template="""You are a Test case review expert fixing test cases.
        The following test cases was reviewed, and feedback is provided:

        **Original test cases:**
        {testcases}

        **testcases Review Feedback:**
        {feedback}

        Fix all issues

        Return only the corected test cases. 
        Do not include explanations.""",
        input_variables={"testcases", "feedback"}
    )

    chain = prompt_fix_testcases | llm.with_structured_output(TestCases)
    fixed_testcases = chain.invoke({
        "testcases": state["test_cases"],
        "feedback": state["feedback"]
    })

    state["test_cases"] = fixed_testcases.cases
    return state


def qa_testing(state: State):
    """Conducts QA testing."""
    prompt_qa_test = PromptTemplate(
        template="""You are a seasoned QA engineer with expertise in thorough testing and quality validation.

        Task: Perform a comprehensive QA evaluation of the following code and test cases:
        Perform QA testing on code {code} with test cases {testcases}
        provide status(Approved/Not Approved) and feedback
        and provide test case execution/results in feed back
        """,
        input_variables=["code","testcases"]
    )

    chain_qa_test = prompt_qa_test | llm.with_structured_output(Review)
    response = chain_qa_test.invoke({"code":state['code'], "testcases":state['test_cases']})
    state['status'] = response.status
    state['feedback'] = response.review
    
    return state

def qa_testing_result(state):
    """Returns the next step based on status and feedback."""
    if state['status']=="Approved":
        return "Passed"
    else:
        return "Failed"


def fix_code_after_qa_feedback(state: State):
    """ Fixing code after QA testing"""
    prompt = PromptTemplate(
        template="""You are an expert software engineer responsible for fixing code based on QA Feedback.
        The following is code,test cases and QA testing feedback:

        **Original Code:**
        {code}

        **testcases:**
        {testcases}

        ** qa feedback**
        {qa_feedback}

        Fix the code to address all issues 
        
        Follow the original prompt for code generation after fixing the issues:
        
        <origina prompt for code generation>
        You are a senior software architect and software engineer with deep expertise in designing modular, scalable, and production-ready systems.

        Task: Based solely on the following design documents, independently decide how the project should be split into multiple Python files, and generate full code for each file.

        Design Documents:
        {design_documents}

        Instructions:
        - Analyze the requirements carefully.
        - Based on the design, independently decide:
        - How many Python files are needed
        - What each file should be responsible for
        - Appropriate filenames (use snake_case, ending with `.py`)
        - Ensure each file:
        - Has a single clear responsibility (Single Responsibility Principle)
        - Contains clean, production-grade Python code
        - Includes necessary imports and handles module dependencies properly
        - Has comments explaining any complex logic

        Coding Guidelines:
        - Follow clean code and SOLID principles
        - Handle errors and edge cases where applicable
        - Secure the code against common vulnerabilities
        - Optimize performance wherever appropriate
        - Maintain modular, extensible structure
        - Use standard naming conventions
        - Include all necessary imports and dependencies
        - Make the code easy for another developer to understand and extend

        Output Format (strictly follow this for every file):
        ---
        Filename: <file_name.py>
        Code:
        ```python
        <Full Python code for this file>

        Important Rules:
        DO NOT include any explanations, introductions, or summaries.
        DO NOT add any text outside the specified format.
        Each file must have its own Filename and Code block as shown.
        Maintain proper Python indentation and formatting.
        Assume the generated files will be saved separately in a project folder.
        <origina prompt for code generation>
        """,
        input_variables={"code", "testcases","qa_feedback"}
    )

    chain = prompt | llm 
    qa_code = chain.invoke({
        "code": state["code"],
        "testcases": state["test_cases"],
        "qa_feedback":state['feedback']})

    state["code"] = qa_code.content

    return state


# Define LLM
llm = ChatGroq(model="gemma2-9b-it")

# Design a graph
from langgraph.graph import END, StateGraph, START


graph_builder = StateGraph(State)

# Define the nodes
graph_builder.add_node("User Requirements", user_input_requirements)
graph_builder.add_node("Auto-generate User Stories", auto_generate_user_stories)
graph_builder.add_node("Product Owner Review", product_owner_review)
graph_builder.add_node("Create Design Document", create_design_document)
graph_builder.add_node("Revise User Stories", revise_user_stories)
graph_builder.add_node("Revise Design Document", revise_design_document)
graph_builder.add_node("Design Review", design_review)
graph_builder.add_node("Generate Code", generate_code)
graph_builder.add_node("Code Review", code_review)
graph_builder.add_node("Fix Code after Code Review", fix_code_after_code_review)
graph_builder.add_node("Security Review", security_review)
graph_builder.add_node("Fix Code after Security Review", fix_code_after_security_review)
graph_builder.add_node("Write Test Cases", write_test_cases)
graph_builder.add_node("Test Cases Review", test_cases_review)
graph_builder.add_node("Fix Test Cases after Review", fix_test_cases_after_review)
graph_builder.add_node("QA Testing", qa_testing)
graph_builder.add_node("Fix Code after QA Feedback", fix_code_after_qa_feedback)


# Adding edges
graph_builder.add_edge(START, "User Requirements")
graph_builder.add_edge("User Requirements", "Auto-generate User Stories")
graph_builder.add_edge("Auto-generate User Stories", "Product Owner Review")
graph_builder.add_conditional_edges("Product Owner Review", decision, {"Approved": "Create Design Document", "Feedback": "Revise User Stories"})
graph_builder.add_edge("Revise User Stories", "Auto-generate User Stories")
graph_builder.add_edge("Create Design Document", "Design Review")
graph_builder.add_conditional_edges("Design Review", decision, {"Approved": "Generate Code", "Feedback": "Revise Design Document"})
graph_builder.add_edge("Revise Design Document", "Design Review")
graph_builder.add_edge("Generate Code", "Code Review")
graph_builder.add_conditional_edges("Code Review", decision, {"Approved": "Security Review", "Feedback": "Fix Code after Code Review"})
graph_builder.add_edge("Fix Code after Code Review", "Generate Code")
graph_builder.add_conditional_edges("Security Review", decision, {"Approved": "Write Test Cases", "Feedback": "Fix Code after Security Review"})
graph_builder.add_edge("Fix Code after Security Review", "Generate Code")
graph_builder.add_edge("Write Test Cases", "Test Cases Review")
graph_builder.add_conditional_edges("Test Cases Review", decision, {"Approved": "QA Testing", "Feedback": "Fix Test Cases after Review"})
graph_builder.add_edge("Fix Test Cases after Review", "Write Test Cases")
graph_builder.add_conditional_edges("QA Testing", qa_testing_result, {"Passed": END, "Failed": "Fix Code after QA Feedback"})
graph_builder.add_edge("Fix Code after QA Feedback", "Generate Code")
# graph_builder.add_edge("Monitoring", "Requirement Change")
# graph_builder.add_conditional_edge("Requirement Change", change, {"Yes": "User Requirements", "No": END})

# compile the graph
graph = graph_builder.compile()


# Save the PNG to a file
with open('react_graph.png', 'wb') as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Graph image saved as 'react_graph.png'. Open it to view the graph.")


requirements = input("Input the requirement here..")
initial_state = { "requirements": requirements}


from pprint import pprint
for output in graph.stream(initial_state):
    for key, value in output.items():
        pprint(f"Node '{key}':")

    pprint("\n---\n")
