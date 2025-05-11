# agentic-sdlc-langgraph
Fully automated software development lifecycle (SDLC) pipeline using LangGraph and ChatGroq.

# Project: AI-Powered SDLC Automation using LangGraph
This project automates the end-to-end software delivery process—from requirements gathering to deployment-ready code—using a LangGraph-based workflow powered by ChatGroq (Gemma 2 9B). It integrates structured LLM prompting, conditional edge flows, and file generation for Python code.

# Tech Stack
LangGraph: For defining and compiling the DAG-based workflow.

ChatGroq (Gemma2-9B-IT): LLM for generating and reviewing all artifacts.

LangChain: Prompt management and structured output parsing.

Python: Core scripting and file handling.

dotenv: Environment variable management.

# Pipeline Nodes
Each node represents a key step in the SDLC:

|Node	| Description |
| --- | --- |
| User Requirements | Accepts textual input from the user |
| Auto-generate User Stories | Uses LLM to convert requirements to INVEST-aligned user stories |
| Product Owner Review | Reviews stories with criteria; conditionally moves to "Revise" |
| Revise User Stories | Regenerates stories based on feedback |
| Create Design Document | Splits into Functional and Technical Design |
| Design Review | Reviews design docs for clarity and feasibility |
| Revise Design Document | Refines design docs based on feedback |
| Generate Code | Produces multi-file, modular Python code |
| Code Review | Evaluates code quality, structure, performance |
| Fix Code after Code Review | Updates code to incorporate review feedback |
| Security Review | Evaluates code for vulnerabilities |
| Fix Code after Security Review | Fixes security flaws |
| Write Test Cases | Generates unit/integration test cases |
| Test Cases Review | Reviews quality and completeness of test cases |
| Fix Test Cases after Review | Refines tests per review |
| QA Testing | Simulates QA testing and gives pass/fail verdict |
| Fix Code after QA Feedback | Final code updates after QA |


# Features
- Structured PromptTemplates for each task

- Automatic Aapproval/feedback routing

- File parser to extract and save Python code from model output

- State-driven execution with retry paths on failure

- Mermaid-based PNG diagram generation for graph visualization

# Output
Saves generated Python files to generated_code/ folder

Outputs feedback and approval status at each step

Saves graph as react_graph.png

# Getting Started
```python
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
touch .env
# Add GROQ_API_KEY and OPENAI_API_KEY

# 3. Run the script
python your_script.py
```

# Flow Diagram
A directed graph with conditional edges, looping back on "Not Approved" responses, ensuring iterative refinement. Saved as react_graph.png in current working directory.
![react_graph](https://github.com/user-attachments/assets/b2c797b1-b6a0-420b-847f-5a45d0fcdd6b)

# TODO
 Add monitoring and deployment stages

 Integrate with GitHub Actions for CI/CD

 Streamline human review using LangGraph interrupt()

