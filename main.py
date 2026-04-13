import os
from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
import json
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
import jinja2

from src.librarian.rag import LibrarianRAG, SemanticChunker
from src.expert.inference import ExpertInference
from src.critic.verifier import CriticVerifier

logging.basicConfig(level=logging.INFO)

# 1. Define State
class AuditState(TypedDict):
    contract_code: str
    rag_context: str
    semantic_chunks: List[Dict[str, Any]]
    expert_findings: Dict[str, Any]
    critic_report: Dict[str, Any]
    re_evaluation_count: int
    max_retries: int

# 2. Node Functions
def librarian_node(state: AuditState) -> AuditState:
    logging.info("--- LIBRARIAN NODE ---")
    librarian = LibrarianRAG()
    # Chunking
    chunks = SemanticChunker.chunk_contract(state["contract_code"])
    state["semantic_chunks"] = chunks
    
    # RAG context formulation. For MVP, we use the contract content to find related vectors
    context = librarian.retrieve_context(state["contract_code"][:500]) # Example query
    state["rag_context"] = context
    return state

def expert_node(state: AuditState) -> AuditState:
    logging.info("--- EXPERT NODE ---")
    expert = ExpertInference()
    findings = expert.analyze_contract(state["contract_code"], state["rag_context"])
    state["expert_findings"] = findings
    return state

def critic_node(state: AuditState) -> AuditState:
    logging.info("--- CRITIC VERIFICATION NODE ---")
    critic = CriticVerifier()
    report = critic.verify(state["contract_code"], state["expert_findings"])
    state["critic_report"] = report
    return state

def conditional_edge(state: AuditState) -> str:
    report = state.get("critic_report", {})
    if report.get("hallucination_detected", False):
        if state["re_evaluation_count"] < state["max_retries"]:
            logging.warning("Hallucination detected! Routing to Re-Evaluation.")
            return "re_evaluate"
        else:
            logging.error("Max retries reached for hallucinations. Proceeding to end report.")
            return "end"
    return "end"

def re_evaluation_node(state: AuditState) -> AuditState:
    logging.info("--- RE-EVALUATION NODE ---")
    state["re_evaluation_count"] += 1
    # We feed the critic's negative feedback back into the RAG context/Expert prompt
    feedback_str = json.dumps(state.get("critic_report", {}))
    state["rag_context"] += f"\n\nCRITIC FEEDBACK: You previously hallucinated. Avoid these errors:\n{feedback_str}"
    # Returning state to be fed into Expert
    return state

# 3. Graph Formulation
def build_graph():
    workflow = StateGraph(AuditState)
    
    workflow.add_node("librarian", librarian_node)
    workflow.add_node("expert", expert_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("re_evaluate", re_evaluation_node)
    
    workflow.set_entry_point("librarian")
    workflow.add_edge("librarian", "expert")
    workflow.add_edge("expert", "critic")
    
    workflow.add_conditional_edges(
        "critic",
        conditional_edge,
        {
            "re_evaluate": "re_evaluate",
            "end": END
        }
    )
    workflow.add_edge("re_evaluate", "expert")
    return workflow.compile()

def render_report(state: AuditState, output_path: str = "report.md"):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
    template = env.get_template('report_template.md')
    
    critic_report = state.get("critic_report", {})
    context = {
        "contract_name": "Target Contract",
        "date": "TBD",
        "confidence_score": critic_report.get("confidence_score", 0),
        "findings": critic_report.get("verified_findings", []),
        "compilation_successful": critic_report.get("compilation_successful", False),
        "solc_warnings": critic_report.get("solc_warnings", ""),
    }
    
    rendered = template.render(context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)
    logging.info(f"Report written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Sentinel")
    parser.add_argument("--contract", type=str, help="Path to .sol file", required=False)
    args = parser.parse_args()
    
    app = build_graph()
    
    contract_code = "contract Example {}"
    if args.contract and os.path.exists(args.contract):
        with open(args.contract, 'r') as f:
            contract_code = f.read()
    elif args.contract:
        logging.error("File not found")
        exit(1)
        
    initial_state = {
        "contract_code": contract_code,
        "rag_context": "",
        "semantic_chunks": [],
        "expert_findings": {},
        "critic_report": {},
        "re_evaluation_count": 0,
        "max_retries": 3 # Solves loop capping
    }
    
    final_state = app.invoke(initial_state)
    render_report(final_state)
