import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# Use the python port of solidity-parser or solidity_parser
try:
    from solidity_parser import parser
except ImportError:
    logging.warning("Please install solidity-parser using pip install solidity-parser")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "vector_store"

class SemanticChunker:
    """
    Implements the Code Chunking Rule: NEVER split code arbitrarily.
    Uses solidity-parser to preserve function-level logic and state transitions.
    """
    @staticmethod
    def chunk_contract(solidity_code: str) -> List[Dict[str, Any]]:
        chunks = []
        try:
            source_unit = parser.parse(solidity_code)
            
            # Simple AST traversal to find function definitions
            def visit_node(node):
                if isinstance(node, dict):
                    if node.get("type") == "FunctionDefinition":
                        # We extract the name or block context
                        # Since solidity_parser doesn't perfectly give substrings easily out of the box in all versions,
                        # we capture the start/end if available or simply treat the node as context
                        func_name = node.get("name", "fallback/receive")
                        chunks.append({
                            "type": "FunctionDefinition",
                            "name": func_name,
                            "ast_node": node
                        })
                    for key, value in node.items():
                        if isinstance(value, (dict, list)):
                            visit_node(value)
                elif isinstance(node, list):
                    for item in node:
                        if isinstance(item, (dict, list)):
                            visit_node(item)

            visit_node(source_unit)
            
            # Fallback: if we couldn't chunk effectively or loc info is missing, just return the whole contract as one chunk
            # for the MVP to prevent arbitrary mid-logic splits.
            if not chunks:
                chunks.append({"type": "Contract", "content": solidity_code})
                
        except Exception as e:
            logging.error(f"Error parsing solidity code: {e}")
            # Fallback to single chunk to never interrupt logic mid-way
            chunks.append({"type": "UnparsedCode", "content": solidity_code})
            
        return chunks

class LibrarianRAG:
    """
    Retrieves relevant exploit post-mortems and known vulnerable patterns.
    """
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_db = Chroma(
            collection_name="sentinel_knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_STORE_DIR)
        )
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Queries the knowledge base for top k similar documents.
        """
        results = self.vector_db.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in results])
        return context
