import os
import json
import logging
import re
from typing import Dict, Any, List, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

class ExpertInference:
    def __init__(self):
        # Using Gemini 3 Flash Preview (2026 Standard)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            temperature=0.1 
        )
        
        self.analysis_prompt = PromptTemplate(
            input_variables=["contract_code", "rag_context"],
            template="""
            You are the Expert Solidity Auditor. Provide highly rigorous analysis prioritizing economic invariants and access controls.

            RAG Reference Context:
            {rag_context}

            Target Contract Code:
            {contract_code}

            Return your findings strictly in JSON format matching this schema:
            {{
                "findings": [
                    {{
                        "title": "Short title",
                        "severity": "High/Medium/Low",
                        "function_name": "functionName",
                        "description": "Detailed explanation",
                        "invariant_violated": "Economic or State invariant violation",
                        "remediation": "How to fix"
                    }}
                ]
            }}
            """
        )

    def analyze_contract(self, contract_code: str, rag_context: str) -> Dict[str, Any]:
        chain = self.analysis_prompt | self.llm
        try:
            response = chain.invoke({
                "contract_code": contract_code,
                "rag_context": rag_context
            })
            
            content = response.content
            
            # FIX: If content is a list (common in 2026 multi-part responses), join it into a string
            if isinstance(content, list):
                content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
            
            # Now content is guaranteed to be a string for regex
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            return json.loads(content)
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return {"findings": []}