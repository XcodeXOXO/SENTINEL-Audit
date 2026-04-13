import os
import json
import logging
import re
from typing import Dict, Any, List
from solcx import compile_source, install_solc, get_installed_solc_versions, set_solc_version
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

class CriticVerifier:
    def __init__(self):
        # UPDATED: Switched to Llama 3.3 Versatile as Llama 3.0 is decommissioned
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0 # Forced 0.0 for deterministic verification
        )
        self.verify_prompt = PromptTemplate(
            input_variables=["contract_code", "expert_findings", "compilation_warnings"],
            template="""
            You are the Critic Agent. Your job is to verify findings and identify HALLUCINATIONS.
            If a finding is not directly supported by the code, it is a hallucination.

            Target Contract Code:
            {contract_code}

            solc Compilation Warnings/Errors:
            {compilation_warnings}

            Expert Findings:
            {expert_findings}

            Return your verification in JSON format ONLY. 

            Schema:
            {{
                "hallucination_detected": true,
                "verified_findings": [
                    {{
                        "title": "...",
                        "severity": "...",
                        "function_name": "...",
                        "description": "...",
                        "invariant_violated": "...",
                        "remediation": "...",
                        "critic_verification_notes": "Your exact reasoning against the AST/Code",
                        "is_hallucination": true
                    }}
                ],
                "confidence_score": 80
            }}
            """
        )
        
        self.default_solc = "0.8.20"
        try:
            installed = get_installed_solc_versions()
            if self.default_solc not in [str(v) for v in installed]:
                logging.info(f"Installing solc version {self.default_solc}...")
                install_solc(self.default_solc)
            set_solc_version(self.default_solc)
        except Exception as e:
            logging.error(f"Failed to initialize py-solc-x: {e}")

    def sanity_compile(self, contract_code: str) -> Dict[str, Any]:
        result = {"successful": False, "warnings": []}
        try:
            compile_source(contract_code, output_values=["ast", "bin"])
            result["successful"] = True
            result["warnings"].append("Compilation successful.")
        except Exception as e:
            result["warnings"].append(str(e))
        return result

    def verify(self, contract_code: str, expert_findings: Dict[str, Any]) -> Dict[str, Any]:
        compilation = self.sanity_compile(contract_code)
        
        chain = self.verify_prompt | self.llm
        try:
            response = chain.invoke({
                "contract_code": contract_code,
                "expert_findings": json.dumps(expert_findings, indent=2),
                "compilation_warnings": "\n".join(compilation["warnings"])
            })
            
            content = str(response.content)
            # Robust JSON cleaning for Groq responses
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            output = json.loads(content)
            output["compilation_successful"] = compilation["successful"]
            output["solc_warnings"] = "\n".join(compilation["warnings"])
            return output
        except Exception as e:
            logging.error(f"Critic error: {e}")
            return {
                "hallucination_detected": False,
                "verified_findings": expert_findings.get("findings", []),
                "confidence_score": 0,
                "compilation_successful": compilation["successful"],
                "solc_warnings": "\n".join(compilation["warnings"])
            }