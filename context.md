# 🏗️ PROJECT SENTINEL: ARCHITECTURE & STATE

## 1. The Vision
A high-precision, hybrid AI system designed to automate the identification and documentation of security vulnerabilities in blockchain smart contracts.

## 2. The Three-Pillar Architecture
1.  **The Librarian (RAG System):** The semantic knowledge base containing SWC Registry patterns, OpenZeppelin standards, and historic DeFi exploit post-mortems. Built with `LangChain` and `ChromaDB`.
2.  **The Expert (Inference Engine):** A LoRA fine-tuned LLM specialized in Solidity state transitions and vulnerability detection. Built using `Unsloth` on `BCCC-VulSCs` datasets.
3.  **The Critic (Verification Agent):** An autonomous loop that cross-references the Expert's findings against the code to filter out false positives and format the final Markdown report. Built via agentic prompt loops (Groq Llama 3).

## 3. Tech Stack & Environment
* **Language:** Python 3.10+
* **Orchestration:** LangChain / LangGraph
* **Vector Store:** ChromaDB (Local)
* **Embeddings:** Google `embedding-004` (via `langchain-google-genai`)
* **LLMs:** * Primary Brain: Gemini 3 Flash (via Google AI Studio)
    * Speed/Critic: Llama 3 (via Groq)
* **Key Libraries:** `beautifulsoup4`, `solidity-parser`, `pydantic`, `tiktoken`

## 4. File Structure (Target)
```text
/Audit-smart
│── .env                     # API Keys (Google, Groq)
│── requirements.txt         # Project dependencies
│── context.md               # You are here. Project state and blueprint.
│── gemini.md                # System prompt and AI behavioral rules.
│── /data
│   ├── /raw                 # Raw SWC/DeFi exploit markdown files
│   └── /fine_tuning         # JSONL files for LoRA training
│── /vector_store            # ChromaDB local persistence directory
│── /scripts                 # Internal operations and data ingestion
│── /templates               # Markdown templates for reports
│── /src
│   ├── /librarian           # RAG logic, semantic chunking, ChromaDB interface
│   ├── /expert              # Model inference and prompt engineering
│   └── /critic              # Verification loop, dynamic compilation
│── /tests
│   └── /vulnerable_contracts # Sample .sol files for testing the system
└── main.py                  # The stateful LangGraph orchestration pipeline