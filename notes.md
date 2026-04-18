# Project Sentinel: Complete Architecture & Concepts Notes

## Dedicated Deep Dive: RAG vs. Fine-Tuning

**Retrieval-Augmented Generation (RAG)**
- **Explanation**: RAG dynamically enhances an LLM's prompt by retrieving relevant facts, context, or examples from an external database at runtime. It grounds the LLM in truth without modifying its weights.
- **Why it was used here (Librarian)**: RAG provides the dynamic context of evolving exploit strategies and specific past vulnerabilities without needing to retrain the model continuously.
- **Codebase Example**: `src/librarian/rag.py` (Lines 83-85)
  ```python
  class LibrarianRAG:
      def __init__(self):
          self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
          self.vector_db = Chroma(...)
          
      def retrieve_context(self, query: str, k: int = 3) -> str:
          results = self.vector_db.similarity_search(query, k=k)
          context = "\n".join([doc.page_content for doc in results])
          return context
  ```
  *(This initializes a vector store and fetches the most similar historical exploit docs based on the queried contract code/text).*

**Fine-Tuning (LoRA / Unsloth)**
- **Explanation**: Fine-tuning adjusts the actual internal matrices (weights) of a pre-trained model so it deeply ingrains specific domain knowledge, stylistic formatting, and task reasoning natively.
- **Why it was used here (Expert)**: To embed deep Solidity semantics and the rigid auditing format (SWC Identification, Invariant Analysis, Remediation) into Llama-3-8B so it operates correctly out-of-the-box as an auditor, minimizing output hallucinations.
- **Codebase Example**: `Sentinel_Expert_Trainer.ipynb`
  ```python
  model = FastLanguageModel.get_peft_model(
      model,
      r = 16,
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
      lora_alpha = 16,
      lora_dropout = 0,
      bias = "none",
      use_gradient_checkpointing = "unsloth",
      random_state = 3407,
  )
  ```
  *(This snippet injects Low-Rank Adaptation matrices into Llama-3's attention and feed-forward layers to efficiently fine-tune it on Google Colab T4 without modifying all 8B base parameters).*

---

## Phase 1: Setup & CLI

### Concept/Library Name: `argparse`
- **Why it was used here**: For the `scripts/ingest_data.py` and `main.py` entry points to handle command-line flags and parameters (e.g., `--contract`, `--github`).
- **Detailed Explanation**: Built-in Python library that parses `sys.argv`. It handles type checking, default values, and automatically generates help/usage messages.
- **Codebase Example**: `scripts/ingest_data.py` (Lines 661-667)
  ```python
  def main() -> None:
      parser = argparse.ArgumentParser(
          description="Project Sentinel — Phase 2 ETL: build instruction fine-tuning dataset."
      )
      parser.add_argument("--github", action="store_true", help="Ingest SWC Registry via GitHub API")
      args = parser.parse_args()
  ```
  *(Initializes the CLI parser for the ingestion script to dynamically decide which dataset sources to scrape based on user input).*
- **Alternative Tech Stack**: `Click` or `Typer`. Not chosen because `argparse` is built-in to Python Core and dependencies were kept strictly minimal where possible.

### Concept/Library Name: `python-dotenv`
- **Why it was used here**: To load secret API keys like `GITHUB_TOKEN` and `HF_TOKEN` securely without permanently hardcoding them into scripts.
- **Detailed Explanation**: Reads key-value pairs from a `.env` file and seamlessly injects them into the machine's `os.environ` array.
- **Codebase Example**: `scripts/ingest_data.py` (Line 44)
  ```python
  from dotenv import load_dotenv
  load_dotenv(override=True) 
  GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
  ```
  *(Loads credentials required to bypass rate limits when pulling dataset information).*
- **Alternative Tech Stack**: Setting `os.environ` parameters manually in bash profiles. Not chosen because local `.env` files are highly developer-friendly and prevent setup friction.

---

## Phase 2: Data Ingestion

### Concept/Library Name: `requests`
- **Why it was used here**: To interface securely with the GitHub REST API and HuggingFace Datasets API to scrape training data.
- **Detailed Explanation**: A synchronous HTTP library for Python used to execute GET requests, manage headers (like Bearer tokens), and process complex JSON responses.
- **Codebase Example**: `scripts/ingest_data.py` (Lines 160-182)
  ```python
  def _github_get(url: str, params: Optional[dict] = None) -> dict | list:
      resp = requests.get(url, headers=_github_headers(), params=params, timeout=30)
      ...
      resp.raise_for_status()
      return resp.json()
  ```
  *(Performs resilient HTTP GET requests while gracefully handling rate-limits via sleep).*
- **Alternative Tech Stack**: `aiohttp` or `httpx`. Not chosen because synchronous execution via `requests` is simple, sufficient for this data volume, and avoids messy `async/await` scaffolding.

### Concept/Library Name: Semantic Chunking (`solidity-parser`)
- **Why it was used here**: To chunk incoming `.sol` files into analyzable segments while abiding strictly by the rule: "NEVER split code arbitrarily."
- **Detailed Explanation**: Parses Solidity language into an Abstract Syntax Tree (AST), preventing chunks from slicing functions algorithmically by character count. It isolates complete `FunctionDefinition` nodes.
- **Codebase Example**: `src/librarian/rag.py` (Lines 33-43)
  ```python
  def visit_node(node):
      if isinstance(node, dict):
          if node.get("type") == "FunctionDefinition":
              func_name = node.get("name", "fallback/receive")
              chunks.append({
                  "type": "FunctionDefinition",
                  "name": func_name,
                  "ast_node": node
              })
  ```
  *(Recursively traverses the AST to cleanly slice out self-contained functions).*
- **Alternative Tech Stack**: Langchain's `RecursiveCharacterTextSplitter`. Not chosen because a naive text splitter doesn't understand Solidity bracing arrays and frequently clips logic midway, breaking the analysis.

### Concept/Library Name: Idempotent Hashing (`hashlib`)
- **Why it was used here**: To natively gracefully resume broken ETL pipelines and prevent duplicate entries in the fine-tuning dataset.
- **Detailed Explanation**: Generates SHA-256 hashes of the raw string to uniquely map and deduplicate them logically in a JSON index file.
- **Codebase Example**: `scripts/ingest_data.py` (Lines 131-133)
  ```python
  def _sha256(text: str) -> str:
      """Return the SHA-256 hex digest of a UTF-8 string."""
      return hashlib.sha256(text.encode("utf-8")).hexdigest()
  ```
  *(Normalizes code items into a fingerprint to prevent duplicating records).*
- **Alternative Tech Stack**: UUID generation. Not chosen because randomly assigning UUIDs does not solve dataset deduplication across arbitrary pipeline runs.

---

## Phase 3: Fine-Tuning & Model Prep

### Concept/Library Name: `unsloth` & `peft`
- **Why it was used here**: To train the Llama-3-8B expert auditor extremely fast on a free cloud GPU (Colab T4) in 4-bit precision without throwing Out-of-Memory (OOM) errors.
- **Detailed Explanation**: 
  - `unsloth` implements highly optimized custom Triton kernels to speed up LLM backpropagation and memory swapping.
  - `peft` (Parameter-Efficient Fine-Tuning) applies LoRA matrices representing task instructions alongside the otherwise frozen 8B model. 
- **Codebase Example**: `Sentinel_Expert_Trainer.ipynb`
  ```python
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = "unsloth/llama-3-8b-bnb-4bit",
      max_seq_length = 2048,
      load_in_4bit = True,
  )
  ```
  *(Boots the massive 8B parameter model quantified forcefully into 4-bit mode using Unsloth's optimized pipeline to squeeze into 16GB VRAM)*.
- **Alternative Tech Stack**: Standard `transformers` with basic `bitsandbytes`. Not chosen because standard setups are consistently ~2-3x slower and use drastically more VRAM than Unsloth's patched framework.

### Concept/Library Name: `trl` (SFTTrainer)
- **Why it was used here**: To cleanly bootstrap the Supervised Instruction Fine Tuning loop and orchestrate dataset alignment.
- **Detailed Explanation**: The SFTTrainer bridges HuggingFace dependencies. It natively packs instructions, calculates causal generation loss, computes gradients globally, and saves adapter artifacts. 
- **Codebase Example**: `Sentinel_Expert_Trainer.ipynb`
  ```python
  trainer = SFTTrainer(
      model = model,
      tokenizer = tokenizer,
      train_dataset = dataset,
      args = SFTConfig(packing = True, max_seq_length = 2048...)
  )
  ```
  *(Kicks off training with memory-efficient `packing` which condenses shorter Alpaca inputs to exactly fit the sequence limit block).*
- **Alternative Tech Stack**: Raw PyTorch Training Loop. Not chosen because writing standard backpropagation logic for mixed-precision models demands unnecessary boilerplate.

---

## Phase 4: Agentic Workflow & Evaluation

### Concept/Library Name: `langgraph`
- **Why it was used here**: To physically orchestrate the requested "Three-Pillar Architecture" (`Librarian`, `Expert`, `Critic`) into a resilient, cyclical AI system.
- **Detailed Explanation**: Builds computational graphs (Nodes & Edges) over LLM tasks. It facilitates true autonomous recursion loops via `conditional_edge` pathways (e.g., allowing the Critic to forcibly reroute flow back up to the Expert upon identifying hallucinative data).
- **Codebase Example**: `main.py` (Line 75-96)
  ```python
  workflow = StateGraph(AuditState)
  workflow.add_node("librarian", librarian_node)
  workflow.add_node("expert", expert_node)
  workflow.add_node("critic", critic_node)
  workflow.add_node("re_evaluate", re_evaluation_node)
  
  workflow.add_conditional_edges(
      "critic",
      conditional_edge,
      {
          "re_evaluate": "re_evaluate",
          "end": END
      }
  )
  ```
  *(Declares state nodes and explicitly builds the Critic's recursive rejection path).*
- **Alternative Tech Stack**: Normal `LangChain` chains / AutoGen. LangChain does not natively cycle states dynamically backwards. AutoGen lacks explicit static typing controls needed for deterministic EVM flows.

### Concept/Library Name: `langchain-chroma` & `langchain-google-genai`
- **Why it was used here**: Provides the localized Vector DB and text embedding engine for the Librarian Agent.
- **Detailed Explanation**: ChromaDB saves vectors onto local disk. Google's GenAI seamlessly calculates the embeddings for solidity contract syntax mapping.
- **Codebase Example**: `src/librarian/rag.py` (Lines 71-77)
  ```python
  self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
  self.vector_db = Chroma(
      collection_name="sentinel_knowledge_base",
      embedding_function=self.embeddings,
      persist_directory=str(VECTOR_STORE_DIR)
  )
  ```
  *(Maps the Google embedding endpoint aggressively to a standard local `SQLite` Chroma wrapper)*
- **Alternative Tech Stack**: Pinecone or Qdrant. Not chosen out of strict adherence to the "Zero-Cost Stack Compliance" directive.

### Concept/Library Name: `rich`
- **Why it was used here**: Used to render a visually striking and structured output UI directly in the console for auditors using the evaluation metrics.
- **Detailed Explanation**: Overrides standard `print`. Maps ASCII styling efficiently into organized summary tables, error bars, panels, and colorful log feeds.
- **Codebase Example**: *Implicit CLI execution outputs located in* `sentinel_colab.ipynb` where standard CLI commands trigger output formatted beautifully using rich components.
- **Alternative Tech Stack**: Default Python `print` and logging. Not chosen because Sentinel mandates a highly polished, premium-feeling security tool, not raw system text logs.

### Concept/Library Name: `jinja2`
- **Why it was used here**: For dynamically publishing the final, organized markdown security `report.md` when the graph reaches endpoint resolution.
- **Detailed Explanation**: Ingests pure Python dictionaries spanning the finalized `AuditState` and safely mounts them inside HTML/Markdown architectural scaffold templates.
- **Codebase Example**: `main.py` (Line 98-100)
  ```python
  def render_report(state: AuditState, output_path: str = "report.md"):
      env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
      template = env.get_template('report_template.md')
      ...
      rendered = template.render(context)
  ```
  *(Injects Critic findings gracefully into a template document without mixing frontend logic with backend execution models).*
- **Alternative Tech Stack**: Python `f-strings`. Not chosen because injecting 10-page multiline code strings alongside formatting headers convolutes the logic files greatly. Templates isolate the UI generation.
