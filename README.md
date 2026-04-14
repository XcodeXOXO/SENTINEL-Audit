# 🛡️ Project Sentinel: Autonomous Smart Contract Auditor

```text
               ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗                             
               ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║                             
               ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║                             
               ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║                             
               ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗                        
               ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝                        
               Autonomous Smart Contract Security Auditor • v1.0.0
               Fine-tuned Llama-3-8B @ LoRA Rank 16 • BCCC-VulSCs
```

---

## 📖 Introduction

### **The Vision**
**Project Sentinel** is a high-precision, AI-driven security platform designed to automate the detection of complex vulnerabilities in Solidity smart contracts. It acts as an autonomous "Security Researcher," identifying exploits before they hit the mainnet.

### **Technical Architecture**
Built on a fine-tuned **Llama-3-8B** backbone, Sentinel utilizes a **Three-Pillar Agent Architecture** (Librarian → Expert → Critic). The model was fine-tuned using **LoRA (Low-Rank Adaptation)** on the **BCCC-VulSCs** dataset, consisting of 1,717 curated vulnerability samples. By leveraging **Unsloth** for 4-bit quantization, Sentinel delivers 2x faster inference and reduced VRAM overhead on T4 GPUs.

---

## 🎯 Project Scope

* **Pattern Recognition**: Detects classic and complex vulnerabilities including Reentrancy, Arithmetic Overflows (SWC-101), and Phishing vectors.
* **Semantic Analysis**: Goes beyond regex-based tools by understanding the **Economic Invariants** of DeFi protocols.
* **Critic Verification**: Every finding is cross-checked by a Critic agent to minimize false positives and ensure remediation accuracy.
* **Production CLI**: Designed for direct integration into developer workflows and CI/CD pipelines.

---

## 💻 Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Core LLM** | Meta Llama-3-8B | Reasoning & code analysis engine |
| **Optimization** | Unsloth / LoRA | Memory-efficient fine-tuning and inference |
| **Agents** | LangGraph | Orchestrating the Librarian, Expert, and Critic agents |
| **Data Source** | BCCC-VulSCs | Training on 1,717 real-world smart contract bugs |
| **Terminal UI** | Rich / Tabulate | Professional-grade CLI visualization |
| **Execution** | Antigravity / Colab | Remote T4 GPU orchestration |

---

## 🚀 How to Run

### **Option A: Local Terminal Execution**
Use this for direct interaction within your local IDE environment.
1.  **Activate Environment**:
    ```bash
    source venv/bin/activate  # Windows: .\venv\Scripts\activate
    ```
2.  **Audit a Contract**:
    ```bash
    python sentinel_cli.py --contract tests/vulnerable_contracts/reentrancy.sol --weights /path/to/weights --save-report
    ```

### **Option B: Antigravity / Colab Extension (Cloud GPU)**
Use this for heavy inference if local hardware lacks a CUDA-capable GPU.
1.  Open `sentinel_colab.ipynb` in the **Antigravity IDE**.
2.  Run the **Setup Cell** to mount Google Drive and install the Unsloth engine.
3.  Execute the **Audit Cell** or prompt the Agent: 
    > *"Audit flash_loan_attack.sol using Sentinel CLI and show the Rich report."*

---

## 🚨 Sample Finding Output
*Generated via Project Sentinel Critic Agent*

### **Reentrancy Vulnerability**
* **Severity**: 🚨 High
* **Description**: The `withdraw` function performs a `msg.sender.call` before updating the user balance.
* **Economic Invariant Violated**: A user's total withdrawals must not exceed their total deposits.
* **Remediation**: Implement the **Checks-Effects-Interactions (CEI)** pattern; update `balances[msg.sender] = 0` before the transfer call.

---

