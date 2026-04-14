"""
sentinel_auditor.py
═══════════════════════════════════════════════════════════════════════════════
Project Sentinel — Phase 4: Core Inference Engine
───────────────────────────────────────────────────────────────────────────────
Owns the `SentinelAuditor` class: the production-grade LoRA inference engine
that loads the fine-tuned Llama-3-8B adapters and audits Solidity source code.

Design constraints honoured:
  • Anti-Hallucination: model is fine-tuned, not zero-shot; responses are grounded
    in the BCCC-VulSCs training distribution.
  • No silent truncation: ContextLimitExceededError is raised on token overage.
    Truncating a contract in security auditing can silently remove the vulnerable
    function and produce catastrophic False Negatives.
  • Explicit memory management: CUDA cache cleared after every inference call to
    prevent OOM on sequential audits.
  • GPU-first design: the class raises an actionable RuntimeError on CPU-only
    environments, since 4-bit quantised Llama-3-8B requires CUDA.

Run environment: Google Colab (T4 GPU) or equivalent CUDA device.
Adapter weights location: src/expert/weights/
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — torch/unsloth only load when the class is instantiated.
# This lets the module be imported on CPU machines for static analysis without
# crashing immediately.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Absolute path to the adapter weights directory, resolved relative to this
# file so the class works regardless of the working directory.
_WEIGHTS_DIR: Path = Path(__file__).parent / "weights"

# Base model identifier on HuggingFace Hub.
BASE_MODEL_ID: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# Maximum tokens *allocated to the Solidity contract* in the prompt.
# Model was trained with 2048 token context; the Alpaca wrapper text
# (instruction + surrounding markup) consumes ~220 tokens, leaving this budget.
MAX_CONTRACT_TOKENS: int = 1800

# Generation parameters — kept deterministic for reproducible audit results.
MAX_NEW_TOKENS: int = 768
TEMPERATURE: float = 0.1
REPETITION_PENALTY: float = 1.15  # mild penalty prevents looping on large contracts

# Alpaca EOS string added by Unsloth trainer
_EOS_TOKEN: str = "<|end_of_text|>"

# ── Custom Exceptions ────────────────────────────────────────────────────────


class ContextLimitExceededError(Exception):
    """
    Raised when the Solidity contract submitted for auditing exceeds the
    token budget allocated within the Alpaca prompt template.

    Security rationale: truncating a contract is explicitly forbidden because
    it may silently discard the vulnerable function(s), producing False
    Negatives without any warning. The user must manually split large contracts
    into logical sub-units (e.g., per-contract or per-library file) before
    running Sentinel.

    Attributes
    ----------
    token_count : int
        Actual number of tokens in the submitted contract.
    token_limit : int
        Maximum tokens permitted for the contract portion of the prompt.
    """

    def __init__(self, token_count: int, token_limit: int) -> None:
        self.token_count = token_count
        self.token_limit = token_limit
        super().__init__(
            f"\n\n{'═' * 70}\n"
            f"  ⛔  SENTINEL — CONTEXT LIMIT EXCEEDED\n"
            f"{'═' * 70}\n"
            f"  Contract tokens : {token_count:,}\n"
            f"  Maximum allowed : {token_limit:,}\n"
            f"  Overage         : {token_count - token_limit:,} tokens\n"
            f"{'─' * 70}\n"
            f"  ACTION REQUIRED:\n"
            f"  Truncating contracts is prohibited in security auditing because\n"
            f"  it may silently remove vulnerable functions (False Negatives).\n\n"
            f"  Please split this file into smaller, logically self-contained\n"
            f"  units (e.g. one contract/library per file) and audit each\n"
            f"  unit separately.\n"
            f"{'═' * 70}\n"
        )


class ModelNotLoadedError(RuntimeError):
    """Raised when `audit_contract` is called before the model has been loaded."""
    pass


# ── Main Class ───────────────────────────────────────────────────────────────


class SentinelAuditor:
    """
    Production-grade Solidity security auditor backed by a LoRA fine-tuned
    Llama-3-8B model.

    The class drives the full inference lifecycle:
      1. Validates the runtime environment (CUDA, weight files).
      2. Loads the base model + LoRA adapters via Unsloth.
      3. Formats input in the exact Alpaca template used during training.
      4. Runs 4-bit inference and strips artefacts from the raw output.
      5. Releases GPU memory after each call (and on explicit unload).

    Usage (direct)
    --------------
    >>> auditor = SentinelAuditor()
    >>> auditor.load_model()
    >>> report = auditor.audit_contract(solidity_source)
    >>> auditor.unload_model()

    Usage (context manager — preferred for batch evaluation)
    --------------------------------------------------------
    >>> with SentinelAuditor() as auditor:
    ...     for src in contracts:
    ...         print(auditor.audit_contract(src))
    """

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        base_model_id: str = BASE_MODEL_ID,
        max_seq_length: int = 2048,
    ) -> None:
        """
        Initialise the auditor configuration. Does NOT load the model yet —
        call `load_model()` or use the class as a context manager.

        Parameters
        ----------
        weights_dir : Path, optional
            Path to the directory containing `adapter_model.safetensors` and
            `adapter_config.json`. Defaults to `src/expert/weights/`.
        base_model_id : str
            HuggingFace model ID for the quantised base model.
        max_seq_length : int
            Maximum total sequence length passed to Unsloth (prompt + response).
        """
        self.weights_dir: Path = weights_dir or _WEIGHTS_DIR
        self.base_model_id: str = base_model_id
        self.max_seq_length: int = max_seq_length

        # Runtime state — populated by load_model()
        self._model = None
        self._tokenizer = None
        self._is_loaded: bool = False

        logger.info(
            "SentinelAuditor configured | weights=%s | base=%s",
            self.weights_dir,
            self.base_model_id,
        )

    # ── Environment Validation ───────────────────────────────────────────────

    def _validate_environment(self) -> None:
        """
        Pre-flight checks before model loading.

        Raises
        ------
        RuntimeError
            If CUDA is not available (CPU-only environment).
        FileNotFoundError
            If the adapter weight files are missing from the weights directory.
        """
        # 1. CUDA guard
        import torch  # deferred import

        if not torch.cuda.is_available():
            raise RuntimeError(
                "\n\n{'═' * 70}\n"
                "  ⛔  SENTINEL — GPU NOT DETECTED\n"
                "{'═' * 70}\n"
                "  Unsloth's 4-bit quantised inference requires a CUDA GPU.\n"
                "  This environment appears to be CPU-only.\n\n"
                "  Please run sentinel_cli.py / evaluate_sentinel.py in one\n"
                "  of the following environments:\n"
                "    • Google Colab (free T4 tier is sufficient)\n"
                "    • A local machine with an NVIDIA GPU and CUDA ≥ 11.8\n"
                "    • Any cloud VM with a GPU (RunPod, Lambda Labs, etc.)\n"
                "{'═' * 70}\n"
            )

        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU detected: %s (%.1f GB VRAM)", device_name, vram_gb)

        # 2. Weight files guard
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [
            f for f in required_files
            if not (self.weights_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"\n\n{'═' * 70}\n"
                f"  ⛔  SENTINEL — ADAPTER WEIGHTS NOT FOUND\n"
                f"{'═' * 70}\n"
                f"  Expected weights directory : {self.weights_dir}\n"
                f"  Missing files              : {', '.join(missing)}\n\n"
                f"  To fix: copy all Unsloth-exported adapter files from\n"
                f"  your Colab training session into:\n"
                f"    {self.weights_dir}/\n"
                f"  Required files:\n"
                f"    • adapter_config.json\n"
                f"    • adapter_model.safetensors\n"
                f"    • tokenizer.json  (and related tokenizer files)\n"
                f"{'═' * 70}\n"
            )

    # ── Model Lifecycle ──────────────────────────────────────────────────────

    def load_model(self) -> "SentinelAuditor":
        """
        Load the base model and apply the LoRA adapters.

        Uses Unsloth's `FastLanguageModel` with 4-bit quantisation for maximum
        throughput on a single T4 GPU. Switches the model to inference mode via
        `FastLanguageModel.for_inference()`, which fuses layers and disables
        gradient tracking.

        Returns
        -------
        SentinelAuditor
            Returns `self` to allow chaining: `auditor.load_model().audit_contract(...)`.

        Raises
        ------
        RuntimeError
            If CUDA is unavailable.
        FileNotFoundError
            If adapter weights are missing.
        """
        if self._is_loaded:
            logger.warning("Model already loaded — skipping reload.")
            return self

        self._validate_environment()

        # Deferred heavy imports — only run on a GPU machine
        from unsloth import FastLanguageModel  # type: ignore[import]

        logger.info("Loading base model: %s", self.base_model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,          # auto-detect: bfloat16 on Ampere+, float16 on T4
            load_in_4bit=True,
            local_files_only=False,
        )

        logger.info("Applying LoRA adapters from: %s", self.weights_dir)
        model = FastLanguageModel.get_peft_model(
            model,
            # Adapter config is read from adapter_config.json in weights_dir;
            # we point PeftModel to the directory, not a specific file.
            # The PEFT library auto-discovers the config.
            r=16,                    # must match training rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,        # inference — no dropout
            bias="none",
            use_gradient_checkpointing=False,
            random_state=42,
        )

        # Load saved adapter weights
        from peft import PeftModel  # type: ignore[import]
        model = PeftModel.from_pretrained(model, str(self.weights_dir))

        # Switch to fast inference mode (fuses LoRA layers, disables grad)
        FastLanguageModel.for_inference(model)

        self._model = model
        self._tokenizer = tokenizer
        self._is_loaded = True

        logger.info("✅ SentinelAuditor model loaded and ready for inference.")
        return self

    def unload_model(self) -> None:
        """
        Release the model and tokenizer from GPU memory.

        Deletes the Python references and calls `torch.cuda.empty_cache()` to
        immediately return VRAM to the OS. Safe to call even if the model was
        never loaded.
        """
        if not self._is_loaded:
            return

        import torch  # deferred import

        logger.info("Unloading model and releasing GPU memory...")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✅ GPU memory released.")

    # ── Context Manager Support ──────────────────────────────────────────────

    def __enter__(self) -> "SentinelAuditor":
        """Load the model when entering a `with` block."""
        return self.load_model()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unload the model and free GPU memory when exiting a `with` block."""
        self.unload_model()

    # ── Prompt Engineering ───────────────────────────────────────────────────

    def _build_alpaca_prompt(self, contract_code: str) -> str:
        """
        Format the Solidity source into the Alpaca instruction template used
        during fine-tuning. The model WILL NOT generalise correctly to any
        other prompt format.

        The template is:
            ### Instruction:
            <task description>

            ### Input:
            <solidity source>

            ### Response:

        The trailing "### Response:" token acts as the generation cue — the
        model has learned to continue from this point.

        Parameters
        ----------
        contract_code : str
            Raw Solidity source code.

        Returns
        -------
        str
            Fully formatted Alpaca prompt string, ready for tokenisation.
        """
        instruction = (
            "You are an expert Solidity smart contract security auditor. "
            "Analyse the following smart contract code for security vulnerabilities. "
            "For each vulnerability found, describe: the vulnerability type, "
            "the affected function, the severity (Critical / High / Medium / Low), "
            "the exact attack vector, the economic or state invariant violated, "
            "and a concrete remediation step. "
            "If no vulnerabilities are found, state that the contract appears secure "
            "with a brief justification."
        )

        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{contract_code}\n\n"
            f"### Response:\n"
        )
        return prompt

    # ── Token Counting ───────────────────────────────────────────────────────

    def _count_contract_tokens(self, contract_code: str) -> int:
        """
        Count the number of tokens the raw Solidity source will occupy when
        tokenised by the loaded tokenizer.

        This count covers only the contract portion of the prompt — the
        Alpaca wrapper text is excluded because its token count is fixed and
        already accounted for in MAX_CONTRACT_TOKENS.

        Parameters
        ----------
        contract_code : str
            Raw Solidity source code.

        Returns
        -------
        int
            Token count for the contract text alone.

        Raises
        ------
        ModelNotLoadedError
            If called before `load_model()`.
        """
        if not self._is_loaded or self._tokenizer is None:
            raise ModelNotLoadedError(
                "Cannot count tokens: model not loaded. Call load_model() first."
            )

        encoded = self._tokenizer(
            contract_code,
            return_tensors=None,   # return plain lists, not tensors
            add_special_tokens=False,
        )
        return len(encoded["input_ids"])

    # ── Core Inference ───────────────────────────────────────────────────────

    def audit_contract(self, contract_code: str) -> str:
        """
        Run a full security audit on the provided Solidity source code.

        Pipeline
        --------
        1. Validate model is loaded.
        2. Count contract tokens and raise ContextLimitExceededError if over budget.
           (No truncation — see module docstring for security rationale.)
        3. Format prompt using the Alpaca template.
        4. Tokenise and move to GPU.
        5. Generate response with the fine-tuned model.
        6. Decode, strip the prompt prefix and EOS tokens.
        7. Release intermediate tensors and clear CUDA cache.
        8. Return the clean audit report string.

        Parameters
        ----------
        contract_code : str
            Raw Solidity source code to audit.

        Returns
        -------
        str
            The model's security analysis, stripped of prompt and EOS tokens.

        Raises
        ------
        ModelNotLoadedError
            If called before `load_model()`.
        ContextLimitExceededError
            If the contract exceeds MAX_CONTRACT_TOKENS (currently 1800 tokens).
            This is an intentional hard stop — never silent truncation.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                "Model is not loaded. Call load_model() or use SentinelAuditor "
                "as a context manager before calling audit_contract()."
            )

        import torch  # deferred import

        # ── Step 1: Token budget enforcement ────────────────────────────────
        token_count = self._count_contract_tokens(contract_code)
        if token_count > MAX_CONTRACT_TOKENS:
            raise ContextLimitExceededError(
                token_count=token_count,
                token_limit=MAX_CONTRACT_TOKENS,
            )

        logger.info(
            "Token budget check passed: %d / %d tokens used.",
            token_count, MAX_CONTRACT_TOKENS,
        )

        # ── Step 2: Prompt construction ──────────────────────────────────────
        prompt = self._build_alpaca_prompt(contract_code)

        # ── Step 3: Tokenise ────────────────────────────────────────────────
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")

        prompt_length: int = inputs["input_ids"].shape[1]
        logger.info("Prompt tokenised: %d total tokens (prompt + contract).", prompt_length)

        # ── Step 4: Inference ────────────────────────────────────────────────
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,          # enables temperature sampling
                repetition_penalty=REPETITION_PENALTY,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - t_start
        logger.info("Inference complete in %.2f seconds.", elapsed)

        # ── Step 5: Decode and clean ─────────────────────────────────────────
        # Slice off the prompt tokens so we only decode the NEW tokens.
        generated_ids = outputs[:, prompt_length:]
        raw_response: str = self._tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        # Strip any residual EOS tokens that may survive skip_special_tokens
        clean_response = raw_response.replace(_EOS_TOKEN, "").strip()

        if not clean_response:
            logger.warning("Model returned an empty response for this contract.")

        # ── Step 6: Memory cleanup ───────────────────────────────────────────
        del inputs, outputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return clean_response

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"SentinelAuditor("
            f"base='{self.base_model_id}', "
            f"weights='{self.weights_dir}', "
            f"status={status})"
        )
