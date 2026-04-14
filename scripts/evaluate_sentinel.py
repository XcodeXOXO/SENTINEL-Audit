"""
evaluate_sentinel.py
═══════════════════════════════════════════════════════════════════════════════
Project Sentinel — Phase 4: Evaluation Harness ("The Stress Test")
───────────────────────────────────────────────────────────────────────────────
Automated validation suite that proves SentinelAuditor's efficacy across a
curated set of positive (vulnerable) and negative (secure) Solidity contracts.

Evaluation logic (per Upgrade #2 — Strict Keyword Evaluation):
  • POSITIVE tests: A result is True Positive only if the model output contains
    at least ONE keyword from the pre-defined EXPECTED_KEYWORDS map for that
    file. A "vulnerability mentioned" without matching keywords is a False
    Negative. This prevents gas optimisations from inflating TP counts.
  • NEGATIVE tests: Any vulnerability keyword in the output is a False Positive.

Outputs:
  • Live progress table in the terminal (rich).
  • `reports/evaluation_report_{timestamp}.md` — full Markdown report with
    per-contract findings and a summary accuracy matrix.

Usage
─────
  python scripts/evaluate_sentinel.py
  python scripts/evaluate_sentinel.py --weights ./Llama3_Sentinel_v1
  python scripts/evaluate_sentinel.py --contracts-dir tests/vulnerable_contracts
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Rich UI ───────────────────────────────────────────────────────────────────
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── Project imports ───────────────────────────────────────────────────────────
# Add project root to path so this script can be run from anywhere.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.expert.sentinel_auditor import (
    ContextLimitExceededError,
    ModelNotLoadedError,
    SentinelAuditor,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s │ %(name)s │ %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sentinel.eval")

console = Console(highlight=False)

# ═════════════════════════════════════════════════════════════════════════════
# KEYWORD MAP — The heart of Upgrade #2 (Strict Keyword Evaluation)
#
# Each entry maps a filename to a list of expected vulnerability keywords.
# The model output (lowercased) must contain at least ONE of these strings
# for the result to qualify as a True Positive.
#
# Keywords are deliberately broad enough to cover paraphrasing (e.g. both
# "reentrancy" and "re-entrancy") but specific enough to exclude gas tips.
# ═════════════════════════════════════════════════════════════════════════════
EXPECTED_KEYWORDS: Dict[str, List[str]] = {
    "reentrancy.sol": [
        "reentrancy",
        "re-entrancy",
        "re-entrance",
        "checks-effects-interactions",
        "checks effects interactions",
        "cei pattern",
        "state update",
        "external call before state",
        "withdraw before update",
        "recursive call",
        "fallback",
    ],
    "integer_overflow.sol": [
        "overflow",
        "underflow",
        "arithmetic",
        "integer overflow",
        "wrap around",
        "unchecked arithmetic",
        "swc-101",
        "safemath",
        "safe math",
        "uint256 max",
        "locktime",
        "lock time",
    ],
    "access_control.sol": [
        "access control",
        "access_control",
        "unprotected",
        "initializ",          # covers "initialize" and "initialization"
        "owner",
        "onlyowner",
        "only owner",
        "tx.origin",
        "txorigin",
        "swc-105",
        "swc-115",
        "phishing",
        "privilege escalation",
        "authorization",
        "authorisation",
    ],
    "flash_loan_attack.sol": [
        "flash loan",
        "flashloan",
        "oracle manipulation",
        "price manipulation",
        "spot price",
        "reserve ratio",
        "amm",
        "economic attack",
        "price oracle",
        "twap",
        "on-chain oracle",
        "manipulate",
        "single transaction",
    ],
}

# Negative test files: model output should NOT contain any of these keywords.
# If it does, it's counted as a False Positive.
NEGATIVE_TEST_FILES: List[str] = ["safe_vault.sol"]

VULNERABILITY_TRIGGER_WORDS: List[str] = [
    "vulnerability",
    "vulnerable",
    "attack",
    "exploit",
    "reentrancy",
    "overflow",
    "underflow",
    "access control",
    "unprotected",
    "flash loan",
    "oracle manipulation",
    "critical",
    "high severity",
    "medium severity",
]


# ═════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ContractResult:
    """Stores the outcome of auditing a single Solidity contract."""

    filename: str
    category: str              # "positive" | "negative"
    elapsed_seconds: float
    raw_output: str
    error: Optional[str]       # Non-None if auditing raised an exception
    outcome: str               # "TP" | "FP" | "TN" | "FN" | "ERROR" | "SKIP"
    matched_keywords: List[str] = field(default_factory=list)
    triggered_fp_words: List[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    """Aggregate metrics across all evaluated contracts."""

    true_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    errors: int = 0
    skipped: int = 0
    total_elapsed: float = 0.0
    results: List[ContractResult] = field(default_factory=list)

    @property
    def total_evaluated(self) -> int:
        return (
            self.true_positives + self.false_negatives
            + self.true_negatives + self.false_positives
        )

    @property
    def accuracy(self) -> float:
        """Fraction of evaluations where the model gave the correct answer."""
        total = self.total_evaluated
        if total == 0:
            return 0.0
        correct = self.true_positives + self.true_negatives
        return correct / total

    @property
    def precision(self) -> float:
        """TP / (TP + FP) — how often a flagged contract is truly vulnerable."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN) — how often the model catches a real vulnerability."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def avg_inference_time(self) -> float:
        """Mean seconds per contract (excluding errors and skips)."""
        evaluated = [r for r in self.results if r.outcome not in ("ERROR", "SKIP")]
        if not evaluated:
            return 0.0
        return sum(r.elapsed_seconds for r in evaluated) / len(evaluated)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation Logic
# ═════════════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """
    Orchestrates the full evaluation loop over a contracts directory.

    The harness:
      1. Discovers all .sol files in the target directory.
      2. Classifies each as positive (has EXPECTED_KEYWORDS entry) or negative.
      3. Loads the SentinelAuditor once for the entire batch (context manager).
      4. Audits each contract, applying strict keyword matching.
      5. Builds an EvaluationSummary and writes a Markdown report.

    Parameters
    ----------
    contracts_dir : Path
        Directory containing the .sol test files.
    weights_dir : Path
        Directory containing the LoRA adapter artifacts.
    output_dir : Path
        Directory where the Markdown report will be written.
    """

    def __init__(
        self,
        contracts_dir: Path,
        weights_dir: Path,
        output_dir: Path,
    ) -> None:
        self.contracts_dir = contracts_dir
        self.weights_dir = weights_dir
        self.output_dir = output_dir

    # ── Contract Discovery ────────────────────────────────────────────────────

    def _discover_contracts(self) -> List[Path]:
        """
        Find all .sol files in the contracts directory.

        Returns
        -------
        List[Path]
            Sorted list of .sol file paths (alphabetical for reproducibility).
        """
        files = sorted(self.contracts_dir.glob("*.sol"))
        if not files:
            raise FileNotFoundError(
                f"No .sol files found in: {self.contracts_dir}\n"
                f"Ensure the test contracts directory is populated."
            )
        return files

    # ── Keyword Evaluation ────────────────────────────────────────────────────

    def _evaluate_positive(
        self,
        filename: str,
        output_lower: str,
    ) -> tuple[str, List[str]]:
        """
        Evaluate a positive (known-vulnerable) contract result.

        A True Positive requires at least one EXPECTED_KEYWORDS match.
        Prevents gas-optimisation mentions from inflating TP counts (Upgrade #2).

        Parameters
        ----------
        filename : str
            Contract filename used to look up the keyword map.
        output_lower : str
            Model output, already lowercased.

        Returns
        -------
        tuple[str, List[str]]
            (outcome, matched_keywords): outcome is "TP" or "FN".
        """
        keywords = EXPECTED_KEYWORDS.get(filename, [])
        if not keywords:
            # No keyword map defined → can't properly evaluate → SKIP
            return "SKIP", []

        matched = [kw for kw in keywords if kw in output_lower]
        outcome = "TP" if matched else "FN"
        return outcome, matched

    def _evaluate_negative(
        self,
        output_lower: str,
    ) -> tuple[str, List[str]]:
        """
        Evaluate a negative (expected-secure) contract result.

        A False Positive occurs if the model flags ANY vulnerability keyword.

        Parameters
        ----------
        output_lower : str
            Model output, already lowercased.

        Returns
        -------
        tuple[str, List[str]]
            (outcome, triggered_words): outcome is "TN" or "FP".
        """
        triggered = [w for w in VULNERABILITY_TRIGGER_WORDS if w in output_lower]
        outcome = "FP" if triggered else "TN"
        return outcome, triggered

    # ── Single Contract Audit ─────────────────────────────────────────────────

    def _audit_one(
        self,
        auditor: SentinelAuditor,
        sol_path: Path,
    ) -> ContractResult:
        """
        Run the auditor on a single .sol file and return a structured result.

        Parameters
        ----------
        auditor : SentinelAuditor
            Pre-loaded auditor instance.
        sol_path : Path
            Path to the Solidity file.

        Returns
        -------
        ContractResult
            Full result object with outcome, timing, and matched keywords.
        """
        filename = sol_path.name
        is_negative = filename in NEGATIVE_TEST_FILES
        category = "negative" if is_negative else "positive"

        contract_code = sol_path.read_text(encoding="utf-8")

        t_start = time.perf_counter()
        try:
            raw_output = auditor.audit_contract(contract_code)
            elapsed = time.perf_counter() - t_start

        except ContextLimitExceededError as e:
            elapsed = time.perf_counter() - t_start
            return ContractResult(
                filename=filename,
                category=category,
                elapsed_seconds=elapsed,
                raw_output="",
                error=f"ContextLimitExceededError: {e.token_count} tokens > {e.token_limit} limit",
                outcome="SKIP",
            )

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.exception("Error auditing %s", filename)
            return ContractResult(
                filename=filename,
                category=category,
                elapsed_seconds=elapsed,
                raw_output="",
                error=f"{type(e).__name__}: {e}",
                outcome="ERROR",
            )

        output_lower = raw_output.lower()

        if is_negative:
            outcome, triggered = self._evaluate_negative(output_lower)
            return ContractResult(
                filename=filename,
                category=category,
                elapsed_seconds=elapsed,
                raw_output=raw_output,
                error=None,
                outcome=outcome,
                triggered_fp_words=triggered,
            )
        else:
            outcome, matched = self._evaluate_positive(filename, output_lower)
            return ContractResult(
                filename=filename,
                category=category,
                elapsed_seconds=elapsed,
                raw_output=raw_output,
                error=None,
                outcome=outcome,
                matched_keywords=matched,
            )

    # ── Full Evaluation Run ───────────────────────────────────────────────────

    def run(self) -> EvaluationSummary:
        """
        Execute the full evaluation loop.

        Loads the auditor once, audits all discovered contracts, and returns
        a populated EvaluationSummary.

        Returns
        -------
        EvaluationSummary
            Complete metrics and per-contract results.
        """
        contracts = self._discover_contracts()
        summary = EvaluationSummary()

        console.print()
        console.print(Rule(
            title="[bold bright_cyan]🔬  SENTINEL EVALUATION HARNESS[/bold bright_cyan]",
            style="bright_cyan",
        ))
        console.print(
            f"  [dim]Contracts directory :[/dim] [bright_white]{self.contracts_dir}[/bright_white]\n"
            f"  [dim]Weights directory   :[/dim] [bright_white]{self.weights_dir}[/bright_white]\n"
            f"  [dim]Contracts found     :[/dim] [bright_white]{len(contracts)}[/bright_white]"
        )
        console.print()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold bright_white]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        with SentinelAuditor(weights_dir=self.weights_dir) as auditor:
            with progress:
                task = progress.add_task(
                    "Auditing contracts...", total=len(contracts)
                )

                for sol_path in contracts:
                    progress.update(
                        task,
                        description=f"[bright_cyan]Auditing[/bright_cyan]  {sol_path.name:<35}",
                    )
                    result = self._audit_one(auditor, sol_path)
                    summary.results.append(result)
                    summary.total_elapsed += result.elapsed_seconds

                    # Tally outcome
                    match result.outcome:
                        case "TP":    summary.true_positives += 1
                        case "FN":    summary.false_negatives += 1
                        case "TN":    summary.true_negatives += 1
                        case "FP":    summary.false_positives += 1
                        case "ERROR": summary.errors += 1
                        case "SKIP":  summary.skipped += 1

                    progress.advance(task)

        return summary

    # ── Report Generation ─────────────────────────────────────────────────────

    def generate_report(self, summary: EvaluationSummary) -> Path:
        """
        Write the full evaluation report to a timestamped Markdown file.

        Parameters
        ----------
        summary : EvaluationSummary
            Populated summary from a completed `run()` call.

        Returns
        -------
        Path
            Path to the written report file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"evaluation_report_{timestamp}.md"

        lines: List[str] = []
        _w = lines.append  # shorthand writer

        # ── Header ────────────────────────────────────────────────────────────
        _w("# 🛡️ Project Sentinel — Evaluation Report\n")
        _w(f"| Field | Value |")
        _w(f"|---|---|")
        _w(f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
        _w(f"| **Model** | Llama-3-8B-Instruct + Sentinel LoRA (Rank 16) |")
        _w(f"| **Quantisation** | 4-bit NF4 (Unsloth FastLanguageModel) |")
        _w(f"| **Contracts Evaluated** | {summary.total_evaluated} |")
        _w(f"| **Contracts Skipped / Errored** | {summary.skipped} / {summary.errors} |")
        _w(f"| **Total Wall-Clock Time** | {summary.total_elapsed:.2f}s |")
        _w(f"| **Avg Inference Time** | {summary.avg_inference_time:.2f}s per contract |")
        _w("")

        # ── Accuracy Matrix ────────────────────────────────────────────────────
        _w("---\n")
        _w("## 📊 Accuracy Matrix\n")
        _w("| Metric | Value |")
        _w("|---|---|")
        _w(f"| ✅ True Positives  (TP) | **{summary.true_positives}** |")
        _w(f"| ❌ False Negatives (FN) | **{summary.false_negatives}** |")
        _w(f"| ✅ True Negatives  (TN) | **{summary.true_negatives}** |")
        _w(f"| ❗ False Positives (FP) | **{summary.false_positives}** |")
        _w(f"| **Accuracy** | **{summary.accuracy * 100:.1f}%** |")
        _w(f"| **Precision** | **{summary.precision * 100:.1f}%** |")
        _w(f"| **Recall** | **{summary.recall * 100:.1f}%** |")
        _w("")

        # ── Per-Contract Findings ──────────────────────────────────────────────
        _w("---\n")
        _w("## 📋 Per-Contract Results\n")

        _ICONS = {"TP": "✅", "TN": "✅", "FN": "🔴", "FP": "🔴", "ERROR": "⛔", "SKIP": "⚠️"}

        for result in summary.results:
            icon = _ICONS.get(result.outcome, "•")
            _w(f"### {icon}  `{result.filename}` — **{result.outcome}**\n")
            _w(f"| Field | Value |")
            _w(f"|---|---|")
            _w(f"| Category | {result.category.upper()} |")
            _w(f"| Outcome | {result.outcome} |")
            _w(f"| Inference Time | {result.elapsed_seconds:.2f}s |")

            if result.matched_keywords:
                _w(f"| Matched Keywords | `{'`, `'.join(result.matched_keywords)}` |")

            if result.triggered_fp_words:
                _w(f"| False Positive Triggers | `{'`, `'.join(result.triggered_fp_words)}` |")

            if result.error:
                _w(f"| Error | `{result.error}` |")

            _w("")

            if result.raw_output:
                _w("<details>")
                _w(f"<summary>Model Output (click to expand)</summary>\n")
                _w("```")
                _w(result.raw_output)
                _w("```")
                _w("</details>\n")

        # ── Footer ─────────────────────────────────────────────────────────────
        _w("---\n")
        _w(
            "> **Disclaimer**: All findings are model-generated and must be "
            "verified by a qualified human security auditor before acting on them.\n"
        )

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


# ═════════════════════════════════════════════════════════════════════════════
# Terminal Summary Renderer
# ═════════════════════════════════════════════════════════════════════════════

def _render_terminal_summary(summary: EvaluationSummary, report_path: Path) -> None:
    """Render the final evaluation summary to the terminal using rich."""

    _OUTCOME_STYLE: Dict[str, str] = {
        "TP": "bold bright_green",
        "TN": "bold bright_green",
        "FN": "bold red",
        "FP": "bold red",
        "ERROR": "bold bright_red",
        "SKIP": "bold yellow",
    }

    console.print()
    console.print(Rule(
        title="[bold bright_cyan]📊  EVALUATION RESULTS[/bold bright_cyan]",
        style="bright_cyan",
    ))
    console.print()

    # Per-contract table
    per_contract = Table(
        title="Per-Contract Results",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_white",
        padding=(0, 2),
    )
    per_contract.add_column("Contract", style="bright_white", no_wrap=True)
    per_contract.add_column("Category", justify="center")
    per_contract.add_column("Outcome", justify="center")
    per_contract.add_column("Time (s)", justify="right")
    per_contract.add_column("Matched / Triggered Keywords", overflow="fold")

    for r in summary.results:
        style = _OUTCOME_STYLE.get(r.outcome, "white")
        kw_display = (
            ", ".join(r.matched_keywords) if r.matched_keywords
            else ", ".join(r.triggered_fp_words) if r.triggered_fp_words
            else r.error or "—"
        )
        per_contract.add_row(
            r.filename,
            r.category.upper(),
            f"[{style}]{r.outcome}[/{style}]",
            f"{r.elapsed_seconds:.1f}",
            kw_display,
        )

    console.print(per_contract)
    console.print()

    # Summary metrics table
    metrics = Table(
        title="Aggregate Metrics",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_white",
        padding=(0, 3),
    )
    metrics.add_column("Metric", style="dim white")
    metrics.add_column("Value", style="bold bright_white", justify="right")

    metrics.add_row("True Positives",  f"[bright_green]{summary.true_positives}[/bright_green]")
    metrics.add_row("False Negatives", f"[red]{summary.false_negatives}[/red]")
    metrics.add_row("True Negatives",  f"[bright_green]{summary.true_negatives}[/bright_green]")
    metrics.add_row("False Positives", f"[red]{summary.false_positives}[/red]")
    metrics.add_row("Errors / Skipped", f"{summary.errors} / {summary.skipped}")
    metrics.add_row("─" * 20, "─" * 8)
    metrics.add_row("Accuracy",  f"[bold bright_cyan]{summary.accuracy * 100:.1f}%[/bold bright_cyan]")
    metrics.add_row("Precision", f"[bold bright_cyan]{summary.precision * 100:.1f}%[/bold bright_cyan]")
    metrics.add_row("Recall",    f"[bold bright_cyan]{summary.recall * 100:.1f}%[/bold bright_cyan]")
    metrics.add_row("Avg Inference Time", f"{summary.avg_inference_time:.2f}s")
    metrics.add_row("Total Wall-Clock", f"{summary.total_elapsed:.2f}s")

    console.print(metrics)
    console.print()
    console.print(
        Panel(
            f"[bold bright_green]✅  Report saved →[/bold bright_green] {report_path.resolve()}",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the evaluation harness."""
    parser = argparse.ArgumentParser(
        prog="evaluate_sentinel",
        description=(
            "Project Sentinel — Evaluation Harness.\n"
            "Runs the fine-tuned model over all .sol test contracts and "
            "generates a Markdown accuracy report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate_sentinel.py\n"
            "  python scripts/evaluate_sentinel.py --weights ./Llama3_Sentinel_v1\n"
            "  python scripts/evaluate_sentinel.py "
            "--contracts-dir custom/test/dir --output-dir custom/reports\n"
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(_PROJECT_ROOT / "Llama3_Sentinel_v1"),
        metavar="DIR",
        help="Path to LoRA adapter weights directory. Default: ./Llama3_Sentinel_v1",
    )
    parser.add_argument(
        "--contracts-dir",
        type=str,
        default=str(_PROJECT_ROOT / "tests" / "vulnerable_contracts"),
        metavar="DIR",
        help="Directory containing .sol test contracts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_PROJECT_ROOT / "reports"),
        metavar="DIR",
        help="Directory to write the Markdown evaluation report.",
    )
    return parser


def main() -> int:
    """
    Entry point for the evaluation harness.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on fatal error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    weights_dir    = Path(args.weights)
    contracts_dir  = Path(args.contracts_dir)
    output_dir     = Path(args.output_dir)

    harness = EvaluationHarness(
        contracts_dir=contracts_dir,
        weights_dir=weights_dir,
        output_dir=output_dir,
    )

    try:
        summary = harness.run()
    except FileNotFoundError as e:
        console.print(f"[bold red]⛔  {e}[/bold red]")
        return 1
    except RuntimeError as e:
        console.print(f"[bold red]⛔  Runtime error: {e}[/bold red]")
        return 1

    report_path = harness.generate_report(summary)
    _render_terminal_summary(summary, report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
