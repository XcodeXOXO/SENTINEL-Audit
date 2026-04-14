"""
sentinel_cli.py
═══════════════════════════════════════════════════════════════════════════════
Project Sentinel — Phase 4: Command Center (CLI Entry Point)
───────────────────────────────────────────────────────────────────────────────
Standalone CLI that loads the SentinelAuditor, runs a security audit on a
target Solidity file, and renders a premium terminal report using `rich`.

This file is intentionally decoupled from main.py's LangGraph pipeline.
It is a direct, single-model auditor — no Librarian RAG, no Critic loop —
making it ideal for rapid, standalone inference in a Colab notebook or GPU VM.

Usage
─────
  python sentinel_cli.py --contract path/to/MyContract.sol
  python sentinel_cli.py --contract path/to/MyContract.sol --weights ./Llama3_Sentinel_v1
  python sentinel_cli.py --contract path/to/MyContract.sol --save-report
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Rich imports for premium terminal output ──────────────────────────────────
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Project imports ───────────────────────────────────────────────────────────
from src.expert.sentinel_auditor import (
    ContextLimitExceededError,
    ModelNotLoadedError,
    SentinelAuditor,
    MAX_CONTRACT_TOKENS,
)

# ── Logging (stderr only — stdout is reserved for rich output) ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s │ %(name)s │ %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sentinel.cli")

# ── Rich console with custom Sentinel theme ───────────────────────────────────
SENTINEL_THEME = Theme({
    "banner":          "bold bright_cyan",
    "section.header":  "bold bright_white",
    "severity.critical": "bold bright_red",
    "severity.high":   "bold red",
    "severity.medium": "bold yellow",
    "severity.low":    "bold bright_blue",
    "severity.info":   "bold green",
    "label":           "dim white",
    "value":           "bright_white",
    "footer":          "dim cyan",
    "success":         "bold bright_green",
    "warning":         "bold yellow",
    "error":           "bold bright_red",
})

console = Console(theme=SENTINEL_THEME, highlight=False)

# ── Severity colour map for panels ───────────────────────────────────────────
_SEVERITY_STYLES: dict[str, str] = {
    "critical": "severity.critical",
    "high":     "severity.high",
    "medium":   "severity.medium",
    "low":      "severity.low",
    "info":     "severity.info",
}

_SEVERITY_ICONS: dict[str, str] = {
    "critical": "💀",
    "high":     "🔴",
    "medium":   "🟡",
    "low":      "🔵",
    "info":     "🟢",
}


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    """Render the Sentinel ASCII banner and version header."""
    banner_text = Text(justify="center")
    banner_text.append(
        "\n"
        "  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     \n"
        "  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     \n"
        "  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     \n"
        "  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     \n"
        "  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗ \n"
        "  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝ \n",
        style="bold bright_cyan",
    )
    banner_text.append(
        "        Autonomous Smart Contract Security Auditor  •  v1.0.0\n"
        "        Fine-tuned Llama-3-8B @ LoRA Rank 16  •  BCCC-VulSCs\n\n",
        style="dim cyan",
    )
    console.print(Panel(banner_text, border_style="bright_cyan", padding=(0, 2)))


# ── Contract Metadata Panel ───────────────────────────────────────────────────

def _print_contract_metadata(
    path: Path,
    token_count: Optional[int] = None,
    token_limit: int = MAX_CONTRACT_TOKENS,
) -> None:
    """Render a metadata panel for the target contract."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="label", width=22)
    table.add_column("Value", style="value")

    stat = path.stat()
    size_kb = stat.st_size / 1024

    table.add_row("📄  Target Contract", str(path.resolve()))
    table.add_row("📦  File Size", f"{size_kb:.2f} KB  ({stat.st_size:,} bytes)")
    table.add_row("🕐  Timestamp", datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    if token_count is not None:
        pct = (token_count / token_limit) * 100
        colour = "green" if pct < 80 else "yellow" if pct < 100 else "red"
        table.add_row(
            "🔢  Contract Tokens",
            f"[{colour}]{token_count:,}[/{colour}] / {token_limit:,}  "
            f"([{colour}]{pct:.1f}% of budget[/{colour}])",
        )

    console.print(
        Panel(table, title="[section.header]CONTRACT METADATA[/section.header]",
              border_style="bright_cyan", padding=(1, 2))
    )


# ── Findings Renderer ─────────────────────────────────────────────────────────

def _render_findings_from_text(raw_report: str, contract_name: str) -> None:
    """
    Parse and render the model's raw text output as a rich terminal report.

    The fine-tuned model returns a free-form Markdown-style text (not JSON).
    This function renders it faithfully, preserving the model's structure,
    while wrapping it in Sentinel's premium panel UI.

    Parameters
    ----------
    raw_report : str
        The clean string returned by `SentinelAuditor.audit_contract()`.
    contract_name : str
        Display name of the contract (used in section header).
    """
    console.print()
    console.print(Rule(
        title="[section.header]🛡  SENTINEL SECURITY FINDINGS[/section.header]",
        style="bright_cyan",
    ))
    console.print()

    if not raw_report.strip():
        console.print(
            Panel(
                "[warning]⚠  The model returned an empty response.[/warning]\n"
                "This may indicate an issue with the model weights or prompt format.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        return

    # ── Severity distribution summary ────────────────────────────────────────
    report_lower = raw_report.lower()
    sev_counts: dict[str, int] = {}
    for sev in ("critical", "high", "medium", "low"):
        count = report_lower.count(sev)
        if count:
            sev_counts[sev] = count

    if sev_counts:
        summary_table = Table(
            show_header=True,
            header_style="bold bright_white",
            box=box.SIMPLE_HEAVY,
            padding=(0, 3),
        )
        summary_table.add_column("Severity", style="bold")
        summary_table.add_column("Mentions", justify="center")

        for sev, count in sev_counts.items():
            style = _SEVERITY_STYLES.get(sev, "white")
            icon = _SEVERITY_ICONS.get(sev, "•")
            summary_table.add_row(
                f"[{style}]{icon}  {sev.upper()}[/{style}]",
                f"[{style}]{count}[/{style}]",
            )

        console.print(
            Panel(
                Align.center(summary_table),
                title="[section.header]SEVERITY SUMMARY[/section.header]",
                border_style="bright_cyan",
                padding=(1, 2),
            )
        )
        console.print()

    # ── Full model output (rendered as Markdown for structure) ────────────────
    console.print(
        Panel(
            Markdown(raw_report),
            title=f"[section.header]DETAILED FINDINGS  •  {contract_name.upper()}[/section.header]",
            border_style="bright_cyan",
            padding=(1, 3),
        )
    )


# ── Footer ────────────────────────────────────────────────────────────────────

def _print_footer(elapsed: float, output_path: Optional[Path] = None) -> None:
    """Render the session footer with runtime statistics."""
    console.print()
    console.print(Rule(style="bright_cyan"))

    stats_table = Table(show_header=False, box=None, padding=(0, 3))
    stats_table.add_column("Label", style="label")
    stats_table.add_column("Value", style="value")
    stats_table.add_row("⏱  Total Inference Time", f"[success]{elapsed:.2f}s[/success]")
    stats_table.add_row("🤖  Model", "Llama-3-8B-Instruct  +  Sentinel LoRA (Rank 16)")
    stats_table.add_row("🔬  Quantisation", "4-bit NF4  (Unsloth FastLanguageModel)")

    if output_path:
        stats_table.add_row("💾  Report Saved", str(output_path.resolve()))

    console.print(
        Panel(
            Align.center(stats_table),
            title="[section.header]SESSION STATISTICS[/section.header]",
            border_style="dim cyan",
            padding=(1, 2),
        )
    )
    console.print(
        Align.center(Text(
            "\nProject Sentinel  •  Lead AI Security Research Platform\n"
            "All findings are model-generated. Always verify with a human auditor.\n",
            style="footer",
        ))
    )


# ── Report Save ───────────────────────────────────────────────────────────────

def _save_report(
    raw_report: str,
    contract_path: Path,
    elapsed: float,
) -> Path:
    """
    Persist the raw model output as a Markdown report file.

    Parameters
    ----------
    raw_report : str
        Clean response from `SentinelAuditor.audit_contract()`.
    contract_path : Path
        Path to the audited `.sol` file (used to name the report).
    elapsed : float
        Inference wall-clock time in seconds.

    Returns
    -------
    Path
        Path to the saved report file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"sentinel_report_{contract_path.stem}_{timestamp}.md"
    report_path = Path("reports") / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# 🛡️ Sentinel Audit Report\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Contract** | `{contract_path.name}` |\n"
        f"| **Full Path** | `{contract_path.resolve()}` |\n"
        f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n"
        f"| **Model** | Llama-3-8B-Instruct + Sentinel LoRA (Rank 16) |\n"
        f"| **Quantisation** | 4-bit NF4 (Unsloth) |\n"
        f"| **Inference Time** | {elapsed:.2f}s |\n\n"
        f"---\n\n"
        f"## Findings\n\n"
    )

    report_path.write_text(header + raw_report, encoding="utf-8")
    return report_path


# ── Argument Parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description=(
            "Project Sentinel — Autonomous Smart Contract Security Auditor.\n"
            "Runs a LoRA fine-tuned Llama-3-8B model locally on your Solidity contract."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python sentinel_cli.py --contract contracts/Token.sol\n"
            "  python sentinel_cli.py --contract contracts/Vault.sol --save-report\n"
            "  python sentinel_cli.py --contract contracts/DEX.sol "
            "--weights ./Llama3_Sentinel_v1 --save-report\n"
        ),
    )
    parser.add_argument(
        "--contract",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the Solidity (.sol) file to audit.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./Llama3_Sentinel_v1",
        metavar="DIR",
        help=(
            "Path to the LoRA adapter weights directory. "
            "Default: ./Llama3_Sentinel_v1"
        ),
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=False,
        help="Save the findings as a Markdown file in the reports/ directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (INFO level from all modules).",
    )
    return parser


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI entry point for Project Sentinel.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on any handled error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Apply verbosity — quiet by default so rich output isn't polluted
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("sentinel").setLevel(logging.WARNING)

    # ── Validate inputs ───────────────────────────────────────────────────────
    contract_path = Path(args.contract)
    weights_dir   = Path(args.weights)

    if not contract_path.exists():
        console.print(
            Panel(
                f"[error]⛔  File not found:[/error]\n\n"
                f"  [value]{contract_path.resolve()}[/value]\n\n"
                f"Please check the path and try again.",
                title="[error]FILE NOT FOUND[/error]",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    if not contract_path.suffix == ".sol":
        console.print(
            Panel(
                f"[warning]⚠  Expected a .sol file, received:[/warning]\n\n"
                f"  [value]{contract_path.name}[/value]",
                title="[warning]INVALID FILE TYPE[/warning]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        return 1

    # ── Read contract ─────────────────────────────────────────────────────────
    contract_code = contract_path.read_text(encoding="utf-8")
    if not contract_code.strip():
        console.print("[error]⛔  The contract file is empty.[/error]")
        return 1

    # ── Banner ─────────────────────────────────────────────────────────────────
    _print_banner()

    # ── Load model and run audit ───────────────────────────────────────────────
    token_count_display: Optional[int] = None
    raw_report: str = ""
    t_start = time.perf_counter()

    try:
        with SentinelAuditor(weights_dir=weights_dir) as auditor:
            # Show token count in metadata panel
            token_count_display = auditor._count_contract_tokens(contract_code)
            _print_contract_metadata(
                contract_path,
                token_count=token_count_display,
                token_limit=MAX_CONTRACT_TOKENS,
            )

            console.print()
            console.print(
                Panel(
                    "[bright_cyan]🔄  Inference engine active. Analysing contract...[/bright_cyan]\n"
                    "[dim]This typically takes 15–60 seconds on a T4 GPU.[/dim]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            console.print()

            raw_report = auditor.audit_contract(contract_code)

    except ContextLimitExceededError as e:
        _print_contract_metadata(contract_path, token_limit=MAX_CONTRACT_TOKENS)
        console.print(
            Panel(
                str(e),
                title="[error]⛔  CONTEXT LIMIT EXCEEDED[/error]",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    except FileNotFoundError as e:
        console.print(
            Panel(
                str(e),
                title="[error]⛔  WEIGHT FILES NOT FOUND[/error]",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    except RuntimeError as e:
        console.print(
            Panel(
                str(e),
                title="[error]⛔  RUNTIME ERROR[/error]",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    except Exception as e:
        logger.exception("Unexpected error during audit")
        console.print(
            Panel(
                f"[error]An unexpected error occurred:[/error]\n\n{type(e).__name__}: {e}",
                title="[error]⛔  UNEXPECTED ERROR[/error]",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    elapsed = time.perf_counter() - t_start

    # ── Render findings ────────────────────────────────────────────────────────
    _render_findings_from_text(raw_report, contract_name=contract_path.stem)

    # ── Save report (optional) ─────────────────────────────────────────────────
    saved_path: Optional[Path] = None
    if args.save_report:
        saved_path = _save_report(raw_report, contract_path, elapsed)
        console.print(f"\n[success]✅  Report saved →[/success] {saved_path.resolve()}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    _print_footer(elapsed, output_path=saved_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
