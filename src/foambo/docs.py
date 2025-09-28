import difflib, textwrap
from typing import Dict, Any
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt

console = Console()

def get_content_and_category(entry) -> tuple[str, str|None]:
    if isinstance(entry, dict):
        content = entry.get("content", "")
        category = entry.get("category")
    else:
        content = entry
        category = None
    return content, category

def fuzzy_search(query: str, docs: Dict[str, Any], n: int = 5):
    """
    Fuzzy search over keys and values. Returns up to `n` best matching keys.
    Special query 'all' returns all keys.
    """
    query_lower = query.lower()
    if query_lower in ("all", "*"):
        return list(docs.keys())

    matches = []
    for k, v in docs.items():
        content, _ = get_content_and_category(v)
        k_lower = k.lower()
        v_lower = content.lower()
        if query_lower in k_lower or query_lower in v_lower:
            matches.append(k)
    if not matches:
        candidates = list(docs.keys()) + [get_content_and_category(v)[0] for v in docs.values()]
        raw_matches = difflib.get_close_matches(query, candidates, n=n, cutoff=0.1)
        matches = [k if k in docs else v for k, v in zip(raw_matches, raw_matches)]
    return matches[:n]

def run_docs_tui(docs: Dict[str, Any]):
    console.print("[bold green]FoamBO Configuration Docs Explorer[/bold green]")
    console.print("Type your query (e.g. 'param'), or ('exit', 'quit', or 'q' ) to quit.")
    console.print("For all configuration keys, user '*' or 'all' as your query\n")

    while True:
        try:
            query = prompt("🔍> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting...[/bold red]")
            break
        if query.strip().lower() in ("exit", "quit", "q"):
            break

        matches = fuzzy_search(query, docs, n=5)
        if not matches:
            console.print(f"[red]No matches found for '{query}'[/red]\n")
            continue

        for m in matches:
            entry = docs[m]
            content, category = get_content_and_category(entry)
            content = textwrap.dedent(content)
            panel = Panel(
                    Markdown(content),
                    title=f"[cyan]{m}[/cyan]",
                    subtitle=f"[magenta]{category}[/magenta]" if category else None,
                    expand=True,
                    border_style="green",
                    padding=(1,1),
                    title_align="left",
                    subtitle_align="right")
            console.print(panel)
            console.print()
