import difflib, textwrap
from typing import Dict, Any

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Collapsible, Static, Markdown
from textual.containers import VerticalScroll
from textual.binding import Binding


def get_content_and_category(entry) -> tuple[str, str | None]:
    if isinstance(entry, dict):
        content = entry.get("content", "")
        category = entry.get("category")
    else:
        content = entry
        category = None
    return content, category


def fuzzy_search(query: str, docs: Dict[str, Any], n: int = 20):
    """
    Fuzzy search over keys and values. Returns up to `n` best matching keys.
    Special query 'all' returns all keys.
    """
    def _norm(s: str) -> str:
        """Normalize for matching: lowercase, underscores as spaces."""
        return s.lower().replace("_", " ")

    query_norm = _norm(query)
    if query_norm in ("all", "*"):
        return list(docs.keys())

    matches = []
    for k, v in docs.items():
        content, _ = get_content_and_category(v)
        k_norm = _norm(k)
        v_norm = _norm(content)
        if query_norm in k_norm or query_norm in v_norm:
            matches.append(k)
    if not matches:
        candidates = [_norm(k) for k in docs.keys()] + [_norm(get_content_and_category(v)[0]) for v in docs.values()]
        raw_matches = difflib.get_close_matches(query_norm, candidates, n=n, cutoff=0.1)
        # Map normalized matches back to original keys
        norm_to_key = {_norm(k): k for k in docs.keys()}
        matches = [norm_to_key[m] for m in raw_matches if m in norm_to_key]
    return matches[:n]


class DocsApp(App):
    """FoamBO Configuration Docs Explorer."""

    CSS = """
    #search-input {
        dock: top;
        margin: 0 1;
    }
    #results {
        margin: 0 1;
    }
    Collapsible {
        margin: 0 0 1 0;
    }
    CollapsibleTitle {
        color: $accent;
    }
    .category-label {
        color: $text-muted;
        text-style: italic;
    }
    .no-results {
        color: $error;
        margin: 1 2;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
        Binding("ctrl+a", "expand_all", "Expand all"),
        Binding("ctrl+d", "collapse_all", "Collapse all"),
    ]

    def __init__(self, docs: Dict[str, Any]):
        super().__init__()
        self.docs = docs

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Input(
            placeholder="Search docs (e.g. 'metric', 'early_stopping', '*' for all)",
            id="search-input",
        )
        yield VerticalScroll(id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "FoamBO Docs"
        self.sub_title = "Type to search, Enter to filter, Ctrl+A/D expand/collapse, Esc to quit"
        self.query_one("#search-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        self._show_results(query)

    def _show_results(self, query: str) -> None:
        container = self.query_one("#results", VerticalScroll)
        container.remove_children()

        matches = fuzzy_search(query, self.docs)
        if not matches:
            container.mount(Static(f"No matches for '{query}'", classes="no-results"))
            return

        # If results fit in ~1 screen (<=3), expand them; otherwise collapse
        auto_expand = len(matches) <= 3

        for key in matches:
            entry = self.docs[key]
            content, category = get_content_and_category(entry)
            content = textwrap.dedent(content).strip()

            title = key
            if category:
                title = f"{key}  [{category}]"

            collapsible = Collapsible(
                Markdown(content),
                title=title,
                collapsed=not auto_expand,
            )
            container.mount(collapsible)

    def action_expand_all(self) -> None:
        for c in self.query(Collapsible):
            c.collapsed = False

    def action_collapse_all(self) -> None:
        for c in self.query(Collapsible):
            c.collapsed = True


def run_docs_tui(docs: Dict[str, Any]):
    app = DocsApp(docs)
    app.run()
