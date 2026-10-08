"""Structure-aware Markdown splitting for knowledge ingestion."""

import re

from customer_service.knowledge_store import KnowledgeStore, NewChunk


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE = re.compile(r"[^。！？.!?\n]+[。！？.!?]?", re.S)


def _split_sentences(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    units = []
    for sentence in _SENTENCE.findall(text):
        if len(sentence) > max_chars:
            clauses = re.findall(r"[^，,；;]+[，,；;]?", sentence)
            units.extend(x.strip() for x in clauses if x.strip())
        elif sentence.strip():
            units.append(sentence.strip())
    result: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and sum(map(len, current)) + len(unit) > max_chars:
            result.append("".join(current))
            overlap: list[str] = []
            for old in reversed(current):
                if sum(map(len, overlap)) + len(old) > overlap_chars and overlap:
                    break
                overlap.insert(0, old)
            current = overlap
            if sum(map(len, current)) + len(unit) > max_chars:
                current = []
        if len(unit) > max_chars:
            # Keep an overlong sentence intact instead of leaving a fragment.
            if current:
                result.append("".join(current))
                current = []
            result.append(unit)
        else:
            current.append(unit)
    if current:
        result.append("".join(current))
    return result


def _split_table(lines: list[str], max_chars: int) -> list[str]:
    header = lines[:2]
    rows = lines[2:]
    if not rows:
        return ["\n".join(lines)]
    prefix = "\n".join(header)
    result, current = [], []
    for row in rows:
        candidate = "\n".join([prefix, *current, row])
        if current and len(candidate) > max_chars:
            result.append("\n".join([prefix, *current]))
            current = []
        current.append(row)
    if current:
        result.append("\n".join([prefix, *current]))
    return result


def parse_markdown(markdown: str, source: str, max_chars: int = 800, overlap_chars: int = 80, content_type: str = "policy") -> list[NewChunk]:
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Invalid split limits")
    heading_levels: dict[int, str] = {}
    sections: list[tuple[list[str], list[str]]] = []
    body: list[str] = []

    def flush():
        if body and "\n".join(body).strip():
            sections.append(([heading_levels[level] for level in sorted(heading_levels)], body.copy()))
        body.clear()

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_levels[level] = match.group(2).strip()
            for deeper in [key for key in heading_levels if key > level]:
                del heading_levels[deeper]
        else:
            body.append(line)
    flush()

    chunks: list[NewChunk] = []
    for path, lines in sections:
        category = " / ".join(path[:-1]) or "未分类"
        question = path[-1] if path else source
        section_path = " / ".join(path) if path else source
        parts: list[str] = []
        prose: list[str] = []

        def flush_prose():
            if prose:
                parts.extend(_split_sentences("\n".join(prose).strip(), max_chars, overlap_chars))
                prose.clear()

        i = 0
        while i < len(lines):
            if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|\s*$", lines[i + 1]):
                flush_prose()
                table = [lines[i], lines[i + 1]]
                i += 2
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    table.append(lines[i])
                    i += 1
                parts.extend(_split_table(table, max_chars))
                continue
            prose.append(lines[i])
            i += 1
        flush_prose()
        for index, part in enumerate(parts):
            chunks.append(NewChunk(category, question, part, f"{source}#{section_path}#{index}", section_path, content_type))
    return chunks


def retire_stale_chunks(store: KnowledgeStore, source: str, active_keys: set[str]) -> list[int]:
    return store.retire_stale(source, active_keys)
