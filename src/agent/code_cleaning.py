import ast
import re
from typing import Optional


CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*(.*?)```",
    flags=re.DOTALL | re.IGNORECASE
)

INCOMPLETE_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*(.*)",
    flags=re.DOTALL | re.IGNORECASE
)


def remove_reasoning_blocks(text: str) -> str:
    patterns = [
        r"<think>.*?</think>",
        r"<reasoning>.*?</reasoning>",
        r"<analysis>.*?</analysis>",
        r"\[thinking\].*?\[/thinking\]",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


def extract_code_fence(text: str) -> Optional[str]:
    """
    Extract Python code from markdown code fences.

    Supports:
    ```python
    code
    ```

    Also supports incomplete fences:
    ```python
    code
    """
    blocks = CODE_FENCE_PATTERN.findall(text)

    if not blocks:
        incomplete = INCOMPLETE_CODE_FENCE_PATTERN.search(text)
        if incomplete:
            return incomplete.group(1).strip()
        return None

    def score_block(block: str) -> int:
        score = 0
        keywords = [
            "import ",
            "from ",
            "def ",
            "class ",
            "print(",
            "if __name__",
            "for ",
            "while ",
            "return ",
            "=",
        ]

        for keyword in keywords:
            if keyword in block:
                score += 1

        return score

    cleaned_blocks = [block.strip() for block in blocks if block.strip()]

    if not cleaned_blocks:
        return None

    return max(cleaned_blocks, key=score_block)


def strip_leading_text(text: str) -> str:
    """
    Remove natural language before the first likely Python line.
    """
    lines = text.splitlines()

    python_start_patterns = (
        "import ",
        "from ",
        "def ",
        "class ",
        "@",
        "if __name__",
        "try:",
        "with ",
        "for ",
        "while ",
    )

    assignment_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=")

    start_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(python_start_patterns):
            start_idx = i
            break

        if assignment_pattern.match(stripped):
            start_idx = i
            break

    if start_idx is None:
        return text.strip()

    # Important: lines[start_idx:] keeps all remaining lines.
    # lines[start_idx] would return one string, and "\n".join(...) would split it into characters.
    return "\n".join(lines[start_idx:]).strip()


def strip_trailing_text_by_ast(text: str) -> str:
    """
    Remove natural language after code by keeping the longest prefix
    that parses as valid Python.
    """
    text = text.strip()
    lines = text.splitlines()

    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    best = None

    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()

        if not candidate:
            continue

        try:
            ast.parse(candidate)
            best = candidate
            break
        except SyntaxError:
            continue

    return best if best is not None else text


def remove_markdown_artifacts(text: str) -> str:
    cleaned = []

    bad_prefixes = (
        "here is",
        "here's",
        "solution:",
        "solution code:",
        "python code:",
        "explanation:",
        "approach:",
        "###",
        "```",
    )

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            cleaned.append(line)
            continue

        if lower.startswith(bad_prefixes):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def extract_python_code(raw: str, debug: bool = False) -> str:
    text = raw.strip()

    if debug:
        print("first:", "=" * 20)
        print(text)
        print("=" * 20)

    text = remove_reasoning_blocks(text)

    if debug:
        print("second:", "=" * 20)
        print(text)
        print("=" * 20)

    fenced_code = extract_code_fence(text)

    if debug:
        print("fenced:", "=" * 20)
        print(fenced_code)
        print("=" * 20)

    if fenced_code is not None:
        text = fenced_code
    else:
        text = strip_leading_text(text)

    text = remove_markdown_artifacts(text)

    if debug:
        print("third:", "=" * 20)
        print(text)
        print("=" * 20)

    text = strip_trailing_text_by_ast(text)

    if debug:
        print("last:", "=" * 20)
        print(text)
        print("=" * 20)

    return text.strip()


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return False