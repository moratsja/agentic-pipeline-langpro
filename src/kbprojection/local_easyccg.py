import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_FINITE_TAGS = {
    "MD",
    "VBD",
    "VBP",
    "VBZ",
}
_LEADING_DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
}


def easyccg_is_available(easyccg_dir: Path) -> bool:
    return (easyccg_dir / "easyccg.jar").exists() and (easyccg_dir / "model_rebank").exists()


_NOISE_PUNCTUATION_RE = re.compile(r"\s*([,:;])\s*")


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str):
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EasyCCG parsing requires the 'spacy' package. "
            "Install project dependencies with 'uv sync' or add spaCy to the active environment."
        ) from exc

    try:
        return spacy.load(model_name, disable=["parser"])
    except OSError as exc:
        raise RuntimeError(
            f"EasyCCG parsing requires the spaCy model '{model_name}'. "
            f"Install it with 'uv run python -m spacy download {model_name}'."
        ) from exc


def _tag_sentence(text: str, model_name: str) -> str:
    nlp = _load_spacy_model(model_name)
    doc = nlp((text or "").strip())
    tagged_tokens = []
    for token in doc:
        ner = token.ent_type_ or "O"
        tagged_tokens.append(f"{token.text}|{token.tag_}|{ner}")
    return " ".join(tagged_tokens)


def build_easyccg_parse_candidates(text: str, model_name: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return [stripped]

    candidates = [stripped]
    try:
        doc = _load_spacy_model(model_name)(stripped)
    except Exception:
        return candidates

    if not doc:
        return candidates

    has_finite_verb = any(token.tag_ in _FINITE_TAGS for token in doc)
    first = doc[0]
    first_lower = first.text.lower()
    if has_finite_verb or first_lower in _LEADING_DETERMINERS or not first.is_alpha or not first.is_lower:
        return candidates

    bare = stripped.rstrip(".!?")
    article_variant = f"A {bare}."
    capitalized_variant = bare[:1].upper() + bare[1:] + "."

    for candidate in (article_variant, capitalized_variant):
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def run_easyccg(sentence: str, easyccg_dir: Path, model_name: str, timeout_seconds: float) -> str:
    return _run_easyccg_with_retry(sentence, easyccg_dir, model_name, timeout_seconds)


def _normalize_easyccg_noise_sentence(text: str) -> str:
    normalized = _NOISE_PUNCTUATION_RE.sub(" ", (text or "").strip())
    return re.sub(r"\s+", " ", normalized).strip()


def _run_easyccg_once(sentence: str, easyccg_dir: Path, model_name: str, timeout_seconds: float) -> str:
    tagged_input = _tag_sentence(sentence, model_name)
    process = subprocess.run(
        [
            "java",
            "-jar",
            str((easyccg_dir / "easyccg.jar").resolve()),
            "--model",
            str((easyccg_dir / "model_rebank").resolve()),
            "--inputFormat",
            "POSandNERtagged",
            "--outputFormat",
            "prolog",
            "--nbest",
            "1",
        ],
        input=tagged_input,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        cwd=easyccg_dir.parents[2],
    )

    output = (process.stdout or "").strip()
    if process.returncode != 0 or not output:
        error_bits = [f"EasyCCG exited with code {process.returncode}"]
        if process.stderr:
            error_bits.append(process.stderr.strip())
        raise RuntimeError(" | ".join(error_bits))

    return output


def _run_easyccg_with_retry(sentence: str, easyccg_dir: Path, model_name: str, timeout_seconds: float) -> str:
    try:
        return _run_easyccg_once(sentence, easyccg_dir, model_name, timeout_seconds)
    except RuntimeError as exc:
        if "Unknown rule type: NOISE" not in str(exc):
            raise

        fallback_sentence = _normalize_easyccg_noise_sentence(sentence)
        if not fallback_sentence or fallback_sentence == (sentence or "").strip():
            raise

        return _run_easyccg_once(fallback_sentence, easyccg_dir, model_name, timeout_seconds)


def extract_arg_from_cat(cat_str: str) -> str:
    depth = 0
    main_op_idx = -1
    for idx, char in enumerate(cat_str):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in ["/", "\\"]:
            main_op_idx = idx
    if main_op_idx != -1:
        return cat_str[main_op_idx + 1 :].strip()
    return cat_str


def parse_easyccg_output(output: str):
    w_pattern = re.compile(
        r"w\((\d+),\s*(\d+),\s*'([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'\)\."
    )
    words: Dict[Tuple[int, int], Tuple[str, str, str, str]] = {}

    for line in output.split("\n"):
        match = w_pattern.match(line.strip())
        if not match:
            continue
        sen_id = int(match.group(1))
        word_id = int(match.group(2))
        words[(sen_id, word_id)] = (
            match.group(3),
            match.group(4),
            match.group(5),
            match.group(6),
        )

    class Node:
        pass

    class BA(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class FA(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class LF(Node):
        def __init__(self, sen_id, word_id, cat):
            self.sen_id = sen_id
            self.word_id = word_id
            self.cat = cat

    class RP(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class LP(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class BX(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class FC(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class CONJ(Node):
        def __init__(self, *args):
            if len(args) == 3:
                self.cat = args[0]
                self.arg = extract_arg_from_cat(args[0])
                self.left = args[1]
                self.right = args[2]
            else:
                self.cat = args[0]
                self.arg = args[1]
                self.left = args[2]
                self.right = args[3]

    class GFC(Node):
        def __init__(self, *args):
            self.cat = args[0]
            if len(args) == 3:
                self.degree = None
                self.left = args[1]
                self.right = args[2]
            else:
                self.degree = args[1]
                self.left = args[2]
                self.right = args[3]

    class GBX(Node):
        def __init__(self, *args):
            self.cat = args[0]
            if len(args) == 3:
                self.degree = None
                self.left = args[1]
                self.right = args[2]
            else:
                self.degree = args[1]
                self.left = args[2]
                self.right = args[3]

    class TR(Node):
        def __init__(self, *args):
            self.cat = args[0]
            if len(args) == 2:
                self.info = None
                self.child = args[1]
            else:
                self.info = args[1]
                self.child = args[2]

    class BC(Node):
        def __init__(self, cat, left, right):
            self.cat = cat
            self.left = left
            self.right = right

    class LTC(Node):
        def __init__(self, cat, info, child):
            self.cat = cat
            self.info = info
            self.child = child

    class RTC(Node):
        def __init__(self, cat, child, info):
            self.cat = cat
            self.child = child
            self.info = info

    class LX(Node):
        def __init__(self, cat, subcat, child):
            self.cat = cat
            self.subcat = subcat
            self.child = child

    class LEX(Node):
        def __init__(self, *args):
            if len(args) == 3:
                self.subcat = args[0]
                self.cat = args[1]
                self.child = args[2]
            else:
                self.cat = args[0]
                self.child = args[1]
                self.subcat = getattr(self.child, "cat", "unk")

    env = {
        "ccg": lambda idx, term: (idx, term),
        "ba": BA,
        "fa": FA,
        "lf": LF,
        "rp": RP,
        "lp": LP,
        "bx": BX,
        "fc": FC,
        "conj": CONJ,
        "gfc": GFC,
        "gbx": GBX,
        "gfxc": GFC,
        "gbxc": GBX,
        "tr": TR,
        "bc": BC,
        "ltc": LTC,
        "rtc": RTC,
        "lx": LX,
        "lex": LEX,
    }

    trees = {}
    buffer = ""
    for raw_line in output.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("w(") or line.startswith("%"):
            continue
        buffer += line
        if not buffer.endswith(")."):
            continue
        eval_buffer = buffer.rstrip(".").replace("\\", "\\\\")
        tree_id, tree = eval(eval_buffer, {"__builtins__": {}}, env)
        trees[tree_id] = tree
        buffer = ""

    return trees, words


def convert_cat(cat_str: str) -> str:
    value = cat_str.lower().replace("[", ":").replace("]", "")
    return "','" if value == "," else value


def _quote_prolog_atom(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def term_to_prolog(node, words: Dict[Tuple[int, int], Tuple[str, str, str, str]]) -> str:
    type_name = type(node).__name__
    if type_name == "LF":
        word, lemma, pos, ner = words.get((node.sen_id, node.word_id), ("unk", "unk", "UNK", "O"))
        return (
            f"t({convert_cat(node.cat)}, {_quote_prolog_atom(word)}, {_quote_prolog_atom(lemma.lower())}, "
            f"{_quote_prolog_atom(pos)}, {_quote_prolog_atom(ner)}, 'O')"
        )

    cat = convert_cat(node.cat)
    if type_name in {"BA", "FA", "RP", "LP", "BX", "FC", "BC"}:
        functor = type_name.lower()
        return f"{functor}({cat},\n  {term_to_prolog(node.left, words)},\n  {term_to_prolog(node.right, words)})"
    if type_name == "LX":
        return f"lx({cat}, {convert_cat(node.subcat)},\n  {term_to_prolog(node.child, words)})"
    if type_name == "LEX":
        return f"lx({cat}, {convert_cat(node.subcat)},\n  {term_to_prolog(node.child, words)})"
    if type_name == "CONJ":
        return (
            f"conj({cat}, {convert_cat(node.arg)},\n  {term_to_prolog(node.left, words)},\n"
            f"  {term_to_prolog(node.right, words)})"
        )
    if type_name in {"GFC", "GBX"}:
        functor = type_name.lower()
        if node.degree is not None:
            return (
                f"{functor}({cat}, {node.degree},\n  {term_to_prolog(node.left, words)},\n"
                f"  {term_to_prolog(node.right, words)})"
            )
        return f"{functor}({cat},\n  {term_to_prolog(node.left, words)},\n  {term_to_prolog(node.right, words)})"
    if type_name == "TR":
        if node.info is not None:
            return f"tr({cat}, {node.info},\n  {term_to_prolog(node.child, words)})"
        return f"tr({cat},\n  {term_to_prolog(node.child, words)})"
    if type_name == "LTC":
        return f"ltc({cat}, {node.info},\n  {term_to_prolog(node.child, words)})"
    if type_name == "RTC":
        return f"rtc({cat},\n  {term_to_prolog(node.child, words)},\n  {node.info})"
    raise ValueError(f"Unsupported EasyCCG node type: {type_name}")


def process_single_output(easyccg_output: str) -> Optional[str]:
    trees, words = parse_easyccg_output(easyccg_output)
    if not trees:
        return None
    original_id = next(iter(trees))
    return term_to_prolog(trees[original_id], words)
