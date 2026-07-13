import re
from functools import lru_cache
from typing import Tuple, List, Set, Union
import nltk
from .downloads import check_nltk
from nltk.stem import WordNetLemmatizer
from .models import KBResult


# =============================================================================
# Cached model loaders
# =============================================================================

@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    check_nltk('wordnet')
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# =============================================================================
# Helper functions
# =============================================================================

def tokenize(text: str) -> List[str]:
    check_nltk('punkt_tab')
    return [t.lower() for t in nltk.word_tokenize(text)]


def remove_underscores(s: str) -> str:
    """Replace underscores with spaces."""
    return s.replace("_", " ")


def normalize_kb_args(kb: str) -> str:
    pred, a, b = parse_kb_injection(kb)
    a = remove_underscores(a)
    b = remove_underscores(b)
    return f"{pred}({a}, {b})"


def drop_leading_preposition(phrase: str) -> str:
    """
    If the phrase has exactly 3 tokens and the first is a preposition (IN),
    drop the first token. Otherwise, return the phrase unchanged.
    """
    check_nltk('punkt_tab')
    check_nltk('averaged_perceptron_tagger_eng')
    tokens = nltk.word_tokenize(phrase)

    if len(tokens) == 3:
        tagged = nltk.pos_tag(tokens)
        word, tag = tagged[0]

        if tag == "IN":   # preposition
            return " ".join(tokens[1:])

    return phrase


def parse_kb_injection(kb: str) -> Tuple[str, str, str]:
    """
    Parse predicate(arg1, arg2) into (predicate, arg1, arg2)
    Works for isa_wn, disj, and any other binary predicate.
    """
    m = re.fullmatch(r'\s*(\w+)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*', kb)
    if not m:
        raise ValueError(f"Invalid KB format: {kb}")
    return m.group(1), m.group(2), m.group(3)


def create_rel(pred: str, arg1: str, arg2: str) -> str:
    return f"{pred}({arg1}, {arg2})"


def normalize_premises(premises: Union[str, List[str]]) -> str:
    """
    Normalize premise input to a single concatenated string.
    Accepts either a single string or a list of strings.
    """
    if isinstance(premises, list):
        return " ".join(premises)
    return premises


ALLOWED_PREDICATES = {"isa_wn", "disj"}


# =============================================================================
# Phase 1: Candidate Generation
# =============================================================================

def generate_all_candidates(
    pred: str,
    arg1: str,
    arg2: str,
    original_text: str,
    post_process: bool = True
) -> List[KBResult]:
    """
    Generate all possible KB relation variants from a single parsed injection.
    
    Variants include:
    - Original (possibly post-processed)
    - Lemmatized version
    - Diff-only version (for multi-word args with same token count)
    - Swapped version (for both isa_wn and disj)
    
    Returns list of KBResult candidates (may contain duplicates, will be deduped later).
    """
    lemmatizer = get_lemmatizer()
    
    # Post-process if enabled
    provenance = "llm"
    original_a, original_b = arg1, arg2
    
    if post_process:
        arg1 = remove_underscores(arg1)
        arg2 = remove_underscores(arg2)
        arg1 = drop_leading_preposition(arg1)
        arg2 = drop_leading_preposition(arg2)
        
        if arg1 != original_a or arg2 != original_b:
            provenance = "post_process"
    
    # Build list of (pred, arg1, arg2, provenance) tuples
    variants: List[Tuple[str, str, str, str]] = []
    
    # 1. Original (post-processed)
    variants.append((pred, arg1, arg2, provenance))
    
    # 2. Lemmatized version
    lemma_a1 = " ".join([lemmatizer.lemmatize(t, pos='v') for t in tokenize(arg1)])
    lemma_a2 = " ".join([lemmatizer.lemmatize(t, pos='v') for t in tokenize(arg2)])
    
    if lemma_a1 != arg1.lower() or lemma_a2 != arg2.lower():
        variants.append((pred, lemma_a1, lemma_a2, "derived_lemma"))
    
    # 3. Diff-only version (extract differing words from multi-word args)
    t1, t2 = arg1.split(), arg2.split()
    if len(t1) == len(t2) and len(t1) > 1:
        diffs = [(w1, w2) for w1, w2 in zip(t1, t2) if w1.lower() != w2.lower()]
        if diffs:
            diff_a1 = " ".join(d[0] for d in diffs)
            diff_a2 = " ".join(d[1] for d in diffs)
            variants.append((pred, diff_a1, diff_a2, "derived_diff"))
    
    # Expand with swapped versions (for both isa_wn and disj)
    all_variants: List[Tuple[str, str, str, str]] = []
    for p, a1, a2, prov in variants:
        # Skip if args are identical
        if a1.lower() == a2.lower():
            continue
        
        all_variants.append((p, a1, a2, prov))
        # Add swapped version - combine provenance with "_swap" suffix
        swap_prov = f"{prov}_swap" if prov != "llm" else "derived_swap"
        all_variants.append((p, a2, a1, swap_prov))
    
    # Convert to KBResult objects
    results = []
    for p, a1, a2, prov in all_variants:
        rel_str = create_rel(p, a1, a2)
        results.append(KBResult(
            relation=rel_str,
            provenance=prov,
            original_text=original_text
        ))
    
    return results


# =============================================================================
# Phase 2: Filtering
# =============================================================================

def is_arg_in_text(
    arg_str: str,
    tokens: List[str],
) -> bool:
    """
    Check if an argument is present in the text using STRICT matching only.
    
    Uses:
    1) Exact token match
    2) Lemma match (both noun and verb forms)
    
    NO semantic similarity matching - this ensures only actual occurrences pass.
    """
    lemmatizer = get_lemmatizer()
    arg_str = (arg_str or "").strip().lower()

    if not arg_str:
        return False

    token_set = {t.lower() for t in tokens}
    token_lemmas = {lemmatizer.lemmatize(t.lower()) for t in tokens}
    token_lemmas_v = {lemmatizer.lemmatize(t.lower(), pos='v') for t in tokens}

    # For multi-word args like "blue shirt", check if ALL parts are found
    parts = tokenize(arg_str)
    if not parts:
        return False

    for part in parts:
        part_l = part.lower()
        part_lemma_n = lemmatizer.lemmatize(part_l)
        part_lemma_v = lemmatizer.lemmatize(part_l, pos='v')

        # Check if this part matches any token
        if part_l in token_set:
            continue
        if part_lemma_n in token_lemmas:
            continue
        if part_lemma_v in token_lemmas_v:
            continue
        
        # Part not found - arg is not in text
        return False

    return True


def is_arg_semantically_similar(
    arg_str: str,
    tokens: List[str],
    st_model=None,
    threshold: float = 0.60,
    use_token_level: bool = True
) -> bool:
    """
    Check if an argument is semantically similar to text tokens.
    
    This is a SOFTER check used for final validation, not for determining
    which sentence contains the argument.
    """
    lemmatizer = get_lemmatizer()
    text_str = " ".join(tokens).lower()
    arg_str = (arg_str or "").strip().lower()

    if not arg_str:
        return False

    # Fallback if no model
    if st_model is None:
        return arg_str in text_str

    # For multi-word args, check each part
    parts = tokenize(arg_str)
    if not parts:
        return False

    token_set = {t.lower() for t in tokens}
    token_lemmas = {lemmatizer.lemmatize(t.lower()) for t in tokens}
    token_lemmas_v = {lemmatizer.lemmatize(t.lower(), pos='v') for t in tokens}

    for part in parts:
        part_l = part.lower()

        # Fast checks first
        if part_l in token_set:
            continue
        if lemmatizer.lemmatize(part_l) in token_lemmas:
            continue
        if lemmatizer.lemmatize(part_l, pos='v') in token_lemmas_v:
            continue

        # Semantic soft match
        if use_token_level:
            cand_tokens = list({t.lower() for t in tokens if t})
            if not cand_tokens:
                return False

            emb = st_model.encode([part_l] + cand_tokens, convert_to_tensor=True)
            part_emb = emb[0]
            token_embs = emb[1:]

            from sentence_transformers import util
            sims = util.cos_sim(part_emb, token_embs)
            best_score = float(sims.max().item())

            if best_score < threshold:
                return False
        else:
            emb = st_model.encode([arg_str, text_str], convert_to_tensor=True)
            from sentence_transformers import util
            score = float(util.cos_sim(emb[0], emb[1]).item())
            if score < threshold:
                return False

    return True


def filter_candidates(
    candidates: List[KBResult],
    premise: Union[str, List[str]],
    hypothesis: str,
    st_model=None,
    threshold: float = 0.60,
    strict: bool = True,
    use_semantic: bool = False
) -> List[KBResult]:
    """
    Filter KB candidates by checking if arg1 is in premise and arg2 is in hypothesis.
    
    Args:
        candidates: List of KBResult candidates to filter
        premise: The premise sentence(s) - can be a string or list of strings
        hypothesis: The hypothesis sentence
        st_model: Sentence transformer model (loaded if None)
        threshold: Similarity threshold for semantic matching (if enabled)
        strict: If True, arg1 must be in premise and arg2 in hypothesis.
                If False, both args can be in either sentence.
        use_semantic: If True, use semantic similarity. If False (default),
                      use only exact/lemma matching for stricter filtering.
    
    Returns:
        Filtered and deduplicated list of KBResult objects
    """
    # Normalize premise to single string (supports list of sentences)
    premise_str = normalize_premises(premise)
    
    if st_model is None and use_semantic:
        st_model = get_st_model()
    
    prem_tokens = tokenize(premise_str)
    hyp_tokens = tokenize(hypothesis)
    all_tokens = prem_tokens + hyp_tokens
    
    filtered: List[KBResult] = []
    seen_relations: Set[str] = set()
    
    for candidate in candidates:
        try:
            pred, arg1, arg2 = parse_kb_injection(candidate.relation)
        except ValueError:
            continue
        
        # Choose which tokens to check against
        left_tokens = prem_tokens if strict else all_tokens
        right_tokens = hyp_tokens if strict else all_tokens
        
        # Check if both args are present in the correct sentences
        if use_semantic:
            # Use semantic similarity (softer check)
            arg1_ok = is_arg_semantically_similar(arg1, left_tokens, st_model, threshold)
            arg2_ok = is_arg_semantically_similar(arg2, right_tokens, st_model, threshold)
        else:
            # Use strict exact/lemma matching only (no semantic similarity)
            arg1_ok = is_arg_in_text(arg1, left_tokens)
            arg2_ok = is_arg_in_text(arg2, right_tokens)
        
        if arg1_ok and arg2_ok:
            # Deduplicate by relation string
            if candidate.relation not in seen_relations:
                seen_relations.add(candidate.relation)
                filtered.append(candidate)
    
    return filtered


# =============================================================================
# Main Pipeline
# =============================================================================

def pipeline_filter_kb_injections(
    kb_list: List[str],
    premise: Union[str, List[str]],
    hypothesis: str,
    st_model=None,
    post_process: bool = True,
    use_semantic: bool = False
) -> List[KBResult]:
    """
    Two-phase KB filtering pipeline:
    
    Phase 1 - Candidate Generation:
      1) Parse predicate(arg1, arg2)
      2) Keep only isa_wn / disj predicates
      3) Post-process: remove underscores, drop leading prepositions
      4) Generate variants: original, lemmatized, diff-only, swapped
    
    Phase 2 - Filtering:
      5) Keep only candidates where arg1 is in premise and arg2 is in hypothesis
         (strict exact/lemma matching by default, no semantic similarity)
      6) Deduplicate results
    
    Args:
        kb_list: List of raw KB injection strings from LLM
        premise: The premise sentence(s) - can be a string or list of strings
        hypothesis: The hypothesis sentence
        st_model: Optional sentence transformer model (only used if use_semantic=True)
        post_process: Whether to normalize args (remove underscores, prepositions)
        use_semantic: If True, use semantic similarity for matching (softer).
                      If False (default), use only exact/lemma matching (stricter).
    
    Returns:
        List of KBResult objects that passed all filters
    """
    
    # =========================================================================
    # Phase 1: Generate all candidates
    # =========================================================================
    all_candidates: List[KBResult] = []
    
    for inj in kb_list:
        try:
            pred, a, b = parse_kb_injection(inj)
        except ValueError:
            continue  # skip malformed entries
        
        # Filter by predicate type
        if pred not in ALLOWED_PREDICATES:
            continue
        
        # Generate all variants for this injection
        candidates = generate_all_candidates(
            pred, a, b,
            original_text=inj,
            post_process=post_process
        )
        all_candidates.extend(candidates)
    
    # =========================================================================
    # Phase 2: Filter candidates
    # =========================================================================
    return filter_candidates(
        all_candidates,
        premise,
        hypothesis,
        st_model=st_model,
        strict=True,
        use_semantic=use_semantic
    )


# =============================================================================
# Legacy function (for backwards compatibility)
# =============================================================================

def filter_kb_by_prem_hyp(
    kb_list,
    premise: Union[str, List[str]],
    hypothesis: str,
    st_model=None,
    threshold: float = 0.60,
    swap_args: bool = True,
    strict: bool = True,
    use_token_level: bool = True,
) -> List[KBResult]:
    """
    Legacy function - now just wraps filter_candidates.
    
    Filters KB injections using exact/lemma matching.
    """
    # Convert string inputs to KBResult if needed
    candidates: List[KBResult] = []
    for kb_item in kb_list:
        if isinstance(kb_item, str):
            candidates.append(KBResult(relation=kb_item, provenance="llm", original_text=kb_item))
        else:
            candidates.append(kb_item)
    
    return filter_candidates(
        candidates,
        premise,
        hypothesis,
        st_model=st_model,
        threshold=threshold,
        strict=strict,
        use_semantic=False  # Use strict matching
    )
