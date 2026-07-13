import json
import hashlib
import asyncio
import time
from itertools import product
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import tempfile
import re
import httpx
from nltk import Tree, TreePrettyPrinter

from .async_runtime import AsyncRunContext, resolve_async_run_context
from .langpro_cache import InMemoryLangProCache, LangProCacheBackend, SQLiteLangProCache
from .local_easyccg import (
    build_easyccg_parse_candidates,
    easyccg_is_available,
    process_single_output,
    _normalize_easyccg_noise_sentence,
    _tag_sentence,
)
from .models import LangProResult, NLILabel
from .settings import (
    DEFAULT_LANGPRO_ENDPOINT,
    format_local_langpro_missing_error,
    get_langpro_settings,
)


INFIX_F = {':', '~>', '@', ',', '/', '\\'}
TYPE_F = {'~>'}
CAT_F = {'/', '\\'}
CATY_F = CAT_F | TYPE_F


# ------------------------------------------------------------
# Abstract base classes that cannot be instantiated but reused
# to define subclasses
class PrologTerm(ABC):
    """Base of everything as everything is a prolog term"""
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class CCGCat(ABC):
    """Base of CCG categories"""
    pass

class CaTy(ABC):
    """Base of functional types and categories"""
    pass

class TreeLike(ABC):
    """Base of tree-like structures such as LLFs, CCG derivations, proofs"""
    pass

# ------------------------------------------------------------
# Base class for all Prolog term types

# superclass for atoms, integers, floats
class Atomic(PrologTerm):
    def __init__(self, value):
        self.value = value

    def __str__(self) -> str:
        return f"{self.value}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Atomic) and self.value == other.value

# Represents a Prolog atom (subclass of atomic)
class Atom(Atomic):
    def __init__(self, value: str):
        if not isinstance(value, str):
            cname = self.__class__.__name__
            raise ValueError(f"{cname} expected arg of type str. Found {type(value)}.")
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

# Represents a Prolog integer (subclass of atomic)
class Integer(Atomic):
    def __init__(self, value: int):
        if not isinstance(value, int):
            cname = self.__class__.__name__
            raise ValueError(f"{cname} expected arg of type int. Found {type(value)}.")
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

# Represents a Prolog float (subclass of atomic)
class Float(PrologTerm):
    def __init__(self, value: float):
        if not isinstance(value, float):
            cname = self.__class__.__name__
            raise ValueError(f"{cname} expected arg of type float. Found {type(value)}.")
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

# Represents a Prolog variable (it is not atomic)
class Var(PrologTerm):
    def __init__(self, value: str):
        if not isinstance(value, str):
            cname = self.__class__.__name__
            raise ValueError(f"{cname} expected arg of type str. Found {type(value)}.")
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __eq__(self, _: object) -> bool:
        NotImplementedError("Prolog var matching is tricky and not supported here.")

# Represents a Prolog compound term (e.g., father(john, X))
class Compound(PrologTerm):
    def __init__(self, f: str, args: List[Any]):
        self.f = f
        self.args = [ Compound(a["functor"], a["args"]) \
                     if isinstance(a, dict) and "functor" in a else a \
                     for a in args ]
        self.nargs = len(args)

    def __str__(self) -> str:
        if self.f in INFIX_F:
            if len(self.args) != 2:
                raise ValueError(f"Arg num for {self.f} expected 2, but got {self.args}")
            ws = ' ' if self.f == ',' else ''
            return f"({self.args[0]}{ws}{self.f}{ws}{self.args[1]})"
        else:
            return f"{self.f}({', '.join(str(arg) for arg in self.args)})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}([{self.f}], {', '.join(repr(arg) for arg in self.args)})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Compound) and
            self.f == other.f and
            self.args == other.args
        )
    
    # def split(self, sep=None, maxsplit=-1):
    #     """Split the string representation"""
    #     return str(self).split(sep, maxsplit)

    def __len__(self) -> int:
        return len(str(self))

# ------------------------------------------------------------
# Linguistic/semantic framework/theory-specific classes

class AtomCaTy(CaTy):
    """Atomic Category/Types"""
    def __init__(self, atom: str):
        self.value = atom
        if ":" in atom:
            self.main, self.feat = atom.split(":", 1)
        else:
            self.main, self.feat = atom, None

    def __str__(self) -> str:
        return self.main + (f":{self.feat}" \
             if not(self.feat == "_" or self.feat is None or not self.feat or self.feat[0].isupper()) else "")    
    
    def __repr__(self) -> str:
        return f"AtomCaTy({self.main}" + \
            (f":{self.feat}" if self.feat is not None else "") + ")"


class CompCaTy(CaTy):
    """Compound Category/Types"""
    def __init__(self, f: str, args: List[Any]):
        if f not in CATY_F:
            raise ValueError(f"CompCaTy uses a wrong functor: {f}")
        if len(args) != 2:
            raise ValueError(f"CompCaTy expects two args: {args}")
        self.f = f
        self.arg = args[0] # already parsed by parse_caty
        self.fun = args[1]

    def __str__(self) -> str:
        return f"({self.arg}-{self.fun})"    
    
    def __repr__(self) -> str:
        return f"CompCaTy({repr(self.arg)}{self.f}{repr(self.fun)})"


class TreeLeaf(Compound):
    """Tree leaf in CCG derivation-like structures"""
    def __init__(self, f: str, args: List[Any]):
        if f not in {'t'}:
            raise ValueError(f"Terminal Compound uses a wrong functor: {f}")
        super().__init__(f, args)
        self.value = super().__str__()

    # different representation to better fit to a tree leaf
    def __str__(self) -> str:
        return '\n'.join(str(a) for a in self.args)

class TT(Compound):
    """Term-Type a format where a term is always paired with its type"""
    def __init__(self, tt: dict):
        f, args = tt['functor'], tt["args"]
        assert len(args) == 2, f"TT should have 2 args but has {len(args)}"
        assert f == ',', f"TT should have ',' functor but has {f}"
        super().__init__(f, args)
        term, ty = args
        self.type = parse_caty(ty)
        if isinstance(term, str):
            self.term = Var(term)
        else:
            self.term = parse_term(term)

    def __repr__(self) -> str:
        return f"TT({repr(self.term)}, {repr(self.type)})"

    def __str__(self) -> str:
        return f"({self.term} : {remove_outer_parens(str(self.type))})"
    
    def compact(self) -> str:
        return remove_outer_parens(compact_tt(self))
    
    def tree(self):
        return TT2Tree(self)
    
    def pretty_printer(self): 
        return TreePrettyPrinter(self.tree())

    def pretty_print(self): 
        return self.tree().pretty_print()

class AppTT(Compound):
    """Application of two TTs"""
    def __init__(self, fun: dict, arg: dict):
        self.fun, self.arg = parse_term(fun), parse_term(arg)
        super().__init__("@", [self.fun, self.arg])

    def __repr__(self) -> str:
        return f"AppTT({repr(self.fun)}, {repr(self.arg)})"

    def __str__(self) -> str:
        return f"{self.fun} @ {self.arg}"

class AbsTT(Compound):
    """Lambda abstraction of Var and TTs"""
    def __init__(self, var: dict, body: dict):
        self.var, self.body = parse_term(var), parse_term(body)
        super().__init__("λ", [self.var, self.body])

    def __repr__(self) -> str:
        return f"AbsTT({repr(self.var)}, {repr(self.body)})"
    
    def __str__(self) -> str:
        return f"λ{self.var}. {self.body}"

class TLP(Compound):
    """lexical constant is lambda terms that are a tuple of strings (token, lemma, pos tag)"""
    def __init__(self, f: str, args: List[Any]):
        super().__init__(f, args)
        self.tok, self.lem, self.pos = args[0], args[1], args[2]

    # different representation to better fit to a tree leaf
    def __repr__(self) -> str:
        return f"TLP({','.join(repr(a) for a in self.args)})"
    
    def __str__(self) -> str:
        return f"[{','.join(str(a) for a in self.args)}]"
    

class TreeNode(str):
    """Node in tableau proof tree"""
    def __new__(cls, trnd: dict):
        # a couple of checks
        try:
            f, (nd, node_id, rule_app, _) = trnd["functor"], trnd["args"]
            mod_list, head, arg_list, sign = nd["args"]
        except Exception as e:
            raise ValueError(f"Invalid tree node structure: {trnd}") from e        
        if f != 'trnd':
            raise ValueError(f"'trnd' functor is expected, found {f}")
        # process trnd args
        sign = sign == "true"
        rule_app = RuleApp(rule_app) if rule_app else None
        try:
            mod_list = list(map(parse_term, mod_list))
            arg_list = list(map(parse_term, arg_list))
            head = parse_term(head)
        except Exception as e:
            raise ValueError(f"Error parsing trnd args: {nd['args']}") from e
        
        # Create the string representation
        # skip modifier and argument lists if they are empty
        c_mods = f"\\n[{', '.join([mod.compact() for mod in mod_list])}]" if mod_list else ""
        c_args = f"\\n[{', '.join([arg.compact() for arg in arg_list])}]" if arg_list else ""
        c_head = head.compact()
        c_rule_app = f"{rule_app}" if rule_app else ""
        
        # TODO: improve formatting
        str_repr = f"{node_id}:{c_rule_app}{c_mods}\\n{c_head}{c_args}\\n{sign}" 
        str_repr = str_repr.replace(') @ (', ')(').replace(' @ ', ' ').replace('. ', '.')
        # str_repr = "⯁"
        
        # Create the str instance with this representation
        instance = super().__new__(cls, str_repr)
        
        # Store additional attributes
        instance.id = node_id
        # instance.rule_app = RuleApp(rule_app) if rule_app else None
        instance.mod = mod_list
        instance.head = head
        instance.arg = arg_list
        instance.sign = sign

        return instance

class RuleApp(Compound):
    """Rule application info in tableau proof tree nodes"""
    def __init__(self, rule_app: dict):
        assert 'functor' in rule_app, f"Rule app has no functor: {rule_app}"
        # print(">>> rule_app = ", rule_app)
        f, args = rule_app['functor'], rule_app["args"]
        super().__init__(f, args)
        self.rule = f
        # define ids and new/old constants
        if len(args) == 1:
            self.ids = args[0]
            self.new = self.old = None
        elif len(args) == 2:
            if isinstance(args[0][0], dict): # first is a list of terms
                old, self.ids = args
                self.new = None
                self.old = [ TT(i).compact() for i in old ]
            elif isinstance(args[1], list): # second is a list of terms
                self.ids, new = args
                self.old = None
                self.new = [ TT(i).compact() for i in new ]
            else:
                raise ValueError(f"Invalid rule app: {args}")
        else:
            raise ValueError(f"Invalid rule app (with 3+ args): {args}")
        
    def __str__(self) -> str:
        new = "" if self.new is None else f"{self.new}, "
        old = "" if self.old is None else f", {self.old}"
        return f"{self.rule}({new}{self.ids}{old})".replace(" ", "")
    
    def __repr__(self) -> str:
        return f"RuleApp({self.rule}, {self.ids}, new={self.new}, old={self.old})"


##############################################################
# Reading JSON data
##############################################################

COMPOUND_F_TYPE_MAP = {'tlp': TLP, 't': TreeLeaf, ':': AtomCaTy,
                       '~>': CompCaTy, '/': CompCaTy, '\\': CompCaTy}

# def parse_langpro_json(json_output: Any, v=0) -> Any:
#     """
#     Recursively converts a Python dictionary/list (from JSON) into PrologTerm objects.
#     """
#     new_output = dict()
#     for key, value in json_output.items():
#         if key == "kb":
#             new_output[key] = parse_kb(value)
#         if key == "prob":
#             new_output[key] = [ parse_langpro_json(v) for v in value ]
#         elif key == "tree":
#             new_output[key] = parse_langpro_json(value)
#         elif key == "ccg_tree":
#             new_output[key] = parse_ccg_tree(value)
#         elif key in ["ccg_term", "corr_term", "llf"]:
#             new_output[key] = parse_term(value)
#         elif key == "proofs":
#             pass
#         else:
#             new_output[key] = value
#     return new_output

##############################################################
# Reading certain Prolog objects
##############################################################

def parse_t_leaf(tleaf):
    """parses terminal nodes of CCG derivations"""
    f, args = tleaf['functor'], tleaf["args"]
    if f != 't':
        raise ValueError(f"'t' functor is expected, found {f} from {tleaf}")
    # process ccg category, while rest are token, lemma, pos, chunking, ner info
    c = parse_caty(args[0])
    return TreeLeaf(f, [c] + args[1:])

def parse_caty(caty: str | dict):
    """parses types and categories"""
    # atomic type or cat (without features)
    if isinstance(caty, str):
        return AtomCaTy(caty)
    f, args = caty['functor'], caty["args"]
    # atomic type or cat with a feature
    if f == ":" and len(args) == 2:
        return AtomCaTy(f"{args[0]}:{args[1]}")
    return CompCaTy(f, [ parse_caty(arg) for arg in args])

def parse_kb(kb: list):
    """parses KB, which is a list of rleations over a pair of words"""
    return [ Compound(r['functor'], r['args']) for r in kb ]


##############################################################
# Reading certain Prolog tree objects as NLTK Tree
##############################################################

def parse_ccg_tree(tree: dict):
    """
    Structure a CCG derivation as an NLTK Tree.
    Combinatory rules with a resulting category are
    used as non-terminal labels.
    """
    f, args = tree['functor'], tree["args"]
    # combinator names used by C&C and EasyCCG
    unary_combinators = {'lx', 'lex', 'tr'}
    binary_combinators = {'fa', 'ba', 'fc', 'bc', 'bx', 'fxc', 'bxc', 'conj',
                          'lp', 'rp', 'ltc', 'rtc', 'gfc', 'gbx', 'gbxc', 'gfxc'}
    # attach the resulted category to the rule name
    root = f"{f}({parse_caty(args[0])})" # combinator + TypeCat
    # process combinatory rules
    if f in unary_combinators:
        children = [parse_ccg_tree(args[-1])]
    elif f in binary_combinators:
        children = [parse_ccg_tree(ch) for ch in args[-2:]]
    # process leaves
    elif f == 't':
        return parse_t_leaf(tree)
    else:
        raise ValueError(f"Unknown combinatory rule: {f}")
    return Tree(root, children)


def parse_term(term: dict):
    """
    Structure a lambda term as an NLTK Tree.
    CatTypes are used as non-terminal labels.
    Here combinatory rules are not present as
    they are replaced with function application & abstraction.
    """
    if not isinstance(term, dict):
        raise ValueError(f"InvNon-dict term: {term}")
    if 'functor' not in term:
        raise ValueError(f"Invalid term: {term}")
    f, args = term['functor'], term["args"]
    # term-type pair
    if f == ',':
        return TT(term)
    # application of terms
    if f == '@':
        return AppTT(args[0], args[1])
    # lambda abstraction
    if f == 'abst':
        return AbsTT(args[0], args[1])
    # lexical term
    if f == 'tlp': # TODO: move this under TT?
        return TLP(f, args)

    # catching unforeseen cases
    raise ValueError(f"Unknown case: {term}")


def TT2Tree(t: TT|AppTT|AbsTT|Var) -> Any:
    """
    Structure a TT as an NLTK Tree.
    The root is the type, and the term is its child.
    """
    if isinstance(t, TT):
        if isinstance(t.term, AppTT):
            func = "@\n" 
        elif isinstance(t.term, AbsTT):
            func = "λ\n"
        else:
            func = ""
        pretty_type = remove_outer_parens(str(t.type))
        return Tree(f"{func}{pretty_type}", TT2Tree(t.term))
    if isinstance(t, AppTT):
        return [TT2Tree(t.fun), TT2Tree(t.arg)]
    if isinstance(t, AbsTT):
        return [TT2Tree(t.var), TT2Tree(t.body)]
    if isinstance(t, TLP):
        # parts = str(t)[1:-1].rsplit(',', 3)
        # return [parts[0] + '\n' + ','.join(parts[1:])]
        return [str(t)[1:-1]]
    if isinstance(t, Var):
        return [str(t)]
    
def compact_tt(t: TT|AppTT|AbsTT|Var) -> str:
    """Represent TT as a single line compact string."""
    if isinstance(t, Var):
        return str(t)
    if isinstance(t, TLP):
        return t.lem
    if isinstance(t, AbsTT):
        return f"(λ{compact_tt(t.var)}. {compact_tt(t.body)})"
    if isinstance(t, AppTT):
        return f"({compact_tt(t.fun)} @ {compact_tt(t.arg)})"
    if isinstance(t, TT):
        return f"{compact_tt(t.term)}"


def parse_info_proof(proof: dict):
    """
    Structure tableau proof as an NLTK Tree.
    Input can be actual proof dict, i.e., the value of "proof" key
    or dict with keys "info" and "proof".
    """
    # get actual proof dict
    p = proof["proof"] if "proof" in proof else proof
    proof_tree = parse_proof_tree(p)
    return proof_tree                               
                
def parse_proof_tree(dict_tree: dict):
    """Recursively parse proof tree"""
    # check that it is a tree
    if 'functor' not in dict_tree:
        print(f"Invalid proof tree: {dict_tree}")
        return None
    assert dict_tree['functor'] == 'tree', \
        f"Proof tree should have 'tree' as a functor but found: {dict_tree['functor']}"
    parent, children = dict_tree['args']
    assert parent['functor'] == 'trnd', \
        f"node should have 'trnd' as a functor but found: {parent['functor']}"
    # process children
    if isinstance(children, list):
        parsed_children = [ parse_proof_tree(child) for child in children ]
    elif isinstance(children, str):
        parsed_children = [children] # corresponds to the Model leaf
    elif isinstance(children, dict):
        #for closure rule
        if children['functor'] == 'closer':
            ids, rule = children['args'][0]
            parsed_children = [ f"Closed\n{rule}({ids})" ]
        else:
            ValueError(f"Unknown proof child type: {children}")
    else:
        raise ValueError(f"Unknown proof child type: {children}")
    return Tree(TreeNode(parent), parsed_children)


# TODO: This is only used for CCG trees for now. adapt it to terms or remove?
def tree_to_line(tree, op=False):
    """Represent NLTk Tree object as a single line string.
       This is useful to represent LLFs as a single line in proof trees
    """
    if isinstance(tree, str):
        n_cnt = tree.count("\n") # tyoe tok lemma pos ... tuple
        if n_cnt > 4:
            lemma = tree.split("\n")[2]
            return lemma
        elif n_cnt == 1: # type-var pair
            return  tree.split("\n")[-1]
        return tree
    if isinstance(tree, TreeLeaf):
        return tree.value # TODO: make more compact?
    if isinstance(tree, Tree) and "abst" in tree.label():
        var, body = tree
        return f"(\\{tree_to_line(var)}. {tree_to_line(body)})"
    # if isinstance(tree, TreeLeaf):
        # return tree.value # TODO: make more compact?
    # else:
    #     print(f">>> {type(tree)}: {tree}")
    # otherwise recurse
    if op:
        return f"({' '.join(tree_to_line(child, op=True) for child in tree)})"
    else:
        return f"{' '.join(tree_to_line(child, op=True) for child in tree)}"
    

def remove_outer_parens(s):
    if s.startswith('(') and s.endswith(')'):
        return s[1:-1]
    return s


_LANGPRO_CACHE_BACKEND: Optional[LangProCacheBackend] = None
_LANGPRO_INFLIGHT: Dict[str, "asyncio.Task[Tuple[Optional[str], Optional[str]]]"] = {}
_LANGPRO_INFLIGHT_LOCK: Optional[asyncio.Lock] = None
HYBRID_LANGPRO_ENDPOINT = "hybrid://auto"
HYBRID_LOCAL_ENDPOINT = "local://auto"
_HYBRID_FAILURE_THRESHOLD = 3
_HYBRID_INITIAL_BACKOFF_SECONDS = 15.0
_HYBRID_MAX_BACKOFF_SECONDS = 300.0

_SEN_ID_PATTERN = re.compile(
    r"sen_id\(\d+,\s*(?:(?P<num>[0-9]+)|'(?P<quoted>(?:\\'|[^'])*)'|(?P<atom>[A-Za-z_][A-Za-z0-9_]*)),\s*'(?P<role>[ph])',\s*'[^']*',\s*'(?P<sentence>(?:\\'|[^'])*)'\)\."
)


@dataclass
class HybridBackendHealth:
    consecutive_failures: int = 0
    disabled_until: float = 0.0
    next_probe_at: float = 0.0
    backoff_seconds: float = _HYBRID_INITIAL_BACKOFF_SECONDS


_HYBRID_BACKEND_HEALTH: Dict[str, HybridBackendHealth] = {
    "remote": HybridBackendHealth(),
    "local": HybridBackendHealth(),
}


@dataclass(frozen=True)
class LocalLangProCorpus:
    dataset_name: str
    sen_relpath: str
    ccg_relpath: Optional[str]
    easyccg_relpath: Optional[str]
    depccg_relpaths: Tuple[str, ...]


@dataclass(frozen=True)
class LocalLangProResolvedProblem:
    corpus: LocalLangProCorpus
    problem_id: str


def _normalize_kb_for_request(kb: Optional[List[str]]) -> List[str]:
    """Trim and de-duplicate KB relations while preserving first-seen order."""
    normalized: List[str] = []
    seen = set()

    for item in kb or []:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return normalized


def _normalize_kb_for_cache(kb: List[str]) -> Tuple[str, ...]:
    """
    Canonicalize KB order for cache keys so equal relation sets share one prover call,
    even when they come from different models or in a different order.
    """
    return tuple(sorted(kb))


def _make_langpro_cache_payload(
    premises: List[str],
    hypothesis: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    prover_config_extra: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "premises": premises,
        "hypothesis": hypothesis,
        "parser": parser,
        "ral": ral,
        "kb": list(_normalize_kb_for_cache(kb)),
        "senses": senses,
        "strong_align": strong_align,
        "intersective": intersective,
    }
    # Only add when set, so existing cache keys stay stable when unused.
    if prover_config_extra:
        payload["prover_config_extra"] = sorted(prover_config_extra)
    return payload


def _make_langpro_cache_key(
    premises: List[str],
    hypothesis: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    prover_config_extra: Tuple[str, ...] = (),
) -> str:
    payload = _make_langpro_cache_payload(
        premises,
        hypothesis,
        parser,
        ral,
        kb,
        senses,
        strong_align,
        intersective,
        prover_config_extra,
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _make_legacy_endpoint_langpro_cache_key(
    premises: List[str],
    hypothesis: str,
    endpoint: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
) -> str:
    payload = {
        "premises": premises,
        "hypothesis": hypothesis,
        "endpoint": endpoint,
        "parser": parser,
        "ral": ral,
        "kb": list(_normalize_kb_for_cache(kb)),
        "senses": senses,
        "strong_align": strong_align,
        "intersective": intersective,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_langpro_cache_backend() -> LangProCacheBackend:
    settings = get_langpro_settings()
    if settings.cache_backend == "memory":
        return InMemoryLangProCache()
    if settings.cache_backend == "sqlite":
        return SQLiteLangProCache(settings.cache_path)
    raise ValueError(
        f"Unknown LangPro cache backend: {settings.cache_backend}. "
        "Expected 'memory' or 'sqlite'."
    )


def get_langpro_cache_backend() -> LangProCacheBackend:
    global _LANGPRO_CACHE_BACKEND
    if _LANGPRO_CACHE_BACKEND is None:
        _LANGPRO_CACHE_BACKEND = _build_langpro_cache_backend()
    return _LANGPRO_CACHE_BACKEND


def set_langpro_cache_backend(cache_backend: LangProCacheBackend) -> None:
    global _LANGPRO_CACHE_BACKEND
    _LANGPRO_CACHE_BACKEND = cache_backend


def clear_langpro_cache() -> None:
    get_langpro_cache_backend().clear()


def _get_langpro_inflight_lock() -> asyncio.Lock:
    global _LANGPRO_INFLIGHT_LOCK
    if _LANGPRO_INFLIGHT_LOCK is None:
        _LANGPRO_INFLIGHT_LOCK = asyncio.Lock()
    return _LANGPRO_INFLIGHT_LOCK


def _normalize_problem_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_local_langpro_endpoint(endpoint: str) -> bool:
    return endpoint.startswith("local://")


def _is_hybrid_langpro_endpoint(endpoint: str) -> bool:
    return endpoint == HYBRID_LANGPRO_ENDPOINT


def _local_endpoint_cache_key(endpoint: str, settings) -> str:
    if not _is_local_langpro_endpoint(endpoint):
        return endpoint
    return (
        f"{endpoint}|root={settings.local_root.resolve()}|swipl={settings.local_swipl}"
        f"|easyccg_dir={settings.local_easyccg_dir.resolve()}"
        f"|easyccg_spacy={settings.local_easyccg_spacy_model}"
    )


def _legacy_endpoint_cache_candidates(resolved_endpoint: str, settings) -> Tuple[str, ...]:
    raw_candidates = [
        resolved_endpoint,
        DEFAULT_LANGPRO_ENDPOINT,
        HYBRID_LOCAL_ENDPOINT,
    ]
    candidates: List[str] = []
    seen = set()
    for endpoint in raw_candidates:
        cache_endpoint = _local_endpoint_cache_key(endpoint, settings)
        if cache_endpoint in seen:
            continue
        seen.add(cache_endpoint)
        candidates.append(cache_endpoint)
    return tuple(candidates)


def _get_and_migrate_legacy_langpro_cache_entry(
    cache_backend: LangProCacheBackend,
    new_cache_key: str,
    premises: List[str],
    hypothesis: str,
    resolved_endpoint: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    settings,
) -> Optional[str]:
    for legacy_endpoint in _legacy_endpoint_cache_candidates(resolved_endpoint, settings):
        legacy_cache_key = _make_legacy_endpoint_langpro_cache_key(
            premises,
            hypothesis,
            legacy_endpoint,
            parser,
            ral,
            kb,
            senses,
            strong_align,
            intersective,
        )
        cached_response_text = cache_backend.get(legacy_cache_key)
        if cached_response_text is not None:
            cache_backend.set(new_cache_key, cached_response_text)
            return cached_response_text
    return None


def _relation_to_prolog_atom(relation: str) -> str:
    value = (relation or "").strip()
    if not value:
        raise ValueError("KB relation cannot be empty")
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", value)
    if match is None:
        if re.fullmatch(r"[A-Za-z0-9_@\-]+", value):
            return value
        raise ValueError(f"Unsupported KB relation syntax for local LangPro: {relation}")

    functor, raw_args = match.groups()
    args = [arg.strip() for arg in raw_args.split(",")]
    if not args or any(not arg for arg in args):
        raise ValueError(f"Unsupported KB relation syntax for local LangPro: {relation}")

    normalized_args = []
    for arg in args:
        if re.fullmatch(r"[A-Za-z0-9_@\-]+", arg):
            normalized_args.append(arg)
        else:
            escaped = arg.replace("\\", "\\\\").replace("'", "\\'")
            normalized_args.append(f"'{escaped}'")

    return f"{functor}({', '.join(normalized_args)})"


@lru_cache(maxsize=1)
def _discover_local_langpro_corpora(local_root: Path) -> Tuple[LocalLangProCorpus, ...]:
    corpus_dir = local_root / "ccg_sen_d"
    if not corpus_dir.exists():
        return ()

    corpora: List[LocalLangProCorpus] = []
    for sen_path in sorted(corpus_dir.glob("*_sen.pl")):
        prefix = sen_path.name.removesuffix("_sen.pl")
        dataset_name = _local_langpro_dataset_name(prefix)
        ccg_relpath = _local_langpro_optional_relpath(corpus_dir, f"{prefix}_ccg.pl")
        easyccg_relpath = _local_langpro_optional_relpath(corpus_dir, f"{prefix}_eccg.pl")
        depccg_relpaths = tuple(
            relpath
            for relpath in (
                _local_langpro_optional_relpath(corpus_dir, f"{prefix}_depccg.trihf.sep.pl"),
                _local_langpro_optional_relpath(corpus_dir, f"{prefix}_depccg.trihf.pl"),
            )
            if relpath is not None
        )
        corpora.append(
            LocalLangProCorpus(
                dataset_name=dataset_name,
                sen_relpath=f"ccg_sen_d/{sen_path.name}",
                ccg_relpath=ccg_relpath,
                easyccg_relpath=easyccg_relpath,
                depccg_relpaths=depccg_relpaths,
            )
        )

    return tuple(corpora)


def _local_langpro_optional_relpath(corpus_dir: Path, filename: str) -> Optional[str]:
    path = corpus_dir / filename
    if not path.exists():
        return None
    return f"ccg_sen_d/{filename}"


def _local_langpro_dataset_name(prefix: str) -> str:
    upper_prefix = prefix.upper()
    if upper_prefix.startswith("SICK"):
        return "sick"
    if upper_prefix.startswith("SNLI"):
        return "snli"
    if upper_prefix.startswith("FRACAS"):
        return "fracas"
    return prefix.split("_", 1)[0].lower()


def _local_langpro_problem_id_from_match(match: re.Match[str]) -> str:
    raw_problem_id = match.group("num") or match.group("quoted") or match.group("atom")
    return raw_problem_id.replace("\\'", "'")


def _local_langpro_corpus_parser_score(corpus: LocalLangProCorpus, parser: str) -> int:
    parser_name = (parser or "").strip().lower()
    if parser_name == "easyccg":
        if corpus.easyccg_relpath:
            return 3
        if corpus.ccg_relpath:
            return 2
        if corpus.depccg_relpaths:
            return 1
        return 0

    if parser_name.startswith("depccg"):
        if corpus.depccg_relpaths:
            return 3
        if corpus.ccg_relpath:
            return 2
        if corpus.easyccg_relpath:
            return 1
        return 0

    if corpus.ccg_relpath:
        return 3
    if corpus.easyccg_relpath:
        return 2
    if corpus.depccg_relpaths:
        return 1
    return 0


@lru_cache(maxsize=1)
def _load_local_langpro_problem_index(
    local_root: Path,
) -> Dict[Tuple[Tuple[str, ...], str], Tuple[LocalLangProResolvedProblem, ...]]:
    index: Dict[Tuple[Tuple[str, ...], str], List[LocalLangProResolvedProblem]] = {}

    for corpus in _discover_local_langpro_corpora(local_root):
        sen_path = local_root / corpus.sen_relpath
        by_problem: Dict[str, Dict[str, Any]] = {}
        with open(sen_path, "r", encoding="utf-8") as handle:
            for line in handle:
                match = _SEN_ID_PATTERN.search(line)
                if not match:
                    continue
                problem_id = _local_langpro_problem_id_from_match(match)
                role = match.group("role")
                sentence = match.group("sentence").replace("\\'", "'")
                entry = by_problem.setdefault(problem_id, {"premises": [], "hypothesis": None})
                if role == "p":
                    entry["premises"].append(_normalize_problem_text(sentence))
                else:
                    entry["hypothesis"] = _normalize_problem_text(sentence)

        for problem_id, entry in by_problem.items():
            hypothesis = entry["hypothesis"]
            if not entry["premises"] or not hypothesis:
                continue
            key = (tuple(entry["premises"]), hypothesis)
            index.setdefault(key, []).append(LocalLangProResolvedProblem(corpus=corpus, problem_id=problem_id))

    return {
        key: tuple(
            sorted(
                refs,
                key=lambda ref: (
                    -max(
                        _local_langpro_corpus_parser_score(ref.corpus, "easyccg"),
                        _local_langpro_corpus_parser_score(ref.corpus, "depccg"),
                        _local_langpro_corpus_parser_score(ref.corpus, ""),
                    ),
                    ref.corpus.sen_relpath,
                    ref.problem_id,
                ),
            )
        )
        for key, refs in index.items()
    }


def _resolve_local_langpro_problem(
    premises: List[str],
    hypothesis: str,
    local_root: Path,
    parser: str,
) -> Optional[LocalLangProResolvedProblem]:
    normalized_key = (
        tuple(_normalize_problem_text(premise) for premise in premises),
        _normalize_problem_text(hypothesis),
    )
    refs = _load_local_langpro_problem_index(local_root).get(normalized_key)
    if not refs:
        return None
    return max(
        refs,
        key=lambda ref: (
            _local_langpro_corpus_parser_score(ref.corpus, parser),
            -len(ref.corpus.sen_relpath),
            ref.corpus.sen_relpath,
        ),
    )


def _local_langpro_ccg_file(corpus: LocalLangProCorpus, parser: str) -> Optional[str]:
    parser_name = (parser or "").strip().lower()
    if parser_name == "easyccg":
        return corpus.easyccg_relpath or corpus.ccg_relpath or next(iter(corpus.depccg_relpaths), None)
    if parser_name.startswith("depccg"):
        return next(iter(corpus.depccg_relpaths), None) or corpus.ccg_relpath or corpus.easyccg_relpath
    return corpus.ccg_relpath or corpus.easyccg_relpath or next(iter(corpus.depccg_relpaths), None)


def _local_langpro_sen_file(corpus: LocalLangProCorpus) -> str:
    return corpus.sen_relpath


def _format_local_langpro_problem_id(problem_id: str) -> str:
    stripped = (problem_id or "").strip()
    if re.fullmatch(r"[0-9]+", stripped):
        return stripped
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        return stripped
    escaped = stripped.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _local_langpro_goal_prefix(ral: int, strong_align: bool, intersective: bool) -> str:
    params = [f"ral({ral})", "mwe"]
    if strong_align:
        params.append("aall")
    if intersective:
        params.append("allInt")
    return "['prolog/main.pl']," + f"parList([{','.join(params)}]),"


def _escape_prolog_string(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def _write_local_langpro_sen_file(
    sen_path: Path,
    premises: List[str],
    hypothesis: str,
) -> None:
    lines = ["% generated by kbprojection local LangPro raw-text fallback"]
    sentence_id = 1
    for premise in premises:
        escaped = _escape_prolog_string(premise)
        lines.append(f"sen_id({sentence_id}, 'prob1', 'p', 'unknown', '{escaped}').")
        sentence_id += 1
    escaped_hypothesis = _escape_prolog_string(hypothesis)
    lines.append(f"sen_id({sentence_id}, 'prob1', 'h', 'unknown', '{escaped_hypothesis}').")
    sen_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run_subprocess_text(
    args: List[str],
    timeout_seconds: float,
    cwd: Optional[Path] = None,
    input_text: Optional[str] = None,
) -> Tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd is not None else None,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input_text.encode("utf-8") if input_text is not None else None),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise
    return (
        process.returncode or 0,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
    )


def _format_async_process_error(command: str, returncode: int, stdout: str, stderr: str) -> str:
    details = (stderr or stdout or "").strip()
    return f"{command} failed with code {returncode}: {details}"


async def _ensure_langpro_clone_async(settings) -> Path:
    destination = settings.local_vendor_root
    if destination.exists() and not (destination / ".git").exists():
        raise RuntimeError(
            f"LangPro clone destination exists but is not a git checkout: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        clone_branch_code, clone_branch_out, clone_branch_err = await _run_subprocess_text(
            ["git", "clone", "--branch", settings.local_repo_ref, settings.local_repo_url, str(destination)],
            timeout_seconds=300.0,
        )
        if clone_branch_code != 0:
            clone_plain_code, clone_plain_out, clone_plain_err = await _run_subprocess_text(
                ["git", "clone", settings.local_repo_url, str(destination)],
                timeout_seconds=300.0,
            )
            if clone_plain_code != 0:
                raise RuntimeError(
                    _format_async_process_error(
                        f"git clone --branch {settings.local_repo_ref} {settings.local_repo_url} {destination}",
                        clone_branch_code,
                        clone_branch_out,
                        clone_branch_err,
                    )
                    + " | "
                    + _format_async_process_error(
                        f"git clone {settings.local_repo_url} {destination}",
                        clone_plain_code,
                        clone_plain_out,
                        clone_plain_err,
                    )
                )

    checkout_code, checkout_out, checkout_err = await _run_subprocess_text(
        ["git", "checkout", settings.local_repo_ref],
        cwd=destination,
        timeout_seconds=120.0,
    )
    if checkout_code != 0:
        raise RuntimeError(
            _format_async_process_error(
                f"git checkout {settings.local_repo_ref}",
                checkout_code,
                checkout_out,
                checkout_err,
            )
        )

    return destination


async def _resolve_local_langpro_root_async(settings) -> Path:
    local_root = settings.local_root
    if local_root.exists():
        return local_root
    if settings.local_auto_clone:
        return await _ensure_langpro_clone_async(settings)
    return local_root


async def _run_easyccg_once_async(
    sentence: str,
    easyccg_dir: Path,
    model_name: str,
    timeout_seconds: float,
) -> str:
    tagged_input = _tag_sentence(sentence, model_name)
    returncode, stdout, stderr = await _run_subprocess_text(
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
        input_text=tagged_input,
        timeout_seconds=timeout_seconds,
        cwd=easyccg_dir.parents[2],
    )
    output = (stdout or "").strip()
    if returncode != 0 or not output:
        error_bits = [f"EasyCCG exited with code {returncode}"]
        if stderr:
            error_bits.append(stderr.strip())
        raise RuntimeError(" | ".join(error_bits))
    return output


async def _run_easyccg_async(
    sentence: str,
    easyccg_dir: Path,
    model_name: str,
    timeout_seconds: float,
) -> str:
    try:
        return await _run_easyccg_once_async(sentence, easyccg_dir, model_name, timeout_seconds)
    except RuntimeError as exc:
        if "Unknown rule type: NOISE" not in str(exc):
            raise
        fallback_sentence = _normalize_easyccg_noise_sentence(sentence)
        if not fallback_sentence or fallback_sentence == (sentence or "").strip():
            raise
        return await _run_easyccg_once_async(fallback_sentence, easyccg_dir, model_name, timeout_seconds)


_EASYCCG_TERM_LOCKS: Dict[str, asyncio.Lock] = {}


def _easyccg_term_cache_path(sentence: str, easyccg_dir: Path, model_name: str) -> Path:
    settings = get_langpro_settings()
    payload = {
        "sentence": sentence or "",
        "easyccg_dir": str(easyccg_dir.resolve()),
        "model_name": model_name,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return settings.cache_dir / "easyccg_terms" / f"{key}.json"


def _read_easyccg_term_cache(cache_path: Path) -> Optional[str]:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    term = payload.get("term")
    return term if isinstance(term, str) and term.strip() else None


def _write_easyccg_term_cache(cache_path: Path, sentence: str, term: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    payload = {"sentence": sentence or "", "term": term}
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(cache_path)


async def _get_easyccg_term_cached(
    sentence: str,
    easyccg_dir: Path,
    model_name: str,
    timeout_seconds: float,
) -> str:
    cache_path = _easyccg_term_cache_path(sentence, easyccg_dir, model_name)
    cached = _read_easyccg_term_cache(cache_path)
    if cached is not None:
        return cached

    cache_key = str(cache_path)
    lock = _EASYCCG_TERM_LOCKS.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _read_easyccg_term_cache(cache_path)
        if cached is not None:
            return cached

        raw_output = await _run_easyccg_async(sentence, easyccg_dir, model_name, timeout_seconds)
        term = process_single_output(raw_output)
        if term is None:
            raise RuntimeError("Failed to convert EasyCCG output to LangPro Prolog terms")
        _write_easyccg_term_cache(cache_path, sentence, term)
        return term


async def _write_local_langpro_ccg_file(
    ccg_path: Path,
    premises: List[str],
    hypothesis: str,
    easyccg_dir: Path,
    spacy_model: str,
    timeout_seconds: float,
) -> None:
    sentences = list(premises) + [hypothesis]
    terms: List[str] = []
    for sentence in sentences:
        terms.append(
            await _get_easyccg_term_cached(
                sentence,
                easyccg_dir,
                spacy_model,
                timeout_seconds,
            )
        )

    lines = [
        "% generated by kbprojection local LangPro EasyCCG fallback",
        ":- op(601, xfx, (/)).",
        ":- op(601, xfx, (\\)).",
        ":- multifile ccg/2.",
        ":- dynamic ccg/2.",
        ":- style_check(-discontiguous).",
        "",
    ]
    for sentence_id, term in enumerate(terms, start=1):
        lines.extend([f"ccg({sentence_id},", term, ").", ""])

    ccg_path.write_text("\n".join(lines), encoding="utf-8")


async def _execute_local_langpro_raw_request(
    parser: str,
    premises: List[str],
    hypothesis: str,
    kb: List[str],
    report: bool,
    timeout_seconds: float,
    ral: int,
    strong_align: bool,
    intersective: bool,
) -> Tuple[Optional[str], Optional[str]]:
    settings = get_langpro_settings()
    try:
        local_root = (await _resolve_local_langpro_root_async(settings)).resolve()
    except RuntimeError as exc:
        return None, str(exc)
    raw_timeout = max(timeout_seconds, 180.0)
    if not local_root.exists():
        return None, format_local_langpro_missing_error()

    parser_name = (parser or "").strip().lower()
    responses: List[str] = []
    errors: List[str] = []

    if parser_name != "easyccg":
        return None, "Local LangPro raw-text fallback only supports parser='easyccg'."

    if easyccg_is_available(settings.local_easyccg_dir):
        premise_candidates = [build_easyccg_parse_candidates(premise, settings.local_easyccg_spacy_model) for premise in premises]
        hypothesis_candidates = build_easyccg_parse_candidates(hypothesis, settings.local_easyccg_spacy_model)

        for premise_variant_group in product(*premise_candidates):
            for hypothesis_variant in hypothesis_candidates:
                response_text, error = await _execute_single_local_langpro_raw_request(
                    list(premise_variant_group),
                    hypothesis_variant,
                    kb,
                    report,
                    raw_timeout,
                    local_root,
                    settings,
                    ral=ral,
                    strong_align=strong_align,
                    intersective=intersective,
                )
                if response_text is None:
                    if error:
                        errors.append(error)
                    continue
                responses.append(response_text)
                if _parse_langpro_output(json.loads(response_text)).label != NLILabel.NEUTRAL:
                    return response_text, None
    else:
        errors.append(
            f"EasyCCG is not available at {settings.local_easyccg_dir}. "
            "Install EasyCCG there or use a preparsed local LangPro corpus."
        )

    if responses:
        return responses[0], None

    if report and errors:
        print("Local LangPro raw-text fallback failed:", " | ".join(errors))
    return None, " | ".join(errors) or "Local LangPro raw-text fallback returned no JSON output"


async def _execute_single_local_langpro_raw_request(
    premises: List[str],
    hypothesis: str,
    kb: List[str],
    report: bool,
    timeout_seconds: float,
    local_root: Path,
    settings,
    ral: int,
    strong_align: bool,
    intersective: bool,
) -> Tuple[Optional[str], Optional[str]]:
    with tempfile.TemporaryDirectory(prefix="kbprojection-langpro-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        ccg_path = temp_dir / "ccg.pl"
        sen_path = temp_dir / "sen.pl"
        _write_local_langpro_sen_file(sen_path, premises, hypothesis)

        try:
            await _write_local_langpro_ccg_file(
                ccg_path,
                premises,
                hypothesis,
                settings.local_easyccg_dir,
                settings.local_easyccg_spacy_model,
                timeout_seconds,
            )
        except (OSError, RuntimeError, ValueError, ImportError, asyncio.TimeoutError) as exc:
            return None, f"Failed to run EasyCCG: {exc}"

        kb_arg = ", ".join(_relation_to_prolog_atom(item) for item in kb)
        goal_prefix = _local_langpro_goal_prefix(ral, strong_align, intersective)
        if kb_arg:
            goal = (
                goal_prefix +
                f"ensure_loaded('{ccg_path.as_posix()}'),"
                f"ensure_loaded('{sen_path.as_posix()}'),"
                "['prolog/task/online_demo.pl'],"
                f"online_demo('prob1',[{kb_arg}],json(80,2,4)),halt."
            )
        else:
            goal = (
                goal_prefix +
                f"ensure_loaded('{ccg_path.as_posix()}'),"
                f"ensure_loaded('{sen_path.as_posix()}'),"
                "['prolog/task/online_demo.pl'],"
                "online_demo('prob1',json(80,2,4)),halt."
            )

        try:
            returncode, stdout, stderr = await _run_subprocess_text(
                [settings.local_swipl, "-q", "-g", goal],
                cwd=local_root,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            return None, f"Failed to run local LangPro on raw parsed input: {exc}"

        output = (stdout or "").strip()
        json_text = _extract_json_object(output)
        if returncode != 0 or json_text is None:
            error_bits = []
            if returncode != 0:
                error_bits.append(f"swipl exited with code {returncode}")
            if stderr:
                error_bits.append(stderr.strip())
            if json_text is None and output:
                error_bits.append(output[-500:])
            if report and error_bits:
                print("Local LangPro raw-text fallback failed:", " | ".join(error_bits))
            return None, " | ".join(error_bits) or "Local LangPro raw-text fallback returned no JSON output"

        try:
            json.loads(json_text)
        except json.decoder.JSONDecodeError as exc:
            return None, f"Local LangPro raw-text fallback returned invalid JSON: {exc}"

        return json_text, None


async def _execute_local_langpro_request(
    premises: List[str],
    hypothesis: str,
    parser: str,
    kb: List[str],
    report: bool,
    timeout_seconds: float,
    ral: int,
    strong_align: bool,
    intersective: bool,
) -> Tuple[Optional[str], Optional[str]]:
    settings = get_langpro_settings()
    try:
        local_root = (await _resolve_local_langpro_root_async(settings)).resolve()
    except RuntimeError as exc:
        return None, str(exc)

    if not local_root.exists():
        return None, format_local_langpro_missing_error()

    resolved = _resolve_local_langpro_problem(premises, hypothesis, local_root, parser)
    if resolved is None:
        return await _execute_local_langpro_raw_request(
            parser,
            premises,
            hypothesis,
            kb,
            report,
            timeout_seconds,
            ral,
            strong_align,
            intersective,
        )

    ccg_file = _local_langpro_ccg_file(resolved.corpus, parser)
    if ccg_file is None:
        return await _execute_local_langpro_raw_request(
            parser,
            premises,
            hypothesis,
            kb,
            report,
            timeout_seconds,
            ral,
            strong_align,
            intersective,
        )
    sen_file = _local_langpro_sen_file(resolved.corpus)
    problem_id = _format_local_langpro_problem_id(resolved.problem_id)
    kb_arg = ", ".join(_relation_to_prolog_atom(item) for item in kb)
    goal_prefix = _local_langpro_goal_prefix(ral, strong_align, intersective)

    if kb_arg:
        goal = (
            goal_prefix +
            f"ensure_loaded('{ccg_file}'),"
            f"ensure_loaded('{sen_file}'),"
            "['prolog/task/online_demo.pl'],"
            f"online_demo({problem_id},[{kb_arg}],json(80,2,4)),halt."
        )
    else:
        goal = (
            goal_prefix +
            f"ensure_loaded('{ccg_file}'),"
            f"ensure_loaded('{sen_file}'),"
            "['prolog/task/online_demo.pl'],"
            f"online_demo({problem_id},json(80,2,4)),halt."
        )

    try:
        returncode, stdout, stderr = await _run_subprocess_text(
            [settings.local_swipl, "-q", "-g", goal],
            cwd=local_root,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        return None, f"Failed to run local LangPro: {exc}"

    output = (stdout or "").strip()
    json_text = _extract_json_object(output)
    if returncode != 0 or json_text is None:
        error_bits = []
        if returncode != 0:
            error_bits.append(f"swipl exited with code {returncode}")
        if stderr:
            error_bits.append(stderr.strip())
        if json_text is None and output:
            error_bits.append(output[-500:])
        if report and error_bits:
            print("Local LangPro failed:", " | ".join(error_bits))
        return None, " | ".join(error_bits) or "Local LangPro returned no JSON output"

    try:
        json.loads(json_text)
    except json.decoder.JSONDecodeError as exc:
        return None, f"Local LangPro returned invalid JSON: {exc}"

    return json_text, None


def _parse_langpro_output(output: Dict[str, Any]) -> LangProResult:
    # parsing the components of the output
    kb_parsed = parse_kb(output.get('kb', []))
    
    # Safe list comprehensions in case 'prob' is missing or malformed
    probs = output.get('prob', [])
    ccg_trees = [ parse_ccg_tree(i['tree']['ccg_tree']) for i in probs if 'tree' in i ]
    ccg_terms = [ parse_term(i['tree']['ccg_term']) for i in probs if 'tree' in i ]
    corr_terms = [ parse_term(i['tree']['corr_term']) for i in probs if 'tree' in i ]
    llfs = [ parse_term(i['tree']['llf']) for i in probs if 'tree' in i ]
    
    raw_proofs = output.get('proofs', {})
    lab_proofs = { label: parse_info_proof(info_proof) 
                    for label, info_proof in raw_proofs.items() }
    
    # derive a predicted inference label
    entailment_info = raw_proofs.get("entailment", {}).get("info", [])
    contradiction_info = raw_proofs.get("contradiction", {}).get("info", [])
    
    entailment = 'closed' in entailment_info
    contradiction = 'closed' in contradiction_info
    
    if entailment and not contradiction:
        label = NLILabel.ENTAILMENT
    elif not entailment and contradiction:
        label = NLILabel.CONTRADICTION
    else:
        label = NLILabel.NEUTRAL
        
    return LangProResult(
        label=label,
        kb=kb_parsed,
        ccg_trees=ccg_trees,
        ccg_terms=ccg_terms,
        terms=corr_terms,
        llfs=llfs,
        proofs=lab_proofs,
        proof_info={
            "entailment": list(entailment_info),
            "contradiction": list(contradiction_info),
        },
    )


def _reset_hybrid_backend_health() -> None:
    for health in _HYBRID_BACKEND_HEALTH.values():
        health.consecutive_failures = 0
        health.disabled_until = 0.0
        health.next_probe_at = 0.0
        health.backoff_seconds = _HYBRID_INITIAL_BACKOFF_SECONDS


def _is_hybrid_backend_eligible(backend: str, now: float) -> bool:
    health = _HYBRID_BACKEND_HEALTH[backend]
    if health.consecutive_failures < _HYBRID_FAILURE_THRESHOLD:
        return True
    return now >= health.next_probe_at


def _eligible_hybrid_backends(now: float) -> List[str]:
    backends = [
        backend
        for backend in ("remote", "local")
        if _is_hybrid_backend_eligible(backend, now)
    ]
    return backends or ["remote", "local"]


def _mark_hybrid_backend_success(backend: str) -> None:
    health = _HYBRID_BACKEND_HEALTH[backend]
    health.consecutive_failures = 0
    health.disabled_until = 0.0
    health.next_probe_at = 0.0
    health.backoff_seconds = _HYBRID_INITIAL_BACKOFF_SECONDS


def _mark_hybrid_backend_failure(backend: str, now: float) -> None:
    health = _HYBRID_BACKEND_HEALTH[backend]
    health.consecutive_failures += 1
    if health.consecutive_failures >= _HYBRID_FAILURE_THRESHOLD:
        next_probe_at = now + health.backoff_seconds
        health.disabled_until = next_probe_at
        health.next_probe_at = next_probe_at
        health.backoff_seconds = min(
            health.backoff_seconds * 2.0,
            _HYBRID_MAX_BACKOFF_SECONDS,
        )


def penalize_hybrid_local_backend_for_timeout() -> None:
    """Bias hybrid LangPro away from the local backend after an outer timeout."""
    _mark_hybrid_backend_failure("local", time.monotonic())


async def _acquire_hybrid_backend_token(
    resolved_context: AsyncRunContext,
    backends: List[str],
) -> str:
    if len(backends) == 1:
        backend = backends[0]
        semaphore = (
            resolved_context.langpro_semaphore
            if backend == "remote"
            else resolved_context.local_langpro_semaphore
        )
        await semaphore.acquire()
        return backend

    acquire_tasks = {
        asyncio.create_task(resolved_context.langpro_semaphore.acquire()): "remote",
        asyncio.create_task(resolved_context.local_langpro_semaphore.acquire()): "local",
    }
    filtered_tasks = {
        task: backend
        for task, backend in acquire_tasks.items()
        if backend in backends
    }
    for task, backend in acquire_tasks.items():
        if backend not in backends:
            task.cancel()

    done, pending = await asyncio.wait(
        set(filtered_tasks),
        return_when=asyncio.FIRST_COMPLETED,
    )
    chosen_task = next(iter(done))
    chosen_backend = filtered_tasks[chosen_task]

    for task in done:
        if task is chosen_task:
            continue
        extra_backend = filtered_tasks[task]
        _release_hybrid_backend_token(resolved_context, extra_backend)

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    return chosen_backend


def _release_hybrid_backend_token(resolved_context: AsyncRunContext, backend: str) -> None:
    if backend == "remote":
        resolved_context.langpro_semaphore.release()
    else:
        resolved_context.local_langpro_semaphore.release()


async def _execute_local_langpro_request_without_limit(
    premises: List[str],
    hypothesis: str,
    parser: str,
    kb: List[str],
    report: bool,
    timeout_seconds: float,
    ral: int,
    strong_align: bool,
    intersective: bool,
) -> Tuple[Optional[str], Optional[str]]:
    return await _execute_local_langpro_request(
        premises,
        hypothesis,
        parser,
        kb,
        report,
        timeout_seconds,
        ral,
        strong_align,
        intersective,
    )


async def _execute_local_langpro_request_with_limit(
    premises: List[str],
    hypothesis: str,
    parser: str,
    ral: int,
    kb: List[str],
    strong_align: bool,
    intersective: bool,
    report: bool,
    timeout_seconds: float,
    context: Optional[AsyncRunContext] = None,
) -> Tuple[Optional[str], Optional[str]]:
    resolved_context = resolve_async_run_context(context)
    async with resolved_context.local_langpro_semaphore:
        return await _execute_local_langpro_request_without_limit(
            premises,
            hypothesis,
            parser,
            kb,
            report,
            timeout_seconds,
            ral,
            strong_align,
            intersective,
        )


async def _execute_remote_langpro_request_without_limit(
    premises: List[str],
    hypothesis: str,
    endpoint: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    timeout_seconds: float,
) -> Tuple[Optional[str], Optional[str]]:
    # preparing an input for the API call
    prob = {'premises': premises, 'hypothesis': hypothesis}
    headers={'Content-Type': 'application/json'}
    parameters = {  'prover_config': [],
                    'parser': parser,
                    'ral': ral,
                    'kb': kb,
                    'senses': senses    }
    if strong_align: parameters['prover_config'].append('aall')
    if intersective: parameters['prover_config'].append('allInt')
    for flag in get_langpro_settings().prover_config_extra:
        if flag not in parameters['prover_config']:
            parameters['prover_config'].append(flag)
    query = {**prob, **parameters}
    js_query = json.dumps(query)

    if curl:
        curl_command = f"curl '{endpoint}' " + \
        " ".join([f"-H '{k}: {v}'" for k, v in headers.items()]) + \
        f" -d '{js_query}'"
        print(curl_command)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(endpoint, content=js_query, headers=headers)
        response.raise_for_status()
        json.loads(response.text)
    except (json.decoder.JSONDecodeError, httpx.HTTPError) as e:
        if report:
            print(f"Failed to call LangPro API: {e}")
        return None, str(e)

    return response.text, None


async def _execute_remote_langpro_request(
    premises: List[str],
    hypothesis: str,
    endpoint: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    timeout_seconds: float,
    context: Optional[AsyncRunContext] = None,
) -> Tuple[Optional[str], Optional[str]]:
    resolved_context = resolve_async_run_context(context)
    async with resolved_context.langpro_semaphore:
        return await _execute_remote_langpro_request_without_limit(
            premises,
            hypothesis,
            endpoint,
            parser,
            ral,
            kb,
            senses,
            strong_align,
            intersective,
            curl,
            report,
            timeout_seconds,
        )


async def _execute_hybrid_backend_with_token(
    backend: str,
    premises: List[str],
    hypothesis: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    timeout_seconds: float,
) -> Tuple[Optional[str], Optional[str]]:
    if backend == "remote":
        return await _execute_remote_langpro_request_without_limit(
            premises,
            hypothesis,
            DEFAULT_LANGPRO_ENDPOINT,
            parser,
            ral,
            kb,
            senses,
            strong_align,
            intersective,
            curl,
            report,
            timeout_seconds,
        )
    return await _execute_local_langpro_request_without_limit(
        premises,
        hypothesis,
        parser,
        kb,
        report,
        timeout_seconds,
        ral,
        strong_align,
        intersective,
    )


async def _execute_hybrid_backend(
    backend: str,
    resolved_context: AsyncRunContext,
    premises: List[str],
    hypothesis: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    timeout_seconds: float,
    *,
    token_acquired: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    if not token_acquired:
        semaphore = (
            resolved_context.langpro_semaphore
            if backend == "remote"
            else resolved_context.local_langpro_semaphore
        )
        await semaphore.acquire()
    try:
        return await _execute_hybrid_backend_with_token(
            backend,
            premises,
            hypothesis,
            parser,
            ral,
            kb,
            senses,
            strong_align,
            intersective,
            curl,
            report,
            timeout_seconds,
        )
    finally:
        _release_hybrid_backend_token(resolved_context, backend)


async def _execute_hybrid_langpro_request(
    premises: List[str],
    hypothesis: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    timeout_seconds: float,
    context: Optional[AsyncRunContext] = None,
) -> Tuple[Optional[str], Optional[str]]:
    resolved_context = resolve_async_run_context(context)
    first_backend = await _acquire_hybrid_backend_token(
        resolved_context,
        _eligible_hybrid_backends(time.monotonic()),
    )
    errors: Dict[str, str] = {}

    response_text, error = await _execute_hybrid_backend(
        first_backend,
        resolved_context,
        premises,
        hypothesis,
        parser,
        ral,
        kb,
        senses,
        strong_align,
        intersective,
        curl,
        report,
        timeout_seconds,
        token_acquired=True,
    )
    if response_text is not None:
        _mark_hybrid_backend_success(first_backend)
        return response_text, None

    errors[first_backend] = error or "unknown error"
    _mark_hybrid_backend_failure(first_backend, time.monotonic())

    fallback_backend = "local" if first_backend == "remote" else "remote"
    response_text, fallback_error = await _execute_hybrid_backend(
        fallback_backend,
        resolved_context,
        premises,
        hypothesis,
        parser,
        ral,
        kb,
        senses,
        strong_align,
        intersective,
        curl,
        report,
        timeout_seconds,
    )
    if response_text is not None:
        _mark_hybrid_backend_success(fallback_backend)
        return response_text, None

    errors[fallback_backend] = fallback_error or "unknown error"
    _mark_hybrid_backend_failure(fallback_backend, time.monotonic())
    return (
        None,
        "Hybrid LangPro failed. "
        f"remote: {errors.get('remote', 'not attempted')} | "
        f"local: {errors.get('local', 'not attempted')}",
    )


async def _execute_langpro_request(
    premises: List[str],
    hypothesis: str,
    endpoint: str,
    parser: str,
    ral: int,
    kb: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    timeout_seconds: float,
    context: Optional[AsyncRunContext] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if _is_hybrid_langpro_endpoint(endpoint):
        return await _execute_hybrid_langpro_request(
            premises,
            hypothesis,
            parser,
            ral,
            kb,
            senses,
            strong_align,
            intersective,
            curl,
            report,
            timeout_seconds,
            context=context,
        )
    if _is_local_langpro_endpoint(endpoint):
        return await _execute_local_langpro_request_with_limit(
            premises,
            hypothesis,
            parser,
            ral,
            kb,
            strong_align,
            intersective,
            report,
            timeout_seconds,
            context=context,
        )
    return await _execute_remote_langpro_request(
        premises,
        hypothesis,
        endpoint,
        parser,
        ral,
        kb,
        senses,
        strong_align,
        intersective,
        curl,
        report,
        timeout_seconds,
        context=context,
    )


async def _execute_langpro_request_for_cache(
    cache_key: str,
    cache_backend: LangProCacheBackend,
    premises_list: List[str],
    hypothesis: str,
    resolved_endpoint: str,
    parser: str,
    ral: int,
    kb_list: List[str],
    senses: str,
    strong_align: bool,
    intersective: bool,
    curl: bool,
    report: bool,
    resolved_timeout: float,
    context: Optional[AsyncRunContext],
) -> Tuple[Optional[str], Optional[str]]:
    lock = _get_langpro_inflight_lock()
    async with lock:
        cached_response_text = cache_backend.get(cache_key)
        if cached_response_text is not None:
            return cached_response_text, None
        task = _LANGPRO_INFLIGHT.get(cache_key)
        if task is None:
            task = asyncio.create_task(
                _execute_langpro_request(
                    premises_list,
                    hypothesis,
                    resolved_endpoint,
                    parser,
                    ral,
                    kb_list,
                    senses,
                    strong_align,
                    intersective,
                    curl,
                    report,
                    resolved_timeout,
                    context=context,
                )
            )
            _LANGPRO_INFLIGHT[cache_key] = task

    try:
        response_text, error = await task
    finally:
        async with lock:
            if _LANGPRO_INFLIGHT.get(cache_key) is task:
                _LANGPRO_INFLIGHT.pop(cache_key, None)

    if response_text is not None:
        cache_backend.set(cache_key, response_text)
    return response_text, error


async def langpro_api_call(premises: list, hypothesis: str,
                           endpoint=None,
                           parser="easyccg", ral=200, kb=None, senses='all',
                           strong_align=True, intersective=True, curl=False, report=False,
                           timeout_seconds: Optional[float] = None,
                           context: Optional[AsyncRunContext] = None) -> LangProResult:
    """ Uses API call to a remote server to run LangPro prover
        and get parsed input sentecnes, tableau proof, and inference label.

        Cache keys preserve premise order but canonicalize KB relation order, so
        the same injected KB can be reused across different model runs.
    """
    settings = get_langpro_settings()
    resolved_endpoint = endpoint or settings.endpoint
    resolved_timeout = timeout_seconds if timeout_seconds is not None else settings.timeout_seconds

    premises_list = list(premises or [])
    kb_list = _normalize_kb_for_request(kb)

    cache_key = _make_langpro_cache_key(
        premises_list,
        hypothesis,
        parser,
        ral,
        kb_list,
        senses,
        strong_align,
        intersective,
        settings.prover_config_extra,
    )

    cache_backend = get_langpro_cache_backend()
    cached_response_text = cache_backend.get(cache_key)
    if cached_response_text is not None:
        return _parse_langpro_output(json.loads(cached_response_text))
    cached_response_text = _get_and_migrate_legacy_langpro_cache_entry(
        cache_backend,
        cache_key,
        premises_list,
        hypothesis,
        resolved_endpoint,
        parser,
        ral,
        kb_list,
        senses,
        strong_align,
        intersective,
        settings,
    )
    if cached_response_text is not None:
        return _parse_langpro_output(json.loads(cached_response_text))

    response_text, error = await _execute_langpro_request_for_cache(
        cache_key,
        cache_backend,
        premises_list,
        hypothesis,
        resolved_endpoint,
        parser,
        ral,
        kb_list,
        senses,
        strong_align,
        intersective,
        curl,
        report,
        resolved_timeout,
        context,
    )

    # Do not cache transient failures; they should be eligible for retry.
    if response_text is None:
        return LangProResult(label=NLILabel.UNKNOWN, error=error)

    return _parse_langpro_output(json.loads(response_text))
