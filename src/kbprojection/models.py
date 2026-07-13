from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator
from enum import Enum

class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    UNKNOWN = "-"

class NLIProblem(BaseModel):
    """
    Represents a single NLI problem (Premises-Hypothesis pair).
    Normalized structure to be used across different datasets.
    """
    id: str
    premises: List[str]  # List of premise sentences
    hypothesis: str
    gold_label: NLILabel
    dataset: str
    split: str
    original_data: Optional[Dict[str, Any]] = Field(default=None, description="Original raw data from the dataset wrapper")



class LLMKBInjection(BaseModel):
    """
    Raw KB injection string from LLM response.
    Matches the prompt expectation of 'predicate(arg1, arg2)' string.
    """
    KB_injection: str = Field(description = "One single KB injection of style: disj(work, rest) or isa_wn(apple, fruit).")

class LLMKBResponse(BaseModel):
    """
    Structure for LLM output parsing.
    """
    output: List[LLMKBInjection] = Field(description = "List of KB injections.")

class LangProResult(BaseModel):
    """
    Result returned by the LangPro prover.
    """
    label: NLILabel
    kb: Any = None
    ccg_trees: List[Any] = Field(default_factory=list)
    ccg_terms: List[Any] = Field(default_factory=list)
    terms: List[Any] = Field(default_factory=list)
    llfs: List[Any] = Field(default_factory=list)
    proofs: Dict[str, Any] = Field(default_factory=dict)
    proof_info: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Raw LangPro proofs[*].info lists (e.g. contains 'closed').",
    )
    error: Optional[str] = None

class ExperimentStepStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ERROR = "error"

class ExperimentStatus(str, Enum):
    UNKNOWN = "unknown"
    BASELINE_SOLVED = "baseline_solved"
    BASELINE_PROVER_FAILED = "baseline_prover_failed"
    KB_GENERATION_FAILED = "kb_generation_failed"
    KB_GENERATION_EMPTY = "kb_generation_empty"
    KB_NORMALISATION_EMPTY = "kb_normalisation_empty"
    RAW_KB_SOLVED = "raw_kb_solved"
    RAW_KB_NOT_SOLVED = "raw_kb_not_solved"
    NORMALISED_KB_SOLVED = "normalised_kb_solved"
    NORMALISED_KB_NOT_SOLVED = "normalised_kb_not_solved"
    NORMALISED_KB_PROVER_FAILED = "normalised_kb_prover_failed"
    KB_NOT_SOLVED = "kb_not_solved"


class ExperimentResult(BaseModel):
    """
    The result of running the projection pipeline on a single problem.
    """
    problem: NLIProblem
    
    # Step 1: No KB
    pred_no_kb: Optional[NLILabel] = None
    status_no_kb: ExperimentStepStatus = ExperimentStepStatus.PENDING
    
    # Step 2: Generation
    kb_raw: Optional[List[str]] = None # List of strings as returned by LLM
    llm_error: Optional[str] = None
    llm_output_raw: Optional[str] = None
    
    # Step 3: Normalisation
    kb_filtered: Optional[List[str]] = None # Normalised list of strings ready for LangPro
    
    # Step 4: With raw KB (unfiltered)
    pred_with_raw_kb: Optional[NLILabel] = None
    status_with_raw_kb: ExperimentStepStatus = ExperimentStepStatus.PENDING
    
    # Step 5: With normalised KB
    pred_with_kb: Optional[NLILabel] = None
    status_with_kb: ExperimentStepStatus = ExperimentStepStatus.PENDING
    
    # Provenance details
    kb_details: Optional[List["KBResult"]] = None

    # Ablation results
    essential_kb: Optional[List[str]] = None  # Best minimal subset (by token count)
    ablation_subsets: Optional[List[List[str]]] = None  # All minimal sufficient subsets
    ablation_results: Optional[Dict[str, NLILabel]] = None  # Detailed log of tested subsets

    # Which KB type solved the problem
    fixed_by: Optional[str] = None  # "raw_kb", "normalised_kb", "both", or None

    # Prover call history
    prover_calls: Optional[List["LangProResult"]] = None

    # Overall Outcome
    final_status: ExperimentStatus = ExperimentStatus.UNKNOWN

    @field_validator("final_status", mode="before")
    @classmethod
    def _migrate_legacy_final_status(cls, value: Any) -> Any:
        legacy_statuses = {
            "already_correct": ExperimentStatus.BASELINE_SOLVED,
            "fixed": ExperimentStatus.NORMALISED_KB_SOLVED,
            "fixed_raw_kb": ExperimentStatus.RAW_KB_SOLVED,
            "still_wrong": ExperimentStatus.KB_NOT_SOLVED,
            "still_wrong_raw_kb": ExperimentStatus.RAW_KB_NOT_SOLVED,
            "error_no_kb": ExperimentStatus.BASELINE_PROVER_FAILED,
            "error_with_kb": ExperimentStatus.NORMALISED_KB_PROVER_FAILED,
            "llm_error": ExperimentStatus.KB_GENERATION_FAILED,
            "empty_kb_after_filter": ExperimentStatus.KB_NORMALISATION_EMPTY,
            "normalised_kb_empty": ExperimentStatus.KB_NORMALISATION_EMPTY,
            "normalised_kb_failed": ExperimentStatus.NORMALISED_KB_PROVER_FAILED,
            "no_kb_generated": ExperimentStatus.KB_GENERATION_EMPTY,
        }
        return legacy_statuses.get(value, value)


class TestMode(str, Enum):
    """
    Controls which KB injection stages to test.
    """
    NO_KB = "no_kb"           # Only test without KB
    RAW_KB = "raw_kb"         # Test with raw LLM output (unfiltered)
    NORMALISED_KB = "normalised"  # Test with normalised KB only
    FILTERED_KB = "filtered"      # Deprecated alias for normalised KB only
    BOTH = "both"                 # Test both raw and normalised KB
    FULL = "full"             # Full pipeline (default, same as current behavior)


class ProblemConfig(BaseModel):
    """
    Configuration for processing a single NLI problem.
    Encapsulates all options for the KB injection pipeline.
    """
    # LLM settings
    llm_provider: str = "openai"
    model: str = "gpt-5-mini"
    prompt_style: str = "icl"
    
    # Processing options
    post_process: bool = True
    
    # Test mode
    test_mode: TestMode = TestMode.FULL
    
    # Ablation
    run_ablation: bool = False
    
    # Output options
    verbose: bool = True

class KBResult(BaseModel):
    """
    Detailed result of a KB injection with provenance.
    """
    relation: str
    provenance: str = "llm"  # "llm", "post_process", "derived_swap", "derived_diff"
    original_text: Optional[str] = None

    def __str__(self):
        return self.relation
