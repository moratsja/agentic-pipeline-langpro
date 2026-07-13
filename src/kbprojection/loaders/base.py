import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Union

from ..models import NLIProblem, NLILabel
from ..settings import get_default_dataset_dir


class DatasetLoader(ABC):
    """
    Abstract base class for NLI dataset loaders.
    
    Subclasses must implement:
        - `_download()`: Downloads the dataset files.
        - `_get_splits()`: Returns a list of available splits.
        - `_get_file_path(split)`: Returns the file path for a given split.
        - `_parse_row(row, split)`: Parses a raw row into an NLIProblem.
        - `_iter_file(file_path)`: Iterates over raw rows in a file.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            dataset_name = self.__class__.__name__.lower().replace("loader", "")
            self.data_dir = get_default_dataset_dir(dataset_name)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage: split -> {id -> NLIProblem}
        self._data: Dict[str, Dict[str, NLIProblem]] = {}
        self._loaded_splits: Set[str] = set()

    # -------------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _download(self) -> None:
        """Downloads the dataset files to self.data_dir."""
        pass

    @abstractmethod
    def _get_splits(self) -> List[str]:
        """Returns a list of available splits (e.g., ['train', 'dev', 'test'])."""
        pass

    @abstractmethod
    def _get_file_path(self, split: str) -> Path:
        """Returns the file path for the given split."""
        pass

    @abstractmethod
    def _iter_file(self, file_path: Path) -> Iterator[dict]:
        """Iterates over raw rows/records in the file. Yields dicts."""
        pass

    @abstractmethod
    def _parse_row(self, row: dict, split: str) -> Optional[NLIProblem]:
        """Parses a raw row dict into an NLIProblem, or None to skip."""
        pass

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load(self, splits: Union[List[str], bool, None] = None) -> None:
        """
        Ensures data is downloaded and optionally loads splits into memory.

        Args:
            splits: Controls what to load into memory.
                - None (default): Only downloads if needed. No memory loading.
                - True: Loads all available splits into memory.
                - List[str]: Loads the specified splits into memory.
        """
        self._download()

        if splits is None:
            return

        if splits is True:
            splits_to_load = self._get_splits()
        else:
            splits_to_load = splits

        for split in splits_to_load:
            if split in self._loaded_splits:
                continue
            self._load_split_to_memory(split)

    def _load_split_to_memory(self, split: str) -> None:
        """Loads a single split into self._data."""
        file_path = self._get_file_path(split)
        if not file_path.exists():
            raise FileNotFoundError(f"File for split '{split}' not found: {file_path}")

        self._data[split] = {}
        for row in self._iter_file(file_path):
            problem = self._parse_row(row, split)
            if problem:
                self._data[split][problem.id] = problem

        self._loaded_splits.add(split)
        print(f"[{self.__class__.__name__}] Loaded {len(self._data[split])} problems from {split}.")

    def iter_problems(
        self, split: str = "dev", label_filter: Optional[Set[str]] = None
    ) -> Iterator[NLIProblem]:
        """
        Iterates over problems in the dataset.

        If the split is loaded in memory, iterates from memory.
        Otherwise, streams directly from file.

        Args:
            split: The dataset split to iterate over.
            label_filter: Optional set of labels to filter by.

        Yields:
            NLIProblem instances.
        """
        # Ensure data exists
        self._download()

        if split in self._loaded_splits:
            # Iterate from memory
            for problem in self._data[split].values():
                if label_filter and problem.gold_label not in label_filter:
                    continue
                yield problem
        else:
            # Stream from file
            file_path = self._get_file_path(split)
            if not file_path.exists():
                raise FileNotFoundError(f"File for split '{split}' not found: {file_path}")

            for row in self._iter_file(file_path):
                problem = self._parse_row(row, split)
                if problem:
                    if label_filter and problem.gold_label not in label_filter:
                        continue
                    yield problem

    def get_problem(self, key: str, split: str = "dev") -> NLIProblem:
        """
        Retrieves a specific problem by its ID/key.

        If the split is not in memory, loads it first.
        """
        if split not in self._loaded_splits:
            self.load(splits=[split])

        raw_key = str(key)
        normalized_key = self.normalize_problem_id(raw_key)
        if normalized_key not in self._data[split]:
            message = f"Key {raw_key!r} not found in {self.__class__.__name__} {split}"
            if normalized_key != raw_key:
                message += f" after normalization to {normalized_key!r}"
            raise KeyError(message)
        return self._data[split][normalized_key]

    def random_problem(
        self,
        split: str = "dev",
        label_filter: Optional[Set[str]] = None,
        exclude_keys: Optional[Set[str]] = None,
    ) -> Optional[NLIProblem]:
        """
        Returns a random problem matching the criteria.

        If the split is not in memory, loads it first to enable random access.
        """
        if split not in self._loaded_splits:
            self.load(splits=[split])

        candidates = []
        for k, prob in self._data[split].items():
            if exclude_keys and k in exclude_keys:
                continue
            if label_filter and prob.gold_label not in label_filter:
                continue
            candidates.append(prob)

        if not candidates:
            return None

        return random.choice(candidates)

    # -------------------------------------------------------------------------
    # Helper for label parsing (common across loaders)
    # -------------------------------------------------------------------------

    def normalize_problem_id(self, key: str) -> str:
        """Normalizes caller-provided problem IDs before lookup."""
        return str(key).strip()

    @staticmethod
    def _parse_label(label: str) -> NLILabel:
        """Parses a label string into an NLILabel enum."""
        label = label.upper()
        if label == "ENTAILMENT":
            return NLILabel.ENTAILMENT
        elif label == "CONTRADICTION":
            return NLILabel.CONTRADICTION
        elif label == "NEUTRAL":
            return NLILabel.NEUTRAL
        else:
            return NLILabel.UNKNOWN
