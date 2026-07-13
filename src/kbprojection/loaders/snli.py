import json
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .base import DatasetLoader
from ..models import NLIProblem
from ..downloads import download_file, extract_zip


class SNLILoader(DatasetLoader):
    """
    Loader for the SNLI dataset.
    Downloads from Stanford NLP if not present.
    """

    URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    SPLITS = ["train", "dev", "test"]

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        super().__init__(data_dir)

    def normalize_problem_id(self, key: str) -> str:
        return str(key).strip().replace(" ", "_")

    def _get_splits(self) -> List[str]:
        return self.SPLITS

    def _get_file_path(self, split: str) -> Path:
        # SNLI extracts to a subdirectory
        possible_paths = [
            self.data_dir / "snli_1.0" / f"snli_1.0_{split}.jsonl",
            self.data_dir / f"snli_1.0_{split}.jsonl",
        ]
        for p in possible_paths:
            if p.exists():
                return p
        # Return the expected path for error messaging
        return possible_paths[0]

    def _download(self) -> None:
        """Downloads and extracts SNLI."""
        target_dir = self.data_dir / "snli_1.0"

        if target_dir.exists():
            return

        zip_path = self.data_dir / "snli_1.0.zip"
        print(f"[SNLILoader] Downloading SNLI...")
        download_file(self.URL, zip_path)
        extract_zip(zip_path, self.data_dir)

    def _iter_file(self, file_path: Path) -> Iterator[dict]:
        """Iterates over JSONL lines in an SNLI file."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _parse_row(self, row: dict, split: str) -> Optional[NLIProblem]:
        """Parses an SNLI JSON row into an NLIProblem."""
        pair_id = row.get("pairID")
        gold_label = row.get("gold_label")

        if not pair_id or gold_label == "-":
            return None

        return NLIProblem(
            id=pair_id,
            premises=[row.get("sentence1", "")],
            hypothesis=row.get("sentence2", ""),
            gold_label=self._parse_label(gold_label),
            dataset="snli",
            split=split,
            original_data=row,
        )
