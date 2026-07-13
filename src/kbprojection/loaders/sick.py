import csv
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .base import DatasetLoader
from ..models import NLIProblem
from ..downloads import download_file


class SICKLoader(DatasetLoader):
    """
    Loader for the SICK dataset.
    Downloads from a reliable mirror if not present.
    """

    BASE_URL = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/"
    FILES = {
        "train": "SICK_train.txt",
        "dev": "SICK_trial.txt",
        "test": "SICK_test_annotated.txt",
    }

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        super().__init__(data_dir)

    def _get_splits(self) -> List[str]:
        return list(self.FILES.keys())

    def _get_file_path(self, split: str) -> Path:
        filename = self.FILES.get(split)
        if not filename:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.FILES.keys())}")
        return self.data_dir / filename

    def _download(self) -> None:
        """Downloads the SICK dataset files."""
        for split, filename in self.FILES.items():
            url = self.BASE_URL + filename
            target = self.data_dir / filename
            if not target.exists():
                print(f"[SICKLoader] Downloading {filename}...")
                download_file(url, target)

    def _iter_file(self, file_path: Path) -> Iterator[dict]:
        """Iterates over rows in a SICK TSV file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                yield row

    def _parse_row(self, row: dict, split: str) -> Optional[NLIProblem]:
        """Parses a SICK row into an NLIProblem."""
        pair_id = row.get("pair_ID") or row.get("id")
        sent_a = row.get("sentence_A")
        sent_b = row.get("sentence_B")
        label = row.get("entailment_judgment") or row.get("entailment_label")

        if not pair_id or not label:
            return None

        return NLIProblem(
            id=str(pair_id),
            premises=[sent_a] if sent_a else [],
            hypothesis=sent_b,
            gold_label=self._parse_label(label),
            dataset="sick",
            split=split,
            original_data=row,
        )
