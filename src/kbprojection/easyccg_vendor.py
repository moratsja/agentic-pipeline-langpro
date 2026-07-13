import shutil
from pathlib import Path
from typing import Optional, Union

from .downloads import download_file
from .local_easyccg import easyccg_is_available
from .settings import (
    DEFAULT_EASYCCG_MODEL_SOURCE,
    DEFAULT_EASYCCG_REPO,
    get_default_easyccg_vendor_dir,
    get_download_timeout_seconds,
    _format_git_error,
    _run_git,
)


def _copy_model_dir(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(source, destination)


def _find_easyccg_model_dir(root: Path) -> Optional[Path]:
    expected_files = {"bias", "binaryRules", "categories", "classifier", "unaryRules"}
    if root.is_dir() and (
        root.name in {"model_rebank", "rebank", "easyccg-model-rebank"}
        or expected_files.issubset({path.name for path in root.iterdir()})
    ):
        return root

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if path.name in {"model_rebank", "rebank", "easyccg-model-rebank"}:
            return path
        if expected_files.issubset({child.name for child in path.iterdir()}):
            return path
    return None


def _download_or_unpack_model(source: str, destination: Path) -> None:
    source_path = Path(source)
    work_dir = destination / "model_download"
    work_dir.mkdir(parents=True, exist_ok=True)

    if source.startswith(("http://", "https://")):
        archive_path = work_dir / Path(source).name
        download_file(source, archive_path)
        shutil.unpack_archive(str(archive_path), str(work_dir))
        model_dir = _find_easyccg_model_dir(work_dir)
    elif source_path.is_file():
        shutil.unpack_archive(str(source_path), str(work_dir))
        model_dir = _find_easyccg_model_dir(work_dir)
    elif source_path.is_dir():
        model_dir = _find_easyccg_model_dir(source_path)
    else:
        raise FileNotFoundError(f"EasyCCG model source not found: {source}")

    if model_dir is None:
        raise RuntimeError(f"Could not find model_rebank or rebank under {work_dir}")

    _copy_model_dir(model_dir, destination / "model_rebank")


def install_local_easyccg(
    destination: Optional[Union[str, Path]] = None,
    *,
    repo_url: str = DEFAULT_EASYCCG_REPO,
    model_source: str = DEFAULT_EASYCCG_MODEL_SOURCE,
) -> Path:
    """
    Install EasyCCG into the kbprojection app-data vendor area.

    The resulting directory contains `easyccg.jar` and `model_rebank`, which is
    the layout expected by kbprojection's local LangPro raw-text fallback.
    """
    target = Path(destination) if destination is not None else get_default_easyccg_vendor_dir()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not (target / "easyccg.jar").exists():
        if target.exists() and any(target.iterdir()) and not (target / ".git").exists():
            raise RuntimeError(
                f"EasyCCG destination exists but does not contain easyccg.jar: {target}"
            )

        if not target.exists():
            clone = _run_git(["clone", "--depth", "1", repo_url, str(target)])
            if clone.returncode != 0:
                raise RuntimeError(_format_git_error(f"git clone {repo_url} {target}", clone))
        else:
            pull = _run_git(["pull", "--ff-only"], cwd=target)
            if pull.returncode != 0:
                raise RuntimeError(_format_git_error("git pull --ff-only", pull))

    if not (target / "model_rebank").exists():
        _download_or_unpack_model(model_source, target)

    if not easyccg_is_available(target):
        raise RuntimeError(
            f"EasyCCG install incomplete at {target}; expected easyccg.jar and model_rebank."
        )

    return target
