import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


DEFAULT_LANGPRO_ENDPOINT = "https://langpro.hum.uu.nl/langpro-api/prove/"
DEFAULT_LANGPRO_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
DEFAULT_LANGPRO_REPO = "https://github.com/kovvalsky/LangPro.git"
DEFAULT_LANGPRO_REF = "nl"
DEFAULT_EASYCCG_REPO = "https://github.com/mikelewis0/easyccg.git"
DEFAULT_EASYCCG_MODEL_SOURCE = (
    "https://bitbucket.org/yoavartzi/amr-resources/downloads/easyccg-model-rebank.tar.gz"
)
# Vendored layout: <repo root>/src/kbprojection/settings.py -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_truthy_env(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_prover_config_extra(value: Optional[str]) -> Tuple[str, ...]:
    """Comma/space-separated extra prover_config flags, e.g. 'no_kb,no_wn'."""
    if not value:
        return ()
    parts = [p.strip() for p in value.replace(" ", ",").split(",")]
    return tuple(p for p in parts if p)


def get_app_dir() -> Path:
    configured = os.environ.get("KBPROJECTION_APP_DIR")
    if configured:
        return Path(configured)

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "kbprojection"
        return Path.home() / "AppData" / "Local" / "kbprojection"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "kbprojection"
    return Path.home() / ".cache" / "kbprojection"


def get_default_dataset_root() -> Path:
    configured = os.environ.get("KBPROJECTION_DATA_DIR")
    return Path(configured) if configured else get_app_dir() / "datasets"


def get_default_dataset_dir(dataset_name: str) -> Path:
    normalized = (dataset_name or "data").strip().lower()
    return get_default_dataset_root() / normalized


def get_default_results_dir() -> Path:
    configured = os.environ.get("KBPROJECTION_RESULTS_DIR")
    return Path(configured) if configured else get_app_dir() / "results"


def get_default_cache_dir() -> Path:
    configured = os.environ.get("KBPROJECTION_CACHE_DIR")
    return Path(configured) if configured else get_app_dir() / "cache"


def get_default_vendor_dir() -> Path:
    configured = os.environ.get("KBPROJECTION_LANGPRO_VENDOR_DIR")
    return Path(configured) if configured else get_app_dir() / "vendor" / "LangPro"


def get_default_easyccg_vendor_dir() -> Path:
    configured = os.environ.get("KBPROJECTION_EASYCCG_VENDOR_DIR")
    return Path(configured) if configured else get_app_dir() / "vendor" / "easyccg"


def _repo_vendor_langpro_root() -> Path:
    return PROJECT_ROOT / "vendor" / "LangPro"


def _sibling_langpro_root() -> Path:
    return PROJECT_ROOT.parent / "LangPro"


def get_local_langpro_search_paths() -> List[Path]:
    paths: List[Path] = []
    env_root = os.environ.get("KBPROJECTION_LANGPRO_LOCAL_ROOT")
    if env_root:
        paths.append(Path(env_root))
    paths.extend(
        [
            _repo_vendor_langpro_root(),
            _sibling_langpro_root(),
            get_default_vendor_dir(),
        ]
    )
    return paths


def _run_git(args: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _format_git_error(command: str, result: subprocess.CompletedProcess) -> str:
    details = (result.stderr or result.stdout or "").strip()
    return f"{command} failed with code {result.returncode}: {details}"


def ensure_langpro_clone(destination: Path, repo_url: str, ref: str) -> Path:
    if destination.exists() and not (destination / ".git").exists():
        raise RuntimeError(
            f"LangPro clone destination exists but is not a git checkout: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        clone_branch = _run_git(["clone", "--branch", ref, repo_url, str(destination)])
        if clone_branch.returncode == 0:
            return destination

        clone_plain = _run_git(["clone", repo_url, str(destination)])
        if clone_plain.returncode != 0:
            raise RuntimeError(
                _format_git_error(
                    f"git clone --branch {ref} {repo_url} {destination}",
                    clone_branch,
                )
                + " | "
                + _format_git_error(f"git clone {repo_url} {destination}", clone_plain)
            )

    checkout = _run_git(["checkout", ref], cwd=destination)
    if checkout.returncode != 0:
        raise RuntimeError(_format_git_error(f"git checkout {ref}", checkout))

    return destination


def resolve_local_langpro_root(allow_clone: bool = False) -> Path:
    env_root = os.environ.get("KBPROJECTION_LANGPRO_LOCAL_ROOT")
    if env_root:
        return Path(env_root)

    paths = get_local_langpro_search_paths()
    for path in paths:
        if path.exists():
            return path

    vendor_root = get_default_vendor_dir()
    if allow_clone and _is_truthy_env(os.environ.get("KBPROJECTION_LANGPRO_AUTO_CLONE")):
        return ensure_langpro_clone(
            vendor_root,
            os.environ.get("KBPROJECTION_LANGPRO_REPO", DEFAULT_LANGPRO_REPO),
            os.environ.get("KBPROJECTION_LANGPRO_REF", DEFAULT_LANGPRO_REF),
        )

    return vendor_root


def format_local_langpro_missing_error() -> str:
    lines = ["Local LangPro checkout not found.", "Searched:"]
    env_root = os.environ.get("KBPROJECTION_LANGPRO_LOCAL_ROOT")
    if env_root:
        lines.append(f"- KBPROJECTION_LANGPRO_LOCAL_ROOT={env_root}")
    lines.extend(f"- {path}" for path in get_local_langpro_search_paths() if str(path) != env_root)
    lines.extend(
        [
            "",
            "Set KBPROJECTION_LANGPRO_LOCAL_ROOT, add vendor/LangPro, keep a sibling ../LangPro checkout,",
            "or enable explicit auto-clone with:",
            "KBPROJECTION_LANGPRO_AUTO_CLONE=1",
            f"KBPROJECTION_LANGPRO_REPO={os.environ.get('KBPROJECTION_LANGPRO_REPO', DEFAULT_LANGPRO_REPO)}",
            f"KBPROJECTION_LANGPRO_REF={os.environ.get('KBPROJECTION_LANGPRO_REF', DEFAULT_LANGPRO_REF)}",
        ]
    )
    return "\n".join(lines)


def _default_langpro_local_easyccg_dir() -> Path:
    return Path(
        os.environ.get(
            "KBPROJECTION_LANGPRO_LOCAL_EASYCCG_DIR",
            get_default_easyccg_vendor_dir(),
        )
    )


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _set_optional_path_env(name: str, value: Optional[Union[str, Path]]) -> None:
    if value is not None:
        os.environ[name] = str(value)


def enable_local(
    local_root: Optional[Union[str, Path]] = None,
    *,
    auto_clone: bool = False,
    vendor_dir: Optional[Union[str, Path]] = None,
    repo_url: str = DEFAULT_LANGPRO_REPO,
    ref: str = DEFAULT_LANGPRO_REF,
    swipl: str = "swipl",
    easyccg_dir: Optional[Union[str, Path]] = None,
    easyccg_spacy_model: str = "en_core_web_sm",
    easyccg_model_source: str = DEFAULT_EASYCCG_MODEL_SOURCE,
) -> "LangProSettings":
    """
    Configure the current Python process to use local LangPro.

    This is a convenience wrapper around the KBPROJECTION_LANGPRO_* environment
    variables. It is intended for notebooks and scripts that want an explicit
    local-mode switch without manually setting each variable.
    """
    os.environ["KBPROJECTION_LANGPRO_ENDPOINT"] = "local://auto"
    os.environ["KBPROJECTION_LANGPRO_AUTO_CLONE"] = "1" if auto_clone else "0"
    os.environ["KBPROJECTION_LANGPRO_REPO"] = repo_url
    os.environ["KBPROJECTION_LANGPRO_REF"] = ref
    os.environ["KBPROJECTION_LANGPRO_LOCAL_SWIPL"] = swipl
    os.environ["KBPROJECTION_LANGPRO_LOCAL_EASYCCG_SPACY_MODEL"] = easyccg_spacy_model
    os.environ["KBPROJECTION_LANGPRO_LOCAL_EASYCCG_MODEL_SOURCE"] = easyccg_model_source

    _set_optional_path_env("KBPROJECTION_LANGPRO_LOCAL_ROOT", local_root)
    _set_optional_path_env("KBPROJECTION_LANGPRO_VENDOR_DIR", vendor_dir)
    _set_optional_path_env("KBPROJECTION_LANGPRO_LOCAL_EASYCCG_DIR", easyccg_dir)

    return get_langpro_settings()


@dataclass(frozen=True)
class LangProSettings:
    app_dir: Path
    dataset_root: Path
    results_dir: Path
    cache_dir: Path
    endpoint: str
    timeout_seconds: float
    prover_config_extra: Tuple[str, ...]
    cache_backend: str
    cache_path: Path
    local_root: Path
    local_vendor_root: Path
    local_auto_clone: bool
    local_repo_url: str
    local_repo_ref: str
    local_swipl: str
    local_easyccg_dir: Path
    local_easyccg_spacy_model: str
    local_easyccg_model_source: str


def get_langpro_settings() -> LangProSettings:
    app_dir = get_app_dir()
    dataset_root = get_default_dataset_root()
    results_dir = get_default_results_dir()
    cache_dir = get_default_cache_dir()
    vendor_root = get_default_vendor_dir()
    return LangProSettings(
        app_dir=app_dir,
        dataset_root=dataset_root,
        results_dir=results_dir,
        cache_dir=cache_dir,
        endpoint=os.environ.get("KBPROJECTION_LANGPRO_ENDPOINT", DEFAULT_LANGPRO_ENDPOINT),
        timeout_seconds=_get_float_env(
            "KBPROJECTION_LANGPRO_TIMEOUT_SECONDS",
            DEFAULT_LANGPRO_TIMEOUT_SECONDS,
        ),
        prover_config_extra=_parse_prover_config_extra(
            os.environ.get("KBPROJECTION_LANGPRO_PROVER_CONFIG_EXTRA")
        ),
        cache_backend=os.environ.get("KBPROJECTION_LANGPRO_CACHE_BACKEND", "sqlite").strip().lower(),
        cache_path=Path(
            os.environ.get(
                "KBPROJECTION_LANGPRO_CACHE_PATH",
                app_dir / "langpro_cache.sqlite3",
            )
        ),
        local_root=resolve_local_langpro_root(allow_clone=False),
        local_vendor_root=vendor_root,
        local_auto_clone=_is_truthy_env(os.environ.get("KBPROJECTION_LANGPRO_AUTO_CLONE")),
        local_repo_url=os.environ.get("KBPROJECTION_LANGPRO_REPO", DEFAULT_LANGPRO_REPO),
        local_repo_ref=os.environ.get("KBPROJECTION_LANGPRO_REF", DEFAULT_LANGPRO_REF),
        local_swipl=os.environ.get("KBPROJECTION_LANGPRO_LOCAL_SWIPL", "swipl"),
        local_easyccg_dir=_default_langpro_local_easyccg_dir(),
        local_easyccg_spacy_model=os.environ.get(
            "KBPROJECTION_LANGPRO_LOCAL_EASYCCG_SPACY_MODEL",
            "en_core_web_sm",
        ),
        local_easyccg_model_source=os.environ.get(
            "KBPROJECTION_LANGPRO_LOCAL_EASYCCG_MODEL_SOURCE",
            DEFAULT_EASYCCG_MODEL_SOURCE,
        ),
    )


def get_download_timeout_seconds() -> float:
    return _get_float_env(
        "KBPROJECTION_DOWNLOAD_TIMEOUT_SECONDS",
        DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    )
