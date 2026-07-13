import shutil
import zipfile
import requests
from pathlib import Path
from typing import Optional

from .settings import get_download_timeout_seconds

def download_file(url: str, destination: Path) -> Path:
    """
    Downloads a file from a URL to a destination path.
    """
    print(f"[Download] Downloading {url} to {destination}...")
    destination.parent.mkdir(parents=True, exist_ok=True)

    timeout_seconds = get_download_timeout_seconds()

    with requests.get(url, stream=True, timeout=timeout_seconds) as r:
        r.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"[Download] Finished: {destination}")
    return destination

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extracts a zip file to a directory, filtering out potentially problematic files.
    """
    print(f"[Extract] Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Skip MACOSX and hidden files
            if member.filename.startswith('__MACOSX') or member.filename.startswith('.'):
                continue
            
            # Skip files with invalid characters for Windows (e.g. Icon\r)
            if '\r' in member.filename:
                print(f"[Extract] Skipping invalid filename: {member.filename}")
                continue
                
            try:
                zip_ref.extract(member, extract_to)
            except OSError as e:
                print(f"[Extract] Warning: Failed to extract {member.filename}: {e}")
                
    print(f"[Extract] Finished.")

_DOWNLOADED_NLTK = set()

def check_nltk(package: str):
    try:
        import nltk
        if package not in _DOWNLOADED_NLTK:
            nltk.download(package, quiet=True)
            _DOWNLOADED_NLTK.add(package)
    except Exception as e:
        print(f"[NLTK] Error: {e}")
