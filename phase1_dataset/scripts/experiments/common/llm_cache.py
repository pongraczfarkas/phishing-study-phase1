import json
import hashlib
from pathlib import Path
from threading import Lock

CACHE_PATH = Path("cache/gpt_cache.json")

_lock = Lock()


def ensure_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.exists():
        CACHE_PATH.write_text("{}", encoding="utf-8")


def load_cache():
    ensure_cache()
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache):
    with _lock:
        CACHE_PATH.write_text(
            json.dumps(cache, indent=2),
            encoding="utf-8"
        )


def make_key(text: str, model: str, prompt_version: str):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}|{prompt_version}|{h}"