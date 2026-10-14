from __future__ import annotations

import base64
import hashlib
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..registry import GENERATORS

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "seed")
FIXED_PAYLOAD: Dict[str, Any] = {"min_p": 0.0}
SEED_MAX = 2 ** 31


def normalize_sampling(s: Mapping[str, Any]) -> Dict[str, Any]:
    s = dict(s)
    missing = [k for k in SAMPLING_FIELDS if k not in s or s[k] is None]
    if missing:
        raise ValueError(f"sampling is missing {missing}; every field is sent explicitly")
    extra = sorted(set(s) - set(SAMPLING_FIELDS) - {"n"})
    if extra:
        raise ValueError(f"unknown sampling fields {extra} (vLLM would silently ignore them)")
    out = {"temperature": float(s["temperature"]), "top_p": float(s["top_p"]),
           "top_k": int(s["top_k"]), "repetition_penalty": float(s["repetition_penalty"]),
           "max_tokens": int(s["max_tokens"]), "seed": int(s["seed"]), "n": int(s.get("n", 1))}
    if out["temperature"] < 0:
        raise ValueError("temperature must be >= 0")
    if not 0.0 < out["top_p"] <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if out["top_k"] <= 0:
        out["top_k"] = -1
    elif out["top_k"] == 1 and out["temperature"] > 0:
        raise ValueError("top_k=1 is greedy decoding; temperature would be inert (eq:temperature)")
    if out["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be > 0 (1.0 = off)")
    if out["max_tokens"] < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 0 <= out["seed"] < SEED_MAX:
        raise ValueError(f"seed must be in [0, 2^31); got {out['seed']}")
    if out["n"] < 1:
        raise ValueError("n must be >= 1")
    return out


@dataclass
class GenRequest:
    key: str
    system: str
    question: str
    image_paths: List[str]
    sampling: Dict[str, Any]
    prefix: Optional[str] = None


@dataclass
class GenResult:
    key: str
    text: Optional[str]
    finish_reason: Optional[str]
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "GenResult":
        return GenResult(key=d["key"], text=d.get("text"), finish_reason=d.get("finish_reason"),
                         usage=dict(d.get("usage") or {}), error=d.get("error"),
                         choices=list(d.get("choices") or []), meta=dict(d.get("meta") or {}))


class ImageError(ValueError):
    pass


@lru_cache(maxsize=1024)
def _png_data_url(path: str) -> str:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@lru_cache(maxsize=8192)
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _where(key: str, rank: int, path: str) -> str:
    return f"request {key!r}, image {rank + 1}"


def image_data_url(path: str, key: str = "?", rank: int = 0) -> str:
    if not os.path.isfile(path):
        raise ImageError(f"missing slice: {_where(key, rank, path)}")
    try:
        return _png_data_url(path)
    except Exception as e:
        raise ImageError(f"undecodable slice ({type(e).__name__}): {_where(key, rank, path)}") from None


def image_sha256(path: str, key: str = "?", rank: int = 0) -> str:
    try:
        return _file_sha256(path)
    except OSError as e:
        raise ImageError(f"unreadable slice ({type(e).__name__}): {_where(key, rank, path)}") from None


def build_messages(system: str, question: str, image_urls: Sequence[str],
                   prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    content.append({"type": "text", "text": question})
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": content})
    if prefix is not None:
        msgs.append({"role": "assistant", "content": prefix})
    return msgs


def build_payload(served_model: str, req: GenRequest) -> Dict[str, Any]:
    s = normalize_sampling(req.sampling)
    urls = [image_data_url(p, req.key, i) for i, p in enumerate(req.image_paths)]
    body: Dict[str, Any] = {"model": served_model,
                            "messages": build_messages(req.system, req.question, urls, req.prefix),
                            **s, **FIXED_PAYLOAD}
    if req.prefix is not None:
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    return body


class _HTTPError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


@GENERATORS.register("vllm")
class VLLMGenerator:
    def __init__(self, base_urls: Sequence[str] | str, served_model: str, concurrency: int = 64,
                 timeout: float = 900, retries: int = 3, backoff: float = 1.5):
        if isinstance(base_urls, str):
            base_urls = base_urls.split(",")
        self.urls = [u.strip().rstrip("/") for u in base_urls if u and u.strip()]
        if not self.urls:
            raise ValueError("VLLMGenerator needs at least one base url")
        if not served_model:
            raise ValueError("served_model is required (the server's --served-model-name)")
        if int(retries) < 1 or int(concurrency) < 1:
            raise ValueError(f"retries ({retries}) and concurrency ({concurrency}) must be >= 1")
        self.served_model = served_model
        self.concurrency = int(concurrency)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self._lock = threading.Lock()
        self._tl = threading.local()
        self._inflight = [0] * len(self.urls)
        self.stats: Dict[str, int] = {"ok": 0, "retry": 0, "fail": 0}
        self.tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def describe(self) -> Dict[str, Any]:
        return {"backend": "vllm", "served_model": self.served_model, "replicas": len(self.urls),
                "fixed_payload": dict(FIXED_PAYLOAD), "concurrency": self.concurrency}

    def _session(self):
        s = getattr(self._tl, "s", None)
        if s is None:
            import requests

            s = requests.Session()
            s.mount("http://", requests.adapters.HTTPAdapter(
                pool_connections=64, pool_maxsize=64, max_retries=0))
            self._tl.s = s
        return s

    def _pick(self, avoid: Optional[int] = None) -> int:
        with self._lock:
            cand = [j for j in range(len(self.urls)) if j != avoid] or list(range(len(self.urls)))
            i = min(cand, key=lambda j: (self._inflight[j], j))
            self._inflight[i] += 1
            return i

    def _release(self, i: int) -> None:
        with self._lock:
            self._inflight[i] -= 1

    def _bump(self, k: str, usage: Optional[Mapping[str, int]] = None) -> None:
        with self._lock:
            self.stats[k] += 1
            for u in self.tokens:
                self.tokens[u] += int((usage or {}).get(u, 0) or 0)

    def _post_once(self, i: int, body: Mapping[str, Any]) -> Dict[str, Any]:
        r = self._session().post(self.urls[i] + "/v1/chat/completions", json=body,
                                 timeout=(30, self.timeout))
        if r.status_code != 200:
            raise _HTTPError(r.status_code, f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _call(self, key: str, body: Mapping[str, Any]) -> GenResult:
        last, avoid, t0 = "", None, time.time()
        for attempt in range(1, self.retries + 1):
            i = self._pick(avoid)
            try:
                js = self._post_once(i, body)
                choices = [{"text": c["message"]["content"], "finish_reason": c.get("finish_reason")}
                           for c in sorted(js["choices"], key=lambda c: c.get("index", 0))]
                if not choices or not all(isinstance(c["text"], str) for c in choices):
                    raise ValueError("response has a choice without text content")
                usage = {k: int(v) for k, v in (js.get("usage") or {}).items() if isinstance(v, int)}
                self._bump("ok", usage)
                return GenResult(key=key, text=choices[0]["text"], finish_reason=choices[0]["finish_reason"],
                                 usage=usage, choices=choices,
                                 meta={"replica": i, "attempts": attempt,
                                       "latency_s": round(time.time() - t0, 3)})
            except _HTTPError as e:
                last = str(e)
                if 400 <= e.status < 500 and e.status not in (408, 429):
                    break
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                self._release(i)
            avoid = i
            if attempt < self.retries:
                self._bump("retry")
                time.sleep(self.backoff * attempt)
        self._bump("fail")
        return GenResult(key=key, text=None, finish_reason=None, error=last or "unknown error",
                         meta={"attempts": attempt, "latency_s": round(time.time() - t0, 3)})

    def generate(self, requests: Sequence[GenRequest],
                 on_result: Optional[Callable[[GenResult], None]] = None) -> List[GenResult]:
        reqs = list(requests)
        keys = [r.key for r in reqs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate request keys")
        bodies = [build_payload(self.served_model, r) for r in reqs]
        out: List[Optional[GenResult]] = [None] * len(reqs)
        if not reqs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(self.concurrency, len(reqs)))) as ex:
            futs = {ex.submit(self._call, r.key, b): j for j, (r, b) in enumerate(zip(reqs, bodies))}
            try:
                for f in as_completed(futs):
                    res = f.result()
                    out[futs[f]] = res
                    if on_result is not None:
                        on_result(res)
            except BaseException:
                for f in futs:
                    f.cancel()
                raise
        return [r for r in out if r is not None]

    def health(self) -> List[bool]:
        ok = []
        for u in self.urls:
            try:
                ok.append(self._session().get(u + "/health", timeout=5).status_code == 200)
            except Exception:
                ok.append(False)
        return ok

    def wait_ready(self, timeout: float = 900, poll: float = 5) -> None:
        t_end = time.time() + timeout
        while True:
            h = self.health()
            if all(h):
                return
            if time.time() > t_end:
                raise TimeoutError(f"replicas not ready after {timeout}s: "
                                   f"{[u for u, x in zip(self.urls, h) if not x]}")
            time.sleep(poll)


def serve_urls(name: str, serve_root: str) -> List[str]:
    d = Path(serve_root) / name
    urls = []
    for f in sorted(d.glob("replica_*.pid")):
        try:
            pid = int(f.read_text().strip())
            os.kill(pid, 0)
            cmdline = Path(f"/proc/{pid}/cmdline")
            if cmdline.exists() and b"vllm" not in cmdline.read_bytes():
                continue
        except (ValueError, OSError):
            continue
        urls.append(f"http://127.0.0.1:{f.stem.split('_', 1)[1]}")
    return urls
