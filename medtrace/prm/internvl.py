from __future__ import annotations

import glob
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .aggregate import KAPPA
from .layouts import (IMG_CONTEXT, NEG_TOKEN, POS_TOKEN, PRMPrompt, SlotMatcher, TrainMatchedLayout,
                      expand_image_tokens)

IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PRMImageError(OSError):
    pass


def load_remote_code_vlm(model_path: str, dtype: Any, **kwargs):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    ref = getattr(cfg, "auto_map", {}).get("AutoModel")
    if ref is None:
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=dtype, **kwargs)
    cls = get_class_from_dynamic_module(ref, model_path)
    if not hasattr(cls, "all_tied_weights_keys") and not getattr(cls, "_medtrace_tied_shim", False):
        original_init = cls.__init__

        def _init(self, *args, **kw):
            original_init(self, *args, **kw)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        cls.__init__ = _init
        cls._medtrace_tied_shim = True
    return cls.from_pretrained(model_path, trust_remote_code=True, dtype=dtype, **kwargs)


def checkpoint_files(weights_path: str) -> List[str]:
    one = os.path.join(weights_path, "model.safetensors")
    if os.path.isfile(one):
        return [one]
    shards = sorted(glob.glob(os.path.join(weights_path, "model*.safetensors")))
    if shards:
        return shards
    pt = os.path.join(weights_path, "pytorch_model.bin")
    return [pt] if os.path.isfile(pt) else []


def _sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def overlay_checkpoint(model, weights_path: str) -> Dict[str, Any]:
    import torch

    files = checkpoint_files(weights_path)
    if not files:
        return {"loaded": False, "reason": f"no checkpoint file in {weights_path}"}
    sd: Dict[str, Any] = {}
    for f in files:
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd.update(load_file(f))
        else:
            sd.update(torch.load(f, map_location="cpu"))
    msd = model.state_dict()
    shared = sorted(k for k in sd if k in msd and tuple(sd[k].shape) == tuple(msd[k].shape))
    probes = shared[:: max(1, len(shared) // 24)]
    before = {k: msd[k].detach().float().clone() for k in probes}
    if shared:
        model.load_state_dict({k: sd[k] for k in shared}, strict=False)
    after = model.state_dict()
    rel, moved = [], 0
    for k, b in before.items():
        d = (after[k].detach().float() - b).abs().mean().item()
        rel.append(d / (b.abs().mean().item() + 1e-12))
        moved += int(d > 0)
    return {
        "loaded": bool(shared),
        "files": [{"name": os.path.basename(f), "bytes": os.path.getsize(f), "sha256": _sha256(f)} for f in files],
        "checkpoint_tensors": len(sd), "model_tensors": len(msd), "matched": len(shared),
        "unexpected": sorted(k for k in sd if k not in msd)[:8],
        "n_unexpected": sum(1 for k in sd if k not in msd),
        "n_shape_mismatch": sum(1 for k in sd if k in msd and tuple(sd[k].shape) != tuple(msd[k].shape)),
        "n_missing": sum(1 for k in msd if k not in sd),
        "probed": len(before), "probes_changed": moved,
        "mean_relative_change": (sum(rel) / len(rel)) if rel else 0.0,
    }


def _proof_problem(proof: Dict[str, Any]) -> Optional[str]:
    if not proof.get("loaded"):
        return proof.get("reason") or "no tensor of the checkpoint matches the model by name and shape"
    for k in ("n_unexpected", "n_shape_mismatch", "n_missing"):
        if proof.get(k):
            return f"checkpoint/model mismatch: {k}={proof[k]} (matched {proof['matched']}/{proof['model_tensors']})"
    if proof.get("probes_changed", 0) == 0:
        return "the checkpoint moved none of the probed tensors (it equals the base model)"
    return None


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num: int = 1, max_num: int = 12, image_size: int = IMAGE_SIZE,
                       use_thumbnail: bool = False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                            for j in range(1, n + 1) if min_num <= i * j <= max_num},
                           key=lambda x: x[0] * x[1])
    ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    cols = target_width // image_size
    tiles = [resized.crop(((i % cols) * image_size, (i // cols) * image_size,
                           ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size))
             for i in range(blocks)]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class TileLoader:
    def __init__(self, cache_size: int = 64):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self._cache: "OrderedDict[Tuple[str, int], Any]" = OrderedDict()
        self._size = cache_size

    def __call__(self, path: str, max_tiles: int):
        import torch
        from PIL import Image

        key = (path, int(max_tiles))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if not os.path.isfile(path):
            raise PRMImageError("missing PRM input image")
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
            tiles = dynamic_preprocess(image, image_size=IMAGE_SIZE, use_thumbnail=True, max_num=int(max_tiles))
            pv = torch.stack([self._transform(t) for t in tiles])
        except Exception as e:
            raise PRMImageError(f"undecodable PRM input image ({type(e).__name__})") from None
        self._cache[key] = pv
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return pv


@dataclass
class ScoreRequest:
    key: str
    system: str
    question: str
    steps: List[str]
    image_paths: List[str]
    group: Optional[str] = None


class PRMScorer:
    def __init__(self, model_path: str, weights_path: Optional[str], device: str = "cuda:0",
                 allow_untrained: bool = False, max_tiles: Optional[int] = None,
                 max_images: Optional[int] = None, batch_size: int = 1):
        self.layout = TrainMatchedLayout(max_tiles=max_tiles, max_images=max_images)
        import torch
        from transformers import AutoTokenizer

        self.batch_size = max(1, int(batch_size))
        self.model_path, self.device = str(model_path), device
        self._torch = torch
        self.dtype = torch.bfloat16

        model = load_remote_code_vlm(self.model_path, self.dtype, low_cpu_mem_usage=True, use_flash_attn=False)
        given = bool(weights_path) and os.path.isdir(str(weights_path)) \
            and os.path.abspath(str(weights_path)) != os.path.abspath(self.model_path)
        self.proof = (overlay_checkpoint(model, str(weights_path)) if given
                      else {"loaded": False, "reason": f"no checkpoint directory at {weights_path!r}"})
        problem = _proof_problem(self.proof)
        if problem and not allow_untrained:
            raise RuntimeError(f"PRM checkpoint not usable: {problem}. Pass allow_untrained=True only to "
                               f"wire up the pipeline; scores would be the base model's, not a trained V_phi.")
        self.trained = problem is None
        self.weights_path = str(weights_path) if weights_path else None
        self.model = model.to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.slots = SlotMatcher.from_encoder(lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"],
                                              self.layout.template)
        for tok, tid in ((POS_TOKEN, self.slots.pos_id), (NEG_TOKEN, self.slots.neg_id)):
            if self.tokenizer.convert_tokens_to_ids(tok) != tid:
                raise RuntimeError(f"{tok!r} is not a single token at the slot position")
        self.img_context_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
        self.model.img_context_token_id = self.img_context_id
        self.num_image_token = int(self.model.num_image_token)
        pad = self.tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else (self.tokenizer.eos_token_id or 0))

        self.refresh_head()
        self.tiles = TileLoader()

    def refresh_head(self) -> None:
        torch = self._torch
        head_w = self.model.language_model.get_output_embeddings()
        idx = torch.tensor([self.slots.pos_id, self.slots.neg_id], device=head_w.weight.device)
        with torch.no_grad():
            self._w2 = head_w.weight.detach().index_select(0, idx).float()
            b = getattr(head_w, "bias", None)
            self._b2 = None if b is None else b.detach().index_select(0, idx).float()

    def describe(self) -> Dict[str, Any]:
        import transformers

        return {"prm": self.model_path, "weights": self.weights_path if self.trained else "UNTRAINED-BACKBONE",
                "trained": self.trained, "layout": self.layout.describe(), "kappa": KAPPA,
                "batch_size": self.batch_size,
                "checkpoint_proof": self.proof, "device": self.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": self._torch.__version__, "transformers": transformers.__version__}

    def score_candidates(self, requests: Sequence[ScoreRequest],
                         progress: Optional[Callable[[int, int], None]] = None) -> List[List[float]]:
        reqs = list(requests)
        prompts = [self.layout.build(r.system, r.question, r.steps, r.image_paths, [POS_TOKEN] * len(r.steps))
                   for r in reqs]
        return self._run(prompts, [r.key for r in reqs], [r.group for r in reqs], progress)

    def _run(self, prompts: Sequence[PRMPrompt], keys: Sequence[str], groups: Sequence[Any],
             progress: Optional[Callable[[int, int], None]] = None) -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(prompts)
        chunk: List[int] = []
        for i in range(len(prompts) + 1):
            if chunk and (i == len(prompts) or len(chunk) >= self.batch_size or groups[i] != groups[chunk[0]]):
                for j, p in zip(chunk, self._forward([prompts[j] for j in chunk], [keys[j] for j in chunk])):
                    out[j] = p
                if progress:
                    progress(chunk[-1] + 1, len(prompts))
                chunk = []
            if i < len(prompts):
                chunk.append(i)
        return out

    def encode(self, pr: PRMPrompt, key: Optional[str] = None):
        tiles = []
        for rank, path in enumerate(pr.image_paths):
            try:
                tiles.append(self.tiles(path, pr.max_tiles))
            except Exception as e:
                what = str(e) if isinstance(e, PRMImageError) else f"unreadable PRM input image ({type(e).__name__})"
                raise PRMImageError(f"{what}: request {key if key is not None else '?'!r}, "
                                    f"image {rank + 1}") from None
        per_block, k = [], 0
        for nb in pr.blocks:
            per_block.append(sum(int(t.shape[0]) for t in tiles[k:k + nb]))
            k += nb
        ids = self.tokenizer(expand_image_tokens(pr.text, per_block, self.num_image_token))["input_ids"]
        pos = self.slots.find(ids)
        if len(pos) != pr.n_slots:
            raise RuntimeError(f"score-position mismatch: {len(pos)} slots found for {pr.n_slots} expected "
                               f"(a step text reproduces the slot pattern, or the template changed)")
        return ids, pos, tiles

    def _forward(self, prompts: Sequence[PRMPrompt], keys: Sequence[str]) -> List[List[float]]:
        torch = self._torch
        enc = [self.encode(pr, key) for pr, key in zip(prompts, keys)]
        B, L = len(enc), max(len(e[0]) for e in enc)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for b, (seq, _, _) in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[b, :len(seq)] = 1
        tiles = [t for e in enc for t in e[2]]
        if tiles:
            pv = torch.cat(tiles, 0)
            flags = torch.ones(pv.shape[0], dtype=torch.long)
        else:
            pv = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            flags = torch.zeros(1, dtype=torch.long)
        ids, mask = ids.to(self.device), mask.to(self.device)
        pv, flags = pv.to(self.dtype).to(self.device), flags.to(self.device)
        b_idx = torch.tensor([b for b, e in enumerate(enc) for _ in e[1]], device=self.device)
        s_idx = torch.tensor([s for e in enc for s in e[1]], device=self.device)
        with torch.inference_mode():
            h = self.hidden(ids, mask, pv, flags)[b_idx, s_idx].float()
            z = (h.unsqueeze(1) * self._w2.unsqueeze(0)).sum(-1)
            if self._b2 is not None:
                z = z + self._b2
            p = torch.sigmoid(z[:, 0] - z[:, 1]).cpu().tolist()
        out, k = [], 0
        for _, pos, _ in enc:
            out.append([float(x) for x in p[k:k + len(pos)]])
            k += len(pos)
        return out

    def hidden(self, ids, mask, pv, flags):
        m = self.model
        emb = m.language_model.get_input_embeddings()(ids).clone()
        vit = m.extract_feature(pv)[flags == 1]
        B, N, C = emb.shape
        emb = emb.reshape(B * N, C)
        sel = ids.reshape(B * N) == self.img_context_id
        if int(sel.sum()) != vit.shape[0] * vit.shape[1]:
            raise RuntimeError(f"{int(sel.sum())} image slots for {vit.shape[0] * vit.shape[1]} vision rows")
        emb[sel] = emb[sel] * 0.0 + vit.reshape(-1, C)
        decoder = m.language_model.model
        return decoder(inputs_embeds=emb.reshape(B, N, C), attention_mask=mask, use_cache=False).last_hidden_state
