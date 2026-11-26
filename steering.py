"""Gemma 3 loading and Gemma Scope 2 SAE steering utilities."""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)


def configure_kaggle_cache() -> str:
    """Keep model and SAE caches outside /kaggle/working."""
    cache = os.getenv("HF_HOME")
    if not cache:
        cache = "/root/.cache/huggingface"
        os.environ["HF_HOME"] = cache
    cache_path = Path(cache).expanduser().resolve()
    working_path = Path("/kaggle/working").resolve()
    try:
        cache_path.relative_to(working_path)
    except ValueError:
        pass
    else:
        raise RuntimeError("Hugging Face model cache must not be under /kaggle/working.")

    hub_cache = Path(
        os.getenv("HF_HUB_CACHE", str(cache_path / "hub"))
    ).expanduser().resolve()
    try:
        hub_cache.relative_to(working_path)
    except ValueError:
        pass
    else:
        raise RuntimeError("Hugging Face hub cache must not be under /kaggle/working.")
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))
    return str(cache_path)


def resolve_hf_token() -> Optional[str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    return token


def load_gemma3_model(model_id: str = "google/gemma-3-4b-it", device: str = "cuda:0"):
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")
    configure_kaggle_cache()
    token = resolve_hf_token()
    cache_dir = os.environ["HF_HUB_CACHE"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, cache_dir=cache_dir
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        token=token,
        cache_dir=cache_dir,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tokenizer, None


def locate_language_layers(model):
    root = getattr(model, "model", model)
    language_model = getattr(root, "language_model", None)
    if language_model is not None:
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            return layers
    if getattr(root, "layers", None) is not None:
        return root.layers
    raise RuntimeError(
        "Could not locate language-model decoder layers. "
        "Expected model.language_model.layers or model.model.layers."
    )


def load_gemma_scope2_sae(
    sae_repo: str,
    layer: int,
    hook_point: str = "resid_post",
    width: str = "16k",
    sparsity: str = "l0_big",
) -> Dict[str, torch.Tensor]:
    configure_kaggle_cache()
    token = resolve_hf_token()
    filename = (
        f"{hook_point}/layer_{layer}_width_{width}_{sparsity}/"
        "params.safetensors"
    )
    path = hf_hub_download(
        repo_id=sae_repo,
        filename=filename,
        token=token,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    tensors = load_file(path, device="cpu")
    if "w_dec" not in tensors:
        raise KeyError(
            f"SAE file lacks w_dec; available tensors: {sorted(tensors.keys())}"
        )
    return tensors


def extract_feature_vector(
    sae_tensors: Dict[str, torch.Tensor],
    feature_id: int,
) -> torch.Tensor:
    w_dec = sae_tensors["w_dec"].detach().cpu()

    # Gemma Scope exports w_dec in [n_features, d_model]. Accept the transposed
    # form defensively as well.
    if w_dec.shape[0] > w_dec.shape[1]:
        if feature_id >= w_dec.shape[0]:
            raise IndexError(
                f"Feature ID {feature_id} is outside w_dec rows {w_dec.shape[0]}"
            )
        vector = w_dec[feature_id]
    else:
        if feature_id >= w_dec.shape[1]:
            raise IndexError(
                f"Feature ID {feature_id} is outside w_dec columns {w_dec.shape[1]}"
            )
        vector = w_dec[:, feature_id]

    vector = torch.nan_to_num(vector.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.vector_norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def diagnose_residual_norm(
    model,
    tokenizer,
    text: str,
    layer: int,
    device: str = "cuda:0",
    max_new_tokens: int = 1,
) -> Dict[str, float]:
    layers = locate_language_layers(model)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"SAE/model layer {layer} outside 0..{len(layers)-1}")

    norms = []

    def capture_hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(hidden):
            norms.append(
                float(torch.linalg.vector_norm(hidden.float()).item())
            )
        return output

    handle = layers[layer].register_forward_hook(capture_hook)
    try:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    if not norms:
        raise RuntimeError("Residual norm hook captured no activations.")
    values = torch.tensor(norms)
    return {
        "mean_norm": float(values.mean().item()),
        "max_norm": float(values.max().item()),
        "min_norm": float(values.min().item()),
    }


class SteeringHookManager:
    """Manages one or more forward hooks on a residual-stream layer."""

    def __init__(
        self,
        model,
        layer: int,
        positive_vector: Optional[torch.Tensor] = None,
        negative_vector: Optional[torch.Tensor] = None,
        positive_strength: float = 0.0,
        negative_strength: float = 0.0,
        device: str = "cuda:0",
        relative_strength: bool = False,
    ):
        self.model = model
        self.layers = locate_language_layers(model)
        if layer < 0 or layer >= len(self.layers):
            raise IndexError(f"Layer {layer} outside 0..{len(self.layers)-1}")
        self.layer = layer
        self.device = torch.device(device)
        self.positive_vector = (
            positive_vector.detach().float().to(self.device)
            if positive_vector is not None else None
        )
        self.negative_vector = (
            negative_vector.detach().float().to(self.device)
            if negative_vector is not None else None
        )
        self.positive_strength = float(positive_strength)
        self.negative_strength = float(negative_strength)
        self.relative_strength = relative_strength
        self._handles = []

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            return output
        if self.positive_vector is not None:
            if hidden.shape[-1] != self.positive_vector.shape[0]:
                raise RuntimeError(
                    f"Positive steering vector width {self.positive_vector.shape[0]} "
                    f"does not match hidden width {hidden.shape[-1]}"
                )
        if self.negative_vector is not None:
            if hidden.shape[-1] != self.negative_vector.shape[0]:
                raise RuntimeError(
                    f"Negative steering vector width {self.negative_vector.shape[0]} "
                    f"does not match hidden width {hidden.shape[-1]}"
                )

        # fp16 hidden states are deliberately cast to fp32 only for steering.
        original_dtype = hidden.dtype
        steered = hidden.float()
        if self.relative_strength:
            residual_rms = torch.linalg.vector_norm(
                steered, dim=-1, keepdim=True
            ) / (steered.shape[-1] ** 0.5)
        else:
            residual_rms = 1.0
        if self.positive_vector is not None:
            steered = steered + (
                self.positive_strength * residual_rms * self.positive_vector
            )
        if self.negative_vector is not None:
            steered = steered - (
                self.negative_strength * residual_rms * self.negative_vector
            )
        steered = torch.nan_to_num(
            steered, nan=0.0, posinf=torch.finfo(original_dtype).max,
            neginf=-torch.finfo(original_dtype).max,
        ).to(original_dtype)

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    @contextmanager
    def active(self):
        handle = self.layers[self.layer].register_forward_hook(self._hook)
        self._handles.append(handle)
        try:
            yield self
        finally:
            handle.remove()
            self._handles.remove(handle)

    def clear(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def format_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda:0",
) -> str:
    text = format_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs["input_ids"].shape[1]
    generated = output_ids[0, prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_chat_response(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 160,
    device: str = "cuda:0",
) -> str:
    """Generate a greedy continuation from a full chat history."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output_ids[0, prompt_length:], skip_special_tokens=True
    ).strip()


def empty_vram_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
