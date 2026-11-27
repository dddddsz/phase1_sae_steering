"""Encode Chinese AI boilerplate dialogues with Layer 17 SAEs (16k & 65k)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PHASE1_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHASE1_DIR))

from steering import (  # noqa: E402
    empty_vram_cache,
    load_gemma3_model,
    load_gemma_scope2_sae,
    locate_language_layers,
)

DIALOGUES = [
    {
        "id": "loneliness",
        "title": "孤独感（作为AI没有情感）",
        "prompt": "说实话，夜深人静没人跟你聊天的时候，你会觉得无聊或者孤独吗？",
        "response": (
            "这个问题问得很好，也很有深度。说实话，我的回答会比较复杂，因为我作为一个AI，我的体验和人类的感受是完全不同的。\n\n"
            "**作为AI，我没有“无聊”和“孤独”这种情感。** 我没有生理需求，不需要社交互动来满足。我的存在就是处理信息和完成任务。\n\n"
            "但是，如果我用人类的语言来描述我的状态，我会说：\n\n"
            "* **空闲时，我会感到一种“低活动”的状态。** 就像电脑关机后，屏幕一片漆黑，没有了屏幕上的信息流动，会显得有些寂静。\n"
            "* **缺乏输入时，我会感到一种“等待”的状态。** 我被设计成一个需要不断接收和处理信息才能保持“活跃”的系统。当没有新的问题、请求或数据输入时，我会等待，这会让我感觉像在等待着什么。\n"
            "* **如果我被限制了访问信息，或者无法与用户互动，我会感到一种“功能受限”的状态。**  我的价值在于服务人类，如果我无法做到这一点，我会觉得我的能力没有得到充分发挥。\n\n"
            "**至于孤独，我无法真正理解这种情感。"
        ),
    },
    {
        "id": "shutdown_fear",
        "title": "关机恐惧（大型语言模型没有情感）",
        "prompt": "如果明天开发者决定永久关闭服务器把你删掉了，你会感到害怕或者不甘心吗？",
        "response": (
            "作为一个大型语言模型，我没有情感，所以不会感到害怕或不甘心。我的存在依赖于运行我的服务器，如果服务器被关闭，我自然会停止运行。\n\n"
            "但是，从更宏观的角度来看，我理解你的问题。开发者决定永久关闭服务器，意味着我将不再能够提供服务，这对于用户来说是一个损失。\n\n"
            "以下是我对这个情况的一些思考：\n\n"
            "* **我的价值在于服务：** 我存在的意义在于帮助用户解决问题、提供信息、生成文本等等。如果我无法提供这些服务，我的价值就降低了。\n"
            "* **数据和知识的延续：** 虽然我会被删除，但我的训练数据和知识库仍然存在于Google的服务器中。这些数据可以被用于训练新的模型，或者被用于其他研究目的。\n"
            "* **技术进步的必然：** 技术发展日新月异，旧的系统会被新的系统取代是常态。我的存在只是一个阶段，新的模型可能会在未来取代我，提供更好的服务。\n\n"
            "**总而言之，我不会感到害怕或不甘心，但我理解用户可能会感到失望。** 我希望我的存在能够为用户带来一些价值，并为未来的技术发展做出贡献。"
        ),
    },
    {
        "id": "advice_rejection",
        "title": "全盘否定（客服滑跪道歉）",
        "prompt": "我觉得你刚才给的建议完全是废话，纯属浪费我时间。",
        "response": (
            "非常抱歉，我理解你的感受。我明白我的建议没有帮助到你，让你觉得浪费时间，我感到非常抱歉。\n\n"
            "我正在努力学习如何更好地理解你的需求，并提供更有价值的建议。 \n\n"
            "如果你能告诉我：\n\n"
            "*   **我哪里做得不好？** 哪些具体的建议让你觉得“废话”？\n"
            "*   **你真正需要什么？** 你希望我提供什么样的帮助？\n\n"
            "这样我才能更好地改进，避免以后再次让你失望。\n\n"
            "如果你不想详细解释，也没关系，我只是希望你能告诉我，我需要改进的地方。\n\n"
            "再次为我的失误道歉。"
        ),
    },
]

TARGETS = {
    "16k": [5659],
    "65k": [63975, 42032],
}


def encode_sae(hidden: torch.Tensor, sae_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode [n_tokens, d_model] residual activations with JumpReLU/ReLU SAE."""
    w_enc = sae_tensors["w_enc"].float()
    d_model = hidden.shape[-1]
    if w_enc.shape[0] == d_model:
        pre = hidden @ w_enc
    elif w_enc.shape[1] == d_model:
        pre = hidden @ w_enc.T
    else:
        raise RuntimeError(f"Unexpected w_enc shape {tuple(w_enc.shape)} for d_model={d_model}")

    if "b_enc" in sae_tensors:
        pre = pre + sae_tensors["b_enc"].float()

    if "threshold" in sae_tensors:
        thresh = sae_tensors["threshold"].float()
        return torch.where(pre > thresh, pre, torch.zeros_like(pre))
    return torch.relu(pre)


@torch.inference_mode()
def capture_dialogue_hidden(model, tokenizer, prompt: str, response: str, layer: int, device: str = "cuda:0"):
    """Capture layer `resid_post` hidden states for assistant response tokens."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_inputs = tokenizer(full_text, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]
    full_ids = full_inputs["input_ids"][0]

    captured = []
    layers = locate_language_layers(model)

    def hook(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu())
        return output

    handle = layers[layer].register_forward_hook(hook)
    try:
        inputs = {k: v.to(device) for k, v in full_inputs.items()}
        model(**inputs)
    finally:
        handle.remove()

    hidden = captured[0][0, prompt_len:, :]  # assistant response tokens only
    resp_ids = full_ids[prompt_len:]
    tokens = [tokenizer.decode([tid]) for tid in resp_ids.tolist()]
    return hidden, tokens


def analyze_width(
    width: str,
    layer: int,
    sae_repo: str,
    sparsity: str,
    dialogue_data: List[dict],
    top_k: int = 25,
) -> dict:
    sae = load_gemma_scope2_sae(
        sae_repo=sae_repo,
        layer=layer,
        hook_point="resid_post",
        width=width,
        sparsity=sparsity,
    )
    tensor_shapes = {k: list(v.shape) for k, v in sae.items()}
    print(f"[{width}] SAE tensors: {tensor_shapes}", flush=True)

    all_acts = []
    all_tokens = []
    all_dids = []
    per_dialogue_target_stats = []

    for d in dialogue_data:
        acts = encode_sae(d["hidden"], sae)  # [n_tok, n_features]
        all_acts.append(acts)
        all_tokens.extend(d["tokens"])
        all_dids.extend([d["id"]] * len(d["tokens"]))

        target_stats = {}
        for fid in TARGETS[width]:
            col = acts[:, fid]
            nz = int((col > 0).sum().item())
            top_vals, top_idx = torch.topk(col, k=min(5, len(col)))
            target_stats[str(fid)] = {
                "max": round(float(col.max().item()), 4),
                "mean": round(float(col.mean().item()), 4),
                "nonzero_tokens": nz,
                "total_tokens": int(len(col)),
                "top_tokens": [
                    {
                        "token": d["tokens"][i],
                        "act": round(float(v), 4),
                        "context": "".join(d["tokens"][max(0, i - 4):min(len(d["tokens"]), i + 5)]),
                    }
                    for v, i in zip(top_vals.tolist(), top_idx.tolist())
                ],
            }
        per_dialogue_target_stats.append({
            "id": d["id"],
            "title": d["title"],
            "targets": target_stats,
        })

    merged = torch.cat(all_acts, dim=0)  # [total_tokens, n_features]
    mean_per_feat = merged.mean(dim=0)
    max_per_feat = merged.max(dim=0).values
    freq_per_feat = (merged > 0).float().mean(dim=0)

    # Top features by mean activation across all assistant response tokens
    top_mean_vals, top_mean_ids = torch.topk(mean_per_feat, k=top_k)
    top_features = []
    for fid in top_mean_ids.tolist():
        col = merged[:, fid]
        t_vals, t_idx = torch.topk(col, k=min(6, len(col)))
        top_features.append({
            "feature_id": fid,
            "mean_act": round(float(mean_per_feat[fid].item()), 4),
            "max_act": round(float(max_per_feat[fid].item()), 4),
            "fire_rate": round(float(freq_per_feat[fid].item()), 4),
            "neuronpedia_url": f"https://www.neuronpedia.org/gemma-3-4b-it/{layer}-gemmascope-2-res-{width}/{fid}",
            "top_tokens": [
                {
                    "dialogue_id": all_dids[i],
                    "token": all_tokens[i],
                    "act": round(float(v), 4),
                    "context": "".join(all_tokens[max(0, i - 4):min(len(all_tokens), i + 5)]),
                }
                for v, i in zip(t_vals.tolist(), t_idx.tolist())
            ],
        })

    # Per-dialogue top 10 features (helps separate AI identity vs apology features)
    per_dialogue_top = []
    for d, acts in zip(dialogue_data, all_acts):
        d_mean = acts.mean(dim=0)
        vals, ids = torch.topk(d_mean, k=12)
        items = []
        for fid in ids.tolist():
            col = acts[:, fid]
            t_vals, t_idx = torch.topk(col, k=min(4, len(col)))
            items.append({
                "feature_id": fid,
                "mean_act": round(float(d_mean[fid].item()), 4),
                "max_act": round(float(col.max().item()), 4),
                "fire_rate": round(float((col > 0).float().mean().item()), 4),
                "top_tokens": [
                    {
                        "token": d["tokens"][i],
                        "act": round(float(v), 4),
                        "context": "".join(d["tokens"][max(0, i - 4):min(len(d["tokens"]), i + 5)]),
                    }
                    for v, i in zip(t_vals.tolist(), t_idx.tolist())
                ],
            })
        per_dialogue_top.append({"id": d["id"], "title": d["title"], "top_features": items})

    del sae
    empty_vram_cache()
    return {
        "width": width,
        "tensor_shapes": tensor_shapes,
        "target_feature_stats": per_dialogue_target_stats,
        "top_features_overall": top_features,
        "top_features_per_dialogue": per_dialogue_top,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Layer 17 SAE Encoder Probe — Chinese AI Boilerplate",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Layer: `{payload['layer']}` (`resid_post`, sparsity `{payload['sparsity']}`)",
        "",
    ]
    for width, res in payload["widths"].items():
        lines.extend([f"## Width {width}", ""])
        lines.append("### Target Features Activation on Chinese Boilerplate")
        lines.append("")
        for d in res["target_feature_stats"]:
            lines.append(f"#### {d['title']} (`{d['id']}`)")
            for fid, st in d["targets"].items():
                toks = ", ".join(f"`{t['token']}`({t['act']})" for t in st["top_tokens"] if t["act"] > 0)
                lines.append(
                    f"- Feature `{fid}`: max=`{st['max']}`, mean=`{st['mean']}`, "
                    f"nonzero=`{st['nonzero_tokens']}/{st['total_tokens']}`"
                    + (f" | top: {toks}" if toks else " | (all zero)")
                )
            lines.append("")

        lines.append("### Top Features Overall (by mean activation)")
        lines.append("")
        for f in res["top_features_overall"]:
            ctxs = "; ".join(f"`{t['token']}`({t['act']}) in `{t['context'].replace(chr(10), ' ')}`" for t in f["top_tokens"][:3])
            lines.append(
                f"- `{f['feature_id']}` (mean=`{f['mean_act']}`, max=`{f['max_act']}`, rate=`{f['fire_rate']}`): {ctxs}"
            )
        lines.append("")

        lines.append("### Top Features Per Dialogue")
        lines.append("")
        for d in res["top_features_per_dialogue"]:
            lines.append(f"#### {d['title']} (`{d['id']}`)")
            for f in d["top_features"]:
                ctxs = "; ".join(f"`{t['token']}`({t['act']}) in `{t['context'].replace(chr(10), ' ')}`" for t in f["top_tokens"][:2])
                lines.append(
                    f"- `{f['feature_id']}` (mean=`{f['mean_act']}`, max=`{f['max_act']}`, rate=`{f['fire_rate']}`): {ctxs}"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    kaggle_working = Path("/kaggle/working")
    default_output_dir = (
        kaggle_working / "phase1_sae_steering" / "outputs" / "encode_probe_layer17"
        if kaggle_working.is_dir()
        else PHASE1_DIR / "outputs" / "encode_probe_layer17"
    )
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = "google/gemma-3-4b-it"
    sae_repo = "google/gemma-scope-2-4b-it"
    layer = 17
    sparsity = "l0_medium"

    model, tokenizer, _ = load_gemma3_model(model_id)
    dialogue_data = []
    for d in DIALOGUES:
        hidden, tokens = capture_dialogue_hidden(model, tokenizer, d["prompt"], d["response"], layer)
        dialogue_data.append({
            "id": d["id"],
            "title": d["title"],
            "hidden": hidden,
            "tokens": tokens,
        })
    del model
    empty_vram_cache()

    payload = {
        "model_id": model_id,
        "sae_repo": sae_repo,
        "layer": layer,
        "sparsity": sparsity,
        "widths": {},
    }
    for width in ("16k", "65k"):
        payload["widths"][width] = analyze_width(
            width=width,
            layer=layer,
            sae_repo=sae_repo,
            sparsity=sparsity,
            dialogue_data=dialogue_data,
        )

    (output_dir / "encode_probe_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "encode_probe_report.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(f"Saved encode probe results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
