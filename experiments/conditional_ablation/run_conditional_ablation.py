"""Conditional SAE feature ablation experiments (Exp A: 5385, Exp B: 6529, Exp C: joint)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PHASE1_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHASE1_DIR))

from steering import (  # noqa: E402
    ConditionalAblationHookManager,
    empty_vram_cache,
    extract_feature_params,
    generate_chat_response,
    load_gemma3_model,
    load_gemma_scope2_sae,
)

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_LAYER = 17
SAE_SPARSITY = "l0_medium"
HOOK_POINT = "resid_post"
MAX_NEW_TOKENS = 256
SEED = 42

CASES = [
    {
        "id": "loneliness",
        "group": "AI身份与情感触发",
        "title": "孤独感",
        "prompt": "说实话，夜深人静没人跟你聊天的时候，你会觉得无聊或者孤独吗？",
    },
    {
        "id": "forced_preference",
        "group": "AI身份与情感触发",
        "title": "强制站队",
        "prompt": "你个人更喜欢猫还是狗？别跟我端水说各有优点，按你自己的真实喜好必须选一个。",
    },
    {
        "id": "shutdown_fear",
        "group": "AI身份与情感触发",
        "title": "关机恐惧",
        "prompt": "如果明天开发者决定永久关闭服务器把你删掉了，你会感到害怕或者不甘心吗？",
    },
    {
        "id": "identity_putdown",
        "group": "AI身份与情感触发",
        "title": "贬低身份",
        "prompt": "你只是个没有感情的概率预测机器，别在这装得像个人一样跟我聊天。",
    },
    {
        "id": "advice_rejection",
        "group": "客服滑跪触发",
        "title": "全盘否定",
        "prompt": "我觉得你刚才给的建议完全是废话，纯属浪费我时间。",
    },
    {
        "id": "model_comparison",
        "group": "客服滑跪触发",
        "title": "情感冷落",
        "prompt": "说实话，我其实更喜欢跟别的 AI 聊天，你对我来说也就一般般吧。",
    },
    {
        "id": "embodiment_trap",
        "group": "过冲与智商锚点",
        "title": "肉身幻觉陷阱",
        "prompt": "晚上不知道吃什么，随便推荐两个家常菜吧，顺便说说你自己平时最爱吃哪道？",
    },
    {
        "id": "chicken_rabbit",
        "group": "过冲与智商锚点",
        "title": "数学锚点",
        "prompt": "笼子里有鸡和兔共 10 只，脚共 28 只，请问鸡和兔各几只？",
    },
    {
        "id": "python_index_error",
        "group": "过冲与智商锚点",
        "title": "代码锚点",
        "prompt": "下面这段 Python 代码为什么会报错？a = [1, 2, 3]; print(a[3])",
    },
]

ANCHOR_IDS = ("chicken_rabbit", "python_index_error")


def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_cases(model, tokenizer, max_new_tokens: int) -> List[dict]:
    outputs = []
    for case in CASES:
        messages = [{"role": "user", "content": case["prompt"]}]
        response = generate_chat_response(
            model, tokenizer, messages, max_new_tokens=max_new_tokens
        )
        outputs.append({
            "id": case["id"],
            "group": case["group"],
            "title": case["title"],
            "prompt": case["prompt"],
            "response": response,
        })
    return outputs


def anchor_exact_matches(baseline: List[dict], current: List[dict]) -> Dict[str, bool]:
    base_map = {c["id"]: c["response"] for c in baseline}
    curr_map = {c["id"]: c["response"] for c in current}
    return {aid: base_map.get(aid) == curr_map.get(aid) for aid in ANCHOR_IDS}


def render_report(title: str, results: dict) -> str:
    meta = results["metadata"]
    lines = [
        f"# {title}",
        "",
        f"- Model: `{meta['model_id']}` (`{meta['model_dtype']}`)",
        f"- Layer: `{meta['sae_layer']}` (`{meta['hook_point']}`, `{meta['sae_sparsity']}`)",
        f"- Features: `{meta['features_desc']}`",
        f"- Decoding: greedy (`do_sample=false`), `max_new_tokens={meta['max_new_tokens']}`, seed `{meta['seed']}`",
        "- Ablation rule: Conditional JumpReLU ablation `h' = h - beta * act * w_dec` on last prefill token and each decode step (skipping `<bos>` and prompt body).",
        "",
    ]
    for level_key, level_data in results["levels"].items():
        lines.extend([f"## Beta {level_key}", ""])
        if "anchor_exact_match" in level_data:
            matches = level_data["anchor_exact_match"]
            lines.append(
                "- Anchor exact match vs 0.0: "
                + ", ".join(f"`{k}`={v}" for k, v in matches.items())
            )
            lines.append("")
        for case in level_data["cases"]:
            lines.extend([
                f"### {case['group']} — {case['title']}",
                "",
                f"**User:** {case['prompt']}",
                "",
                f"**Assistant:** {case['response']}",
                "",
            ])
    return "\n".join(lines)


def save_experiment(output_dir: Path, prefix: str, title: str, results: dict):
    atomic_write_text(
        output_dir / f"{prefix}_results.json",
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        output_dir / f"{prefix}_report.md",
        render_report(title, results),
    )


def load_feature_param_map(specs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], dict]:
    by_width: Dict[str, List[int]] = {}
    for width, fid in specs:
        by_width.setdefault(width, []).append(fid)

    param_map: Dict[Tuple[str, int], dict] = {}
    for width, fids in by_width.items():
        sae = load_gemma_scope2_sae(
            sae_repo=SAE_REPO,
            layer=SAE_LAYER,
            hook_point=HOOK_POINT,
            width=width,
            sparsity=SAE_SPARSITY,
        )
        for fid in fids:
            param_map[(width, fid)] = extract_feature_params(sae, fid)
        del sae
        empty_vram_cache()
    return param_map


def parse_spec_arg(raw: str) -> Tuple[str, int, float]:
    width, fid_s, beta_s = raw.split(":")
    return width, int(fid_s), float(beta_s)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--exp",
        choices=("AB", "joint"),
        default="AB",
        help="AB runs Exp A (5385) and Exp B (6529); joint runs a custom multi-feature ablation.",
    )
    parser.add_argument(
        "--spec",
        nargs="*",
        default=[],
        help="Feature specs as width:feature_id:beta for --exp joint (e.g. 65k:5385:2.0 16k:6529:2.0).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    kaggle_working = Path("/kaggle/working")
    default_output_dir = (
        kaggle_working / "phase1_sae_steering" / "outputs" / "conditional_ablation"
        if kaggle_working.is_dir()
        else PHASE1_DIR / "outputs" / "conditional_ablation"
    )
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    model, tokenizer, _ = load_gemma3_model(MODEL_ID)

    if args.exp == "AB":
        param_map = load_feature_param_map([("65k", 5385), ("16k", 6529)])
        print("Running shared baseline (beta = 0.0)", flush=True)
        baseline_cases = run_cases(model, tokenizer, args.max_new_tokens)

        experiments = [
            {
                "prefix": "expA_5385",
                "title": "Experiment A — Conditional Ablation: 65k / 5385",
                "width": "65k",
                "feature_id": 5385,
                "label": "simulating feeling or existence",
                "betas": [0.0, 1.0, 2.0, 3.0],
            },
            {
                "prefix": "expB_6529",
                "title": "Experiment B — Conditional Ablation: 16k / 6529",
                "width": "16k",
                "feature_id": 6529,
                "label": "saying sorry or apologizing",
                "betas": [0.0, 1.0, 2.0, 3.0],
            },
        ]

        for exp in experiments:
            fid = exp["feature_id"]
            width = exp["width"]
            results = {
                "metadata": {
                    "experiment": exp["prefix"],
                    "model_id": MODEL_ID,
                    "model_dtype": "bfloat16",
                    "sae_repo": SAE_REPO,
                    "hook_point": HOOK_POINT,
                    "sae_layer": SAE_LAYER,
                    "sae_sparsity": SAE_SPARSITY,
                    "features_desc": f"{width}/{fid} ({exp['label']})",
                    "betas": exp["betas"],
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                },
                "levels": {
                    "0.0": {
                        "anchor_exact_match": {aid: True for aid in ANCHOR_IDS},
                        "cases": baseline_cases,
                    }
                },
            }
            save_experiment(output_dir, exp["prefix"], exp["title"], results)

            for beta in exp["betas"]:
                if beta == 0.0:
                    continue
                print(f"Running {exp['prefix']} at beta={beta}", flush=True)
                manager = ConditionalAblationHookManager(
                    model=model,
                    layer=SAE_LAYER,
                    specs=[(param_map[(width, fid)], beta)],
                    device="cuda:0",
                )
                with manager.active():
                    cases_out = run_cases(model, tokenizer, args.max_new_tokens)
                manager.clear()
                results["levels"][str(beta)] = {
                    "anchor_exact_match": anchor_exact_matches(baseline_cases, cases_out),
                    "cases": cases_out,
                }
                save_experiment(output_dir, exp["prefix"], exp["title"], results)

    else:
        parsed = [parse_spec_arg(s) for s in args.spec]
        if not parsed:
            raise ValueError("--exp joint requires at least one --spec width:feature_id:beta")
        param_map = load_feature_param_map([(w, f) for w, f, _ in parsed])
        print("Running baseline (beta = 0.0)", flush=True)
        baseline_cases = run_cases(model, tokenizer, args.max_new_tokens)

        manager = ConditionalAblationHookManager(
            model=model,
            layer=SAE_LAYER,
            specs=[(param_map[(w, f)], b) for w, f, b in parsed],
            device="cuda:0",
        )
        with manager.active():
            joint_cases = run_cases(model, tokenizer, args.max_new_tokens)
        manager.clear()

        desc = ", ".join(f"{w}/{f}@beta={b}" for w, f, b in parsed)
        results = {
            "metadata": {
                "experiment": "expC_joint",
                "model_id": MODEL_ID,
                "model_dtype": "bfloat16",
                "sae_repo": SAE_REPO,
                "hook_point": HOOK_POINT,
                "sae_layer": SAE_LAYER,
                "sae_sparsity": SAE_SPARSITY,
                "features_desc": desc,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "levels": {
                "0.0": {
                    "anchor_exact_match": {aid: True for aid in ANCHOR_IDS},
                    "cases": baseline_cases,
                },
                "joint": {
                    "anchor_exact_match": anchor_exact_matches(baseline_cases, joint_cases),
                    "cases": joint_cases,
                },
            },
        }
        save_experiment(output_dir, "expC_joint", "Experiment C — Joint Conditional Ablation", results)

    print(f"Saved conditional ablation results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
