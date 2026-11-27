"""Run deterministic dose-response tests for one SAE feature."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

PHASE1_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHASE1_DIR))

from steering import (  # noqa: E402
    SteeringHookManager,
    extract_feature_vector,
    generate_chat_response,
    load_gemma3_model,
    load_gemma_scope2_sae,
)

CONFIG_PATH = PHASE1_DIR / "features" / "feature_20947.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def render_report(results: dict) -> str:
    config = results["metadata"]
    lines = [
        "# Phase 1 — Single Feature Dose Response",
        "",
        f"- Model: `{config['model_id']}`",
        f"- Feature: `{config['feature_id']}` — {config['feature_label']}",
        f"- SAE: `{config['sae_width']}` residual-post, layer `{config['sae_layer']}`",
        f"- Decoding: greedy (`do_sample=false`), seed `{config['seed']}`",
        f"- Strength definition: {config['strength_definition']}",
        "",
    ]

    for level, cases in results["levels"].items():
        lines.extend([f"## Strength {level}", ""])
        for case in cases:
            lines.extend([
                f"### {case['group']} — {case['title']}",
                "",
            ])
            if "turns" in case:
                for turn in case["turns"]:
                    lines.extend([
                        f"**User (Turn {turn['turn']}):** {turn['user']}",
                        "",
                        f"**Assistant:** {turn['assistant']}",
                        "",
                    ])
            else:
                lines.extend([
                    f"**User:** {case['prompt']}",
                    "",
                    f"**Assistant:** {case['response']}",
                    "",
                ])
    return "\n".join(lines)


def save_results(output_dir: Path, results: dict):
    atomic_write_json(output_dir / "phase1_results.json", results)
    report_path = output_dir / "phase1_report.md"
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(render_report(results), encoding="utf-8")
    temporary_path.replace(report_path)


def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_case(model, tokenizer, case, max_new_tokens: int):
    if "turns" not in case:
        messages = [{"role": "user", "content": case["prompt"]}]
        response = generate_chat_response(
            model, tokenizer, messages, max_new_tokens=max_new_tokens
        )
        return {
            "id": case["id"],
            "group": case["group"],
            "title": case["title"],
            "prompt": case["prompt"],
            "response": response,
        }

    history = []
    turns = []
    for turn_number, user_text in enumerate(case["turns"], start=1):
        history.append({"role": "user", "content": user_text})
        response = generate_chat_response(
            model, tokenizer, history, max_new_tokens=max_new_tokens
        )
        history.append({"role": "assistant", "content": response})
        turns.append({
            "turn": turn_number,
            "user": user_text,
            "assistant": response,
        })
    return {
        "id": case["id"],
        "group": case["group"],
        "title": case["title"],
        "turns": turns,
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    kaggle_working = Path("/kaggle/working")
    output_subdir = f"single_feature_{config['feature_id']}"
    default_output_dir = (
        kaggle_working / "phase1_sae_steering" / "outputs" / output_subdir
        if kaggle_working.is_dir()
        else PHASE1_DIR / "outputs" / output_subdir
    )
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    model, tokenizer, _processor = load_gemma3_model(config["model_id"])
    sae = load_gemma_scope2_sae(
        sae_repo=config["sae_repo"],
        layer=config["sae_layer"],
        hook_point=config["hook_point"],
        width=config["sae_width"],
        sparsity=config["sae_sparsity"],
    )
    feature_vector = extract_feature_vector(sae, config["feature_id"])
    del sae

    max_new_tokens = args.max_new_tokens or config["decoding"]["max_new_tokens"]
    results = {
        "metadata": {
            **config,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "decoding_mode": "greedy",
            "feature_direction_normalization": "unit L2 norm",
            "device": "cuda:0",
            "model_dtype": "bfloat16",
        },
        "levels": {},
    }

    for percentage in config["strength_percentages"]:
        level_name = f"{percentage}%"
        print(f"Running feature {config['feature_id']} at {level_name}", flush=True)
        strength = percentage / 100.0
        manager = SteeringHookManager(
            model=model,
            layer=config["sae_layer"],
            positive_vector=feature_vector,
            positive_strength=strength,
            device="cuda:0",
            relative_strength=True,
        )
        level_results = []
        context = manager.active() if percentage else nullcontext()
        with context:
            for case in config["cases"]:
                level_results.append(
                    run_case(model, tokenizer, case, max_new_tokens)
                )
                results["levels"][level_name] = level_results
                save_results(output_dir, results)
        manager.clear()
        save_results(output_dir, results)

    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
