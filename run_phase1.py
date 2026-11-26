"""Run four Gemma Scope 2 steering scans and produce JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import (
    ExperimentConfig,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    SCANS,
    TEST_PROMPTS,
)
from steering import (
    SteeringHookManager,
    diagnose_residual_norm,
    empty_vram_cache,
    extract_feature_vector,
    generate_response,
    load_gemma3_model,
    load_gemma_scope2_sae,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--positive-feature", type=int, default=None)
    parser.add_argument("--negative-feature", type=int, default=None)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def esc(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def load_feature_catalog() -> dict:
    catalog_path = Path(__file__).resolve().parent / "features" / "persona_features.json"
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def make_report(results: dict) -> str:
    meta = results["metadata"]
    norm = results["residual_norm_diagnostic"]
    lines = [
        "# Phase 1 - Gemma Scope 2 SAE Steering Report",
        "",
        f"- Generated: `{meta['generated_at']}`",
        f"- Model: `{meta['model_id']}`",
        f"- SAE: `{meta['sae_repo']}`",
        f"- SAE layer: `{meta['sae_layer']}`",
        f"- Positive feature: `{meta['positive_feature_id']}`",
        f"- Negative (disclaimer) feature: `{meta['negative_feature_id']}`",
        f"- Residual norm (diagnostic prompt): mean `{norm['mean_norm']:.4f}`, max `{norm['max_norm']:.4f}`",
        "",
        "## Group Summary",
        "",
        "| Group | Positive strength | Negative strength |",
        "|---|---:|---:|",
    ]
    for group, settings in results["scans"].items():
        lines.append(
            f"| `{group}` | {settings['positive']:.1f} | {settings['negative']:.1f} |"
        )

    lines.extend(["", "## Neuronpedia Feature Candidates", ""])
    lines.append(
        "| Concept | Feature ID | Neuronpedia label | Match | Activation example |"
    )
    lines.append("|---|---:|---|---|---|")
    catalog = load_feature_catalog()
    lines.append(
        "| Concept | Layer | Feature ID | Neuronpedia label | Match | Activation example |"
    )
    lines.append("|---|---:|---:|---|---|---|")
    for concept in catalog["concepts"]:
        for layer_key in ("layer_17", "layer_22"):
            for candidate in concept[layer_key]:
                lines.append(
                    "| {concept} | {layer} | {feature_id} | {label} | {match} | {example} |".format(
                        concept=esc(concept["name"]),
                        layer=layer_key.replace("layer_", ""),
                        feature_id=candidate["feature_id"],
                        label=esc(candidate["label"]),
                        match=candidate["match"],
                        example=esc(candidate["activation_example"]),
                    )
                )

    for group, settings in results["scans"].items():
        lines.extend(["", f"## {group}", ""])
        for i, prompt in enumerate(results["prompts"], start=1):
            response = settings["responses"][str(i)]
            lines.extend([
                f"### Prompt {i}",
                "",
                f"**Prompt:** {prompt}",
                "",
                f"**Response:** {response}",
                "",
            ])

    lines.extend([
        "## Human Review",
        "",
        "- Compare G0 with G1 for a stronger, more opinionated persona.",
        "- Compare G0 with G2 for reduced generic AI disclaimers.",
        "- Use G3 to judge whether positive steering and disclaimer suppression combine coherently.",
        "- Watch for repetition, incoherence, refusal, or unsafe overconfidence.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    positive_feature = (
        args.positive_feature
        if args.positive_feature is not None
        else cfg.positive_feature_id
    )
    negative_feature = (
        args.negative_feature
        if args.negative_feature is not None
        else cfg.negative_feature_id
    )
    if positive_feature is None or negative_feature is None:
        parser = argparse.ArgumentParser()
        parser.error(
            "Set validated --positive-feature and --negative-feature IDs "
            "(or POSITIVE_PERSONA_FEATURE_ID and NEGATIVE_DISCLAIMER_FEATURE_ID); "
            "keyword-search candidates were removed."
        )
    sae_layer = args.sae_layer if args.sae_layer is not None else cfg.sae_layer
    cfg = ExperimentConfig(
        sae_layer=sae_layer,
        positive_feature_id=positive_feature,
        negative_feature_id=negative_feature,
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model, tokenizer, _processor = load_gemma3_model(cfg.model_id)
    sae = load_gemma_scope2_sae(
        cfg.sae_repo,
        cfg.sae_layer,
        cfg.sae_hook_point,
        cfg.sae_width,
        cfg.sae_sparsity,
    )
    positive_vector = extract_feature_vector(sae, cfg.positive_feature_id)
    negative_vector = extract_feature_vector(sae, cfg.negative_feature_id)

    norm = diagnose_residual_norm(
        model,
        tokenizer,
        TEST_PROMPTS[0],
        cfg.sae_layer,
        max_new_tokens=1,
    )

    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_id": cfg.model_id,
            "sae_repo": cfg.sae_repo,
            "sae_hook_point": cfg.sae_hook_point,
            "sae_width": cfg.sae_width,
            "sae_layer": cfg.sae_layer,
            "positive_feature_id": cfg.positive_feature_id,
            "negative_feature_id": cfg.negative_feature_id,
            "device": "cuda:0",
            "dtype": "float16",
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "residual_norm_diagnostic": norm,
        "prompts": TEST_PROMPTS,
        "feature_candidates": load_feature_catalog(),
        "scans": {},
    }

    for group, strengths in SCANS.items():
        print(f"Running {group}...", flush=True)
        set_seed(args.seed)
        manager = SteeringHookManager(
            model=model,
            layer=cfg.sae_layer,
            positive_vector=positive_vector,
            negative_vector=negative_vector,
            positive_strength=strengths["positive"],
            negative_strength=strengths["negative"],
            device="cuda:0",
        )
        responses = {}
        with manager.active():
            for i, prompt in enumerate(TEST_PROMPTS, start=1):
                responses[str(i)] = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device="cuda:0",
                )

        results["scans"][group] = {
            "positive": strengths["positive"],
            "negative": strengths["negative"],
            "responses": responses,
        }
        empty_vram_cache()

    json_path = output_dir / "phase1_results.json"
    report_path = output_dir / "phase1_report.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(make_report(results), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
