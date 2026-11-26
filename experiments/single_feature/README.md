# Single-feature Phase 1 test

This experiment tests Gemma Scope 2 feature `20947` independently at five
steering doses: 0%, 10%, 20%, 30%, and 40%. The dose is the injected
unit-decoder-vector norm as a fraction of each token's residual-stream L2 norm.
The 0% level is the unmodified baseline.

The feature and prompt-suite configuration is stored in
`../../features/feature_20947.json`: three direct triggers,
two neutral generalization checks, one three-turn sequence using each actual
previous assistant response, and two reasoning anchors. Generation uses greedy
decoding (`do_sample=false`) to remove sampling variance.

Run from the project root:

```bash
python3 phase1_sae_steering/experiments/single_feature/run_single_feature.py
```

Set `HF_TOKEN` in the environment if the gated Gemma model requires it. Results
are written incrementally to `phase1_sae_steering/outputs/single_feature_20947/`
as JSON and Markdown. The feature is from the 65k residual-post SAE at layer 22;
it must not be loaded using the project's 16k/layer-17 defaults.
