# Gemma 3 SAE Steering — Phase 1

轻量实验管线，用于在单张 Kaggle T4 上测试 Gemma 3 4B IT 的 Gemma Scope 2
SAE 特征干预。当前推荐入口是 Feature 20947 的单特征剂量实验。

## 项目结构

```text
phase1_sae_steering/
├── features/
│   ├── persona_features.json  # 探针检索记录与候选特征说明
│   └── feature_20947.json     # 20947 元信息、剂量档位及固定测试题
├── experiments/
│   └── single_feature/
│       ├── run_single_feature.py
│       └── README.md
├── outputs/                   # 自动写入实验结果
├── config.py
├── run_phase1.py              # 旧版 G0–G3 双特征扫描入口
├── steering.py
└── requirements.txt
```

特征配置集中存放在 `features/`。Feature ID 必须与对应 SAE 的层、宽度和
hook point 匹配；例如 20947 属于 layer 22、65k、`resid_post`，不可用于
旧版默认的 16k 配置。

## 安装与鉴权

```bash
python3 -m pip install -r phase1_sae_steering/requirements.txt
```

如需访问受限的 Hugging Face 仓库，请通过 Kaggle Secrets 或环境变量提供
`HF_TOKEN`。不要将 Token 写入代码或配置文件。模型与 SAE 缓存使用
Hugging Face 缓存目录，不写入 `/kaggle/working`。运行前需在 Hugging Face
接受 Gemma 模型许可，并确保 Kaggle Notebook 已启用 Internet 和 T4 GPU。

在 Kaggle Notebook 中可用以下步骤启动：

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```bash
!python -m pip install -q -r /kaggle/working/phase1_sae_steering/requirements.txt
!python /kaggle/working/phase1_sae_steering/experiments/single_feature/run_single_feature.py
```

## 运行单特征实验

从项目根目录执行：

```bash
python3 phase1_sae_steering/experiments/single_feature/run_single_feature.py
```

实验使用 0%、10%、20%、30%、40% 五档；0% 为未干预基线。解码采用贪心策略
（`do_sample=false`），多轮题将模型的实际回复接入后续上下文。结果会增量写入；
Kaggle 默认目录如下，本地则写入项目下的 `outputs/`：

```text
/kaggle/working/phase1_sae_steering/outputs/single_feature_20947/
├── phase1_results.json
└── phase1_report.md
```

## 硬件与精度

模型和 SAE steering vector 固定在 `cuda:0`；模型以 FP16 加载。Hook 中的
残差流加法使用 FP32 完成，再转换回原始激活 dtype。Gemma 3 多模态模型的
语言层路径由 `steering.py` 自动定位。

旧版 G0–G3 扫描入口为 `run_phase1.py`，要求显式提供适配其 16k SAE 配置的
正、负 Feature ID。不要把 65k 的 Feature 20947 传给该入口。
