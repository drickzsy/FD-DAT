## 📊 Dataset Preparation

The experiments are conducted on four benchmarks. Please organize the data as follows:

- **SAR Ship Detection:** GF-3 and SSDD.

- **Aircraft Detection:** MAR20 (Optical) and SAR-AIR (SAR).

Plaintext

```
data/
├── GF-3/
│   ├── train/
│   └── test/
├── SSDD/
│   └── ...
└── MAR20/
│   └── ...
└── SAR-AIR/
    └── ...
```

## 🚀 Training and Evaluation

### Training (Example: GF-3 to SSDD)

To train the FD-DAT model with the optimized hyperparameters:

Bash

```
python train.py --config config.yaml --source gf3 --target ssdd --backbone resnet101
```

### Evaluation

To evaluate the detection performance (mAP, PR, RE):

Bash

```
python eval.py --resume checkpoint.pth --dataset ssdd
```

## 📈 Experimental Results

Our method achieves state-of-the-art performance across multiple tasks:

| **Task**            | **mAP (Ours)** | **Baseline (DETR)** | **Gain** |
| ------------------- | -------------- | ------------------- | -------- |
| **GF-3 → SSDD**     | **87.2%**      | 64.6%               | +22.6%   |
| **SSDD → GF-3**     | **82.1%**      | 56.5%               | +25.6%   |
| **MAR20 → SAR-AIR** | **67.0%**      | 52.5%               | +14.5%   |
