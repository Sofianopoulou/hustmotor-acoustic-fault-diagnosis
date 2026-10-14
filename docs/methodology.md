# Methodology

## Scope

This repository implements the **acoustic-only M2-style baseline** for the HUSTmotor motor experiments. The objective is supervised six-class classification under leave-one-speed-out domain generalization.

## Upstream preprocessing

The processed dataset is obtained from Gentian Qinami's project repository:

https://github.com/Genti24-bit/ISCF_Project/

Metadata:

```text
win=1024
vib=X
norm=zscore
split=block
fs=25600
```

For the acoustic branch, each model input is therefore a 1024-point, z-score-normalized sound window corresponding to 40 ms at 25.6 kHz.

The block-based split differs from the paper's stated random 800/200 split. It was selected upstream to reduce leakage between related/overlapping windows from the same acquisition block.

## Domain-generalization tasks

| Task | Source speeds | Unseen target |
|---|---|---|
| 9 | 10, 20, 30 Hz | 5 Hz |
| 10 | 5, 20, 30 Hz | 10 Hz |
| 11 | 5, 10, 30 Hz | 20 Hz |
| 12 | 5, 10, 20 Hz | 30 Hz |

For each task:
- training: 14,400 acoustic windows;
- source validation: 3,600 windows;
- unseen target: 1,200 windows.

The target split is not used for training, tuning, or checkpoint selection.

## Architecture

The feature extractor follows Table 1 of Zhao et al. (2025):

| Block | Channels | Kernel | Stride | Pool |
|---|---:|---:|---:|---:|
| 1 | 1 → 16 | 64 | 1 | 2 |
| 2 | 16 → 32 | 16 | 1 | 2 |
| 3 | 32 → 64 | 5 | 1 | 2 |
| 4 | 64 → 64 | 5 | 1 | 2 |

Implementation decisions not fully specified by the paper:
- convolution padding: `padding=0`;
- layer order: `Conv → BatchNorm → ReLU → MaxPool`.

With a 1024-point input and valid convolutions, the final feature map is `(64, 55)`, which is flattened to 3520 features before the six-class linear classifier.

## Softmax and loss

The paper reports a Softmax state classifier. In PyTorch, the model returns raw logits because `CrossEntropyLoss` expects logits and internally performs the appropriate log-softmax operation. Explicit Softmax is used only when probabilities are required.

## Training

- Adam optimizer;
- 50 epochs;
- batch size 256;
- ten independent seeds;
- paper learning-rate schedule:

\[
lr = \frac{0.0005}{(1 + 10p)^{0.75}}.
\]

The checkpoint with the highest **source-validation accuracy** is restored before the unseen target is evaluated.

## Reproducibility

The Colab notebook:
- verifies the preprocessed dataset SHA-256;
- stores the preprocessing repository commit;
- seeds Python, NumPy, and PyTorch;
- requests deterministic cuDNN behavior;
- stores per-seed checkpoints and CSV results.
