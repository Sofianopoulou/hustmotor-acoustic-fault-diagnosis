# Results

## Task-level accuracy

| Task | Target | Our accuracy | Paper M2 | Difference |
|---|---:|---:|---:|---:|
| 9 | 5 Hz | 26.36 ± 3.00% | 49.23% | −22.87 pp |
| 10 | 10 Hz | 60.96 ± 4.02% | 61.89% | −0.93 pp |
| 11 | 20 Hz | 57.29 ± 4.88% | 59.14% | −1.85 pp |
| 12 | 30 Hz | 49.19 ± 2.80% | 28.13% | +21.06 pp |

Overall across Tasks 9–12:

**48.45 ± 2.14%**

Reference paper M2 average:

**49.60%**

The global difference is only −1.15 percentage points, but the task-wise behavior is substantially different.

## Interpretation

### Task 9 — unseen 5 Hz

This is the strongest failure case. Source-validation accuracy remains high, yet unseen-target accuracy falls to 26.36 ± 3.00%.

Mean class recall:
- Healthy: 18.30%
- Bearing fault: 59.15%
- Bowed rotor: 0.85%
- Broken rotor bars: 38.05%
- Rotor misalignment: 2.85%
- Voltage unbalance: 38.95%

The model frequently maps healthy samples to voltage unbalance and bowed-rotor samples to broken rotor bars.

### Task 10 — unseen 10 Hz

Task 10 is close to the paper's M2 value. Bearing fault reaches 99.95% recall, and rotor misalignment reaches 100% recall. The latter has lower precision, indicating that other classes are also attracted to the rotor-misalignment label.

### Task 11 — unseen 20 Hz

The overall result is again close to the paper. Bearing fault reaches 100% recall, but healthy recognition collapses to 3.05% recall; many healthy samples are predicted as broken rotor bars.

### Task 12 — unseen 30 Hz

The baseline substantially exceeds the reference paper's M2 result. Bearing fault and bowed rotor generalize strongly, while healthy and broken rotor bars remain difficult.

## Macro-F1 means

| Task | Macro-F1 |
|---|---:|
| 9 | 23.28% |
| 10 | 57.83% |
| 11 | 57.45% |
| 12 | 47.72% |

## Main conclusion

The experiment supports a key motivation for the wider multimodal project: **source-domain classification quality does not guarantee robustness to unseen operating conditions**.

The acoustic modality contains useful fault information, but fault classes respond very differently to speed shift. This motivates combining acoustic features with vibration representations in the later multimodal stage.
