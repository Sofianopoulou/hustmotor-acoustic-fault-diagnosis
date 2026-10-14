# Result files

- `acoustic_10runs_raw.csv` — one row per task/seed experiment.
- `acoustic_10runs_summary.csv` — mean/std target accuracy and mean best epoch.
- `acoustic_tasks_9_12_seed0.csv` — first complete four-task run with seed 0.
- `history_task9_seed0.csv` — epoch-level Task 9 training history.
- `acoustic_final_comparison_10runs.csv` — final paper-vs-ours task comparison including min/max and percentage-point difference.
- `acoustic_class_metrics_10runs_summary.csv` — mean/std precision, recall and F1 for each class and task.
- `acoustic_macro_f1_means.csv` — macro-F1 mean computed from the class-level ten-run summaries.

Raw datasets and `.pt` model checkpoints are intentionally excluded from GitHub.
