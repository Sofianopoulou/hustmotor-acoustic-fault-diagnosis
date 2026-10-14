# Publish this repository on GitHub

## 1. Create an empty GitHub repository

Suggested name:

```text
hustmotor-acoustic-domain-generalization
```

Do not initialize it with a README, `.gitignore`, or license because those files are already included here.

## 2. Push this folder

From a terminal opened in this folder:

```bash
git init
git add .
git commit -m "Initial public release of acoustic HUSTmotor baseline"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/hustmotor-acoustic-domain-generalization.git
git push -u origin main
```

## 3. Recommended repository settings

- Add topics:
  - `pytorch`
  - `deep-learning`
  - `fault-diagnosis`
  - `domain-generalization`
  - `signal-processing`
  - `acoustic`
  - `1d-cnn`
  - `hustmotor`
- Pin the repository on your GitHub profile if you want to use it as a portfolio project.
- In the repository About section, use a short description such as:

> Acoustic-only 1D CNN baseline for cross-speed electrical motor fault diagnosis on HUSTmotor, with 10-run domain-generalization evaluation.

## 4. Do not commit

The `.gitignore` already excludes:
- `*.npz` datasets;
- `*.pt` / `*.pth` checkpoints;
- virtual environments;
- LaTeX build files.

The processed data remain available through the linked preprocessing repository.
