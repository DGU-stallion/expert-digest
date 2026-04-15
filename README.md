# ExpertDigest

ExpertDigest is a local expert-content knowledge distillation project. The MVP
will focus on importing local expert articles, storing them, retrieving evidence,
answering questions, recommending original texts, and generating a Markdown
learning handbook.

This repository is currently initialized for phase 0 only: project structure,
configuration, tests, linting, and Git workflow.

## Requirements

- Python 3.11+
- Git

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development tools:

```powershell
python -m pip install -e ".[dev]"
```

## Verify

Run tests:

```powershell
python -m pytest
```

Run lint checks:

```powershell
python -m ruff check .
```

Verify the package import:

```powershell
python -c "import expert_digest; print(expert_digest.__version__)"
```

## GitHub Remote Setup

After creating an empty repository on GitHub, connect this local repository with
one of the following commands.

HTTPS:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

SSH:

```powershell
git remote add origin git@github.com:<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

If `origin` already exists, replace it:

```powershell
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
```
