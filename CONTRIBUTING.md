# Contributing

Thanks for your interest in improving this project.

## Setup

```bash
git clone https://github.com/ayush-mahansaria/rag-chatbot-demo.git
cd rag-chatbot-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v
```

## Making Changes

1. Create a branch: `git checkout -b feat/your-feature`
2. Make your changes and run tests
3. Commit with a clear message: `git commit -m "feat: description"`
4. Open a pull request on GitHub

## Commit Style

Use conventional commit prefixes:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `chore:` — maintenance (deps, config, etc.)
