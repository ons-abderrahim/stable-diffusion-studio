# Contributing to Stable Diffusion Studio

Thank you for considering contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/yourusername/stable-diffusion-studio.git
cd stable-diffusion-studio
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

## Workflow

1. Fork the repo and create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes with clear, focused commits.
3. Add or update tests in `tests/`.
4. Run the test suite:
   ```bash
   pytest tests/ -v --cov=src
   ```
5. Lint and format:
   ```bash
   black src/ scripts/ tests/
   isort src/ scripts/ tests/
   ruff check src/
   ```
6. Open a Pull Request against `main`.

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add interpolation between prompts`
- `fix: handle empty prompt gracefully`
- `docs: update fine-tuning guide`
- `test: add LoRA trainer unit tests`
- `refactor: simplify scheduler selection`

## Areas Where Help is Welcome

- Adding new fine-tuning techniques (Control-Net, IP-Adapter)
- Improving evaluation metrics
- More Gradio UI features
- Additional prompt style presets
- Documentation improvements and examples
- Performance benchmarks

## Code Style

- Python 3.10+, type hints everywhere
- `black` formatting, `isort` imports
- Docstrings on all public classes and functions
- Unit tests for new functionality

## Questions?

Open a [GitHub Discussion](https://github.com/yourusername/stable-diffusion-studio/discussions) — we're friendly!
