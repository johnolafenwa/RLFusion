# Agent Guidelines

This repository contains a minimalist post-training library for LLMs (SFT, GRPO/RLVR, on-policy distillation).

## Scope
- Keep changes focused on trainers, datasets, and examples.
- Prefer small, well-contained updates with tests when behavior changes.

## Style
- Follow existing code style and typing conventions.
- Use `ruff` for linting and `ty` for type checking.
- Keep comments concise and only when they clarify non-obvious logic.

## Commands
- Lint: `uv run ruff check .`
- Type check: `uv run ty check src tests`
- Tests: `uv run pytest`
