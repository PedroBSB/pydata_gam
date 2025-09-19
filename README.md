# PyData Seattle 2025: Generalized Additive Models: Explainability Strikes Back

This repository uses **[Poetry](https://python-poetry.org/)** to manage dependencies and a local, in-project virtual environment at `./.venv`.

## Prerequisites
- Python 3.12 (or set a different version via `PYTHON_VERSION=...`)
- Poetry installed (recommended via `pipx install poetry`), or run `make bootstrap` to install via pipx

## Quick Start

```bash
# 1) (Optional) Install Poetry if you don't have it
make bootstrap

# 2) Create the local .venv and install dependencies
make install            # or: make venv

# 3) Activate a subshell inside the virtual environment
make shell

# 4) Verify packages are present
make list
```

## Common Tasks

- **Update dependencies**:
  ```bash
  make update
  ```

- **Lock** dependency graph without updating installed packages:
  ```bash
  make lock
  ```

- **Export** a `requirements.txt`:
  ```bash
  make freeze
  ```

- **Clean** (remove `.venv`, build artifacts):
  ```bash
  make clean
  ```

## Notes

- The package **pyGAM** is installed from PyPI as `pygam`.
- You can set a different Python version, e.g.:
  ```bash
  make install PYTHON_VERSION=3.12
  ```
- One-off commands: `poetry run python -c "import numpy; print('ok')"`
