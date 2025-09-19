# -----------------------------
# Project settings
# -----------------------------
PYTHON_VERSION ?= 3.12
POETRY ?= poetry
PROJECT_NAME ?= repo-skeleton

POETRY_EXISTS := $(shell command -v $(POETRY) 2>/dev/null)

.PHONY: help bootstrap configure install venv shell update lock clean list freeze test

help: ## Show available targets
	@awk 'BEGIN {FS=":.*##"} /^[a-zA-Z0-9_\-]+:.*##/ {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## (Optional) Install Poetry via pipx if not present
ifndef POETRY_EXISTS
	@echo "Installing poetry via pipx..."
	pipx install poetry
else
	@echo "Poetry already installed: $(shell poetry --version)"
endif

configure: ## One-time: use in-project .venv and set python version
	$(POETRY) config virtualenvs.in-project true
	@echo "Configured Poetry to use in-project .venv"

install: configure ## Install project and all dependencies into .venv
	$(POETRY) env use python$(PYTHON_VERSION)
	$(POETRY) install --no-interaction --no-root

venv: install ## Create/refresh the .venv (alias for install)

shell: ## Enter the Poetry-managed virtual environment
	$(POETRY) shell

update: ## Update all dependencies to the newest compatible versions
	$(POETRY) update

lock: ## Re-resolve and write the lock file without updating installed pkgs
	$(POETRY) lock --no-update

list: ## Show installed packages (inside the venv)
	$(POETRY) run pip list

freeze: ## Export concrete versions to requirements.txt (useful for other tools)
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	@echo "Wrote requirements.txt"

clean: ## Remove the local virtualenv and caches
	@echo "Removing .venv and build artifacts..."
	rm -rf .venv dist build *.egg-info .pytest_cache __pycache__
	@echo "Clean complete. Run 'make install' to recreate the env."

test: ## Run tests (if/when you add them)
	$(POETRY) run python -c "print('No tests yet 🚧')"
