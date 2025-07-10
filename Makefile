.DEFAULT_GOAL := help

uv:
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	#@uv venv --python '3.12'


.PHONY: install
install: uv ## Install a virtual environment
	@uv venv --python '3.12'
	@uv pip install --upgrade pip
	@uv sync --all-extras --frozen


.PHONY: fmt
fmt: uv ## Run autoformatting and linting
	@uvx pre-commit run --all-files


.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f


.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort


.PHONY: test
test: install  ## Run pytests
	@uv pip install pytest
	@uv run pytest src/tests
