BUILD_DIR_TEST := build_test

.PHONY: install
install:
	@echo "Creating virtual environment and installing dependencies using uv..."
	uv venv
	uv lock
	uv sync --all-groups
	uv run pre-commit install

.PHONY: build
build:
	@echo "Installing build dependencies..."
	uv pip install scikit-build-core numpy
	@echo "Compiling C-extensions..."
	uv pip install --no-build-isolation -e . -Ccmake.build-type=Release
	@echo "Build complete."

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR_TEST)
	rm -f  rnd/trapezoid_rnm.c
	find . -name "*.so"  -delete
	find . -name "*.pyd" -delete
	rm -rf _skbuild build dist *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf .venv

.PHONY: rebuild
rebuild: clean install build

.PHONY: build-tests
build-tests:
	# We don't need OpenMP for unit tests and hence disable it here
	@echo "Configuring test build..."
	cmake -S . -B $(BUILD_DIR_TEST) -DCMAKE_BUILD_TYPE=Debug -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON
	@echo "Compiling C unit tests..."
	cmake --build $(BUILD_DIR_TEST)

.PHONY: test-c
test-c: build-tests
	@echo "Running C unit tests..."
	ctest --rerun-failed --output-on-failure --test-dir $(BUILD_DIR_TEST) -R "^ext/"

.PHONY: test-py
test-py:
	@echo "Running Python unit tests..."
	uv run pytest tests/ -v

.PHONY: test
test: test-c test-python

.PHONY: lint
lint:
	@echo "Running linter with ruff..."
	uv run ruff format . --config pyproject.toml
	@echo "Running checks with ruff..."
	uv run ruff check . --config pyproject.toml

.PHONY: ci
ci:
	@echo "This target attempts to simulate running tests and linting"
	$(MAKE) test
	$(MAKE) lint

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install      - Set up virtual environment and install all dependencies"
	@echo "  build        - Compile C extensions (editable install)"
	@echo "  rebuild      - Clean and recompile everything from scratch"
	@echo "  clean        - Remove all build artifacts (.so, dist, .venv)"
	@echo "  test-c       - Build and run C unit tests"
	@echo "  test-python  - Run Python unit tests"
	@echo "  test         - Run C and Python unit tests"
	@echo "  lint         - Run ruff and cython-lint"
	@echo "  ci           - Full pipeline: rebuild + test-all + lint"
