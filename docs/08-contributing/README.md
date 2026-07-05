# Contributing to ReviewSentinel

Thank you for your interest in contributing to ReviewSentinel! This document outlines the process for proposing changes, setting up your environment, and meeting the project's quality standards.

---

## 1. Development Environment

Before writing code, ensure you have a working local environment.

1. Fork the repository and clone your fork.
2. Follow the [Getting Started](../02-guides/local-development.md#getting-started) guide to install prerequisites (Docker, Git).
3. Set up a Python virtual environment (recommended: Python 3.10+):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Verify you can bring the Docker Compose stack up successfully.

---

## 2. Branching Strategy

We use a standard feature-branch workflow.

- **`main`** is the stable, protected branch. Direct commits are blocked.
- Create feature branches from `main` using the format: `type/short-description`.
  - Examples: `feature/semantic-search`, `fix/minio-race-condition`, `docs/update-model-card`

```bash
git checkout main
git pull
git checkout -b feature/your-feature-name
```

---

## 3. Commit Convention

ReviewSentinel follows a loose Conventional Commits standard. Your commit messages should be descriptive and explain *why* the change was made, not just *what* files were changed.

**Format:** `type(scope): clear description`

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `refactor`: Code changes that neither fix a bug nor add a feature
- `test`: Adding or correcting tests
- `chore`: Infrastructure, dependency updates, or pipeline changes

**Example:**
```
fix(api): handle missing text field gracefully in batch requests

Previously, submitting a batch request without the optional `text` 
field caused an IndexError during sequence concatenation. This adds 
a fallback to empty strings.
```

---

## 4. Code Quality & Formatting

All Python code must pass linting and formatting checks before a PR can be merged. We use `ruff` for both.

To check your code locally:
```bash
# Check for linting errors
ruff check src/ scripts/

# Automatically format code
ruff format src/ scripts/
```

> [!IMPORTANT]
> The GitHub Actions CI pipeline (`pr-checks.yaml`) will fail your PR if `ruff check` reports any violations. Run it locally before pushing!

---

## 5. Testing Requirements

If you add a new feature, you must add tests demonstrating that it works. If you fix a bug, you must add a test demonstrating that the bug is fixed (preventing regression).

Run tests locally:
```bash
pytest tests/ -v
```

*(Note: The test suite is currently undergoing a refactor. Ensure any existing tests pass, but we do not enforce a strict Codecov percentage block at this time.)*

---

## 6. Documentation Requirements

Code changes rarely happen in a vacuum. If your PR changes how the system behaves, you must update the documentation:

- **New API endpoint or field?** Update `docs/api/reference.md`.
- **New configuration variable?** Update `configs/pipeline_params.yaml` and the relevant ML or Architecture doc.
- **New dependency?** Ensure it is in `requirements.txt` and the Dockerfiles.
- **Major architectural change?** Propose an ADR (Architecture Decision Record) in `docs/decisions/`.

**Link Checking:**
We strongly recommend running a markdown link checker locally to ensure your documentation changes don't introduce broken links:
```bash
npx markdown-link-check docs/**/*.md README.md
```

---

## 7. The Pull Request Process

1. Push your branch to your fork.
2. Open a Pull Request against the `main` branch of the upstream repository.
3. Fill out the Pull Request template provided.
4. **CI Checks:** Wait for GitHub Actions to complete. Your PR must pass all checks (linting, tests, docker build).
5. **Review:** A maintainer will review your code. Be prepared to make requested changes.
6. **Merge:** Once approved and CI passes, a maintainer will squash-merge your PR.

### PR Checklist
Before requesting review, ensure:
- [ ] You have run `ruff format` and `ruff check`
- [ ] You have written tests for your changes
- [ ] You have updated the relevant documentation in `docs/`
- [ ] Your code does not break the `docker-compose up` flow
