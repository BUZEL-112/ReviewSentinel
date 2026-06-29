# ADR 008: Documentation Structure

## Status
Accepted

## Context
The previous documentation structure grew organically and mixed concepts, tutorials, and references in ways that made it difficult to navigate. Different personas (developers, operators, data scientists) had to hunt across multiple directories to find relevant information. Furthermore, `README.md` became bloated with architectural details, limitations, and design decisions, detracting from its primary role as a quick landing page.

## Decision
We are adopting a persona-driven, numbered folder structure for all documentation:
1.  **Numbered Directories (`00-quickstart` to `08-contributing`)**: Enforces a logical reading sequence and groups related topics cleanly.
2.  **Persona-driven Routing (`INDEX.md`)**: A central index routes users based on what they want to achieve (e.g., Developer vs. Operator vs. ML Engineer).
3.  **Thin README**: `README.md` is strictly a landing page (project name, one-liner, badges, minimal quickstart, and a link to `INDEX.md`).
4.  **No Duplication**: Topics are documented in one place only. For instance, architecture is explained once, and all deep-dive design choices live strictly in the `06-decisions` (ADRs) folder.

## Consequences
- **Positive:** New users and contributors can find relevant documentation faster.
- **Positive:** `README.md` is less overwhelming.
- **Negative:** Existing links from external sources to the old folder structure (`docs/architecture/`, `docs/ml/`, etc.) will break and result in 404s.
