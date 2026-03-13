# MoleculeInsight Copilot Instructions

## Critical Rules - Git Operations

### 🚫 ABSOLUTE RESTRICTION
- **ONLY** allowed git command: `git status` (read-only, informational)
- **NEVER** run without explicit user permission:
  - `git add`
  - `git commit`
  - `git push`
  - `git pull`
  - `git reset`
  - `git branch`
  - Any other git command that modifies the repository

### ⚠️ Protocol
- **Always ask the user first** before executing ANY git command except `git status`
- Wait for explicit approval before proceeding
- Example: "Should I enable this feature? Permission to run `git add`?"

## Stack
-  ⁠Streamlit for UI
-⁠  ⁠uv for dependency management, ruff for linting

## Key rules
- ⁠All new dependencies go through ⁠ uv add ⁠
- ⁠Follow ruff formatting rules
- Add type hints to all functions and variables where possible
- Write docstrings for all functions and classes
- Full logging and error handling for all operations
- Never put import statements inside functions; all imports must be at the top of the file
- Remove basic examples from docstrings to keep them concise; focus on describing parameters, return values, and behavior instead
- Do not add icons unless explicitly requested
- Replace `use_container_width` with `width`. `use_container_width` will be removed after 2025-12-31. For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.



