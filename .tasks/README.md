# Tasks

Store implementation tasks for AI agents.

Workflow:
1. Create a task file from task-template.md.
2. Define requirements and acceptance criteria.
3. Implement code changes.
4. Run local C++ Build & Tests.
5. Run local Python Lint (Ruff).
6. Verify against Checklist.
7. Execute local agent_pipeline (ps1/sh) to trigger git commands.
8. Push to GitHub (Triggers `ci.yml` / `python-ci.yml`).
