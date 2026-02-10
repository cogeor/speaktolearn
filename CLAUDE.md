# CLAUDE.md

## Python Environment

This project uses `uv` with a `.venv` virtual environment. Run Python commands with:
```bash
uv run python <script>
```

Or activate the venv directly:
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## Delegate

This project uses the Delegate plugin for spec-driven development.

**A loop is a focused unit of work that results in exactly one commit.** Each loop has a draft describing what to do, acceptance tests for verification, and a clear scope. Loops are the fundamental unit — everything in delegate either creates loops or implements them.

**Commands:**
| Command | Purpose |
|---------|---------|
| `/dg:study [model] [theme]` | Explore codebase, produce drafts in `.delegate/loop_plans/` |
| `/dg:work [args]` | Implement loops in `.delegate/loops/` (plan, execute, test, commit) |

**Workflow:**
1. `/dg:study` — explores codebase, web, tests; produces drafts with feature proposals, implementation plans, and test approaches
2. `/dg:work plan` — review proposed drafts
3. `/dg:work 02` or `/dg:work add logout button` — implement from drafts (plan, execute, test, commit each)

Study drafts live in `.delegate/loop_plans/`. When `/dg:work` executes a draft, it creates a full loop in `.delegate/loops/` with detailed plans, implementation records, and test results.
