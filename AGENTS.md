# Instructions for coding agents

Read PROJECT.md first. Work one milestone at a time, and finish its tests and results file before starting the next.

## Conventions

- Python 3.12 with uv, a `src/injclf/` package layout, type hints, ruff and pytest.
- Set seeds everywhere and record them in `configs/`. Every run must be reproducible from a config file.
- Scripts write metrics to `results/*.json`. Nothing else writes there, and nobody edits those files by hand.

## Rules

- Every number in README.md must come from a file in `results/` that a real run produced. Never type, round up or estimate a metric.
- The held-out set built from `../owasp-rag-eval/data/attacks.jsonl` never touches training or threshold selection.
- Never modify files in `../owasp-rag-eval/data/`.
- Commit small, working steps with the real current date. Never set `GIT_AUTHOR_DATE` or `GIT_COMMITTER_DATE`, and never rewrite pushed history.
- Write commit messages, comments and docs as plain descriptions of the change. Do not mention AI tools or assistants anywhere.
- Do not add docs beyond README.md and PROJECT.md. No blog posts, papers or build diaries.
- Keep the Hugging Face token in `.env`, which is gitignored. Never commit it.
- At the end of each session, explain in chat what you changed and why, in plain language, so the user can explain it in an interview.
