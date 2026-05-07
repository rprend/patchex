# Codex Notes

- If you are working inside this Git repository, always commit and push after completing a change.
- If the repository has an existing deployment target or deploy workflow, run the deploy after the change as well.

## Patchex Run Inspection

- When the user says something like "this run failed <run_id>" or asks what happened in a Patchex run, first collect the full run harness bundle by ID. Do not infer from the HTML page shell alone.
- Preferred local command:
  - `scripts/inspect_run.py <run_id>`
- If the site server is running, these API endpoints return the same harness context:
  - `GET http://127.0.0.1:8765/api/reconstructions/<run_id>`
  - `GET http://127.0.0.1:8765/api/reconstructions/<run_id>/logs`
  - `GET http://127.0.0.1:8765/api/reconstructions/<run_id>/harness`
- The harness bundle includes the inferred lifecycle status, manifest, `events.jsonl`, `raw_subprocess.log`, `logs/*.log`, artifact URLs, and inline content for text artifacts such as prompts, answers, JSON reports, summaries, and logs.
- Local fallbacks:
  - `scripts/show_run.py <run_id>`
  - `scripts/tail_run.py <run_id> --no-follow`
  - `scripts/tail_run.py <run_id> --agent <agent_name> --no-follow`
  - run files live under `text2fx_gemini/ui_runs/<run_id>/`.

## Secrets and API keys

- Never ask the user to paste API keys, tokens, passwords, or other secrets into chat.
- When a task needs a secret that is not already available, use the macOS hidden-input helper at `~/.codex/scripts/ask-secret.sh`.
- Default usage: `~/.codex/scripts/ask-secret.sh ENV_NAME ~/.codex/secrets.env "Enter ENV_NAME"`, then load it with `set -a; source ~/.codex/secrets.env; set +a` for the command that needs it.
- For repo-local secrets, pass the target env file explicitly, for example `~/.codex/scripts/ask-secret.sh OPENAI_API_KEY .env "Enter OpenAI API key"`.
- The helper writes env files with `0600` permissions. Do not print secret values, commit secret files, or include secret values in final answers.

## gstack

- Gary Tan's gstack skills are installed globally in `~/.codex/skills/gstack`.
- Use the `/browse` skill from gstack for web browsing and QA-style browser automation work when available.
- Never use `mcp__claude-in-chrome__*` tools when the gstack `/browse` skill can handle the task.
