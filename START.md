# START: playbook for the agent

You're helping someone decide whether deterministic retrieval is worth setting up for their notes, and then setting it up if they want it. Scripts in this folder do the mechanical work: measuring, converting, indexing, searching, installing, running the eval. Your job is to explain what's happening, ask the questions only the person can answer, and adapt when their machine does something unexpected. Don't reimplement what a script already does, and don't improvise another retrieval method.

Read this whole file before you start. Run commands from this folder.

## Ground rules

- Never write to the person's notes folder. Every command here that reads it is read-only, and the conversion works on a copy.
- Never edit `~/.claude/settings.json`, `~/.claude.json`, `~/.codex/hooks.json`, `~/.codex/config.toml` or any other settings file by hand. `kit.py install` does it, with a backup. Show the person the paths it will edit before you run it.
- API keys go in `.env`, which the person edits themselves. Never ask them to paste a key into the chat, and never print `.env`.
- Ask before anything that takes noticeable time, uses their Claude or Codex usage, downloads more than a few megabytes, or changes a settings file. Say the cost in plain terms when you ask.
- Ask one question at a time. If your harness has a question tool, use it.
- Report what the numbers say, including when they say the kit isn't worth it. If a command fails, read its output, fix the cause, and tell the person what happened. If the same thing fails three times, stop and explain what you tried.
- Linux and Windows are untested. Say so if that's what they're on, and carry on carefully. Codex support (the hook and the search tool) follows Codex's own help text and a working config but has never been run live, so say that too if they use Codex.

## Step 0: find a Python

Everything needs Python 3.11 or newer, and a Mac's built-in `python3` is 3.9. Find one:

1. Try `python3.13 --version`, then `python3.12`, then `python3.11`, then `python3`. Use the first that reports 3.11 or newer.
2. If none does, check for Homebrew (`brew --version`). If it's there, offer `brew install python@3.13` (a few minutes, a few hundred MB). If it isn't, send them to python.org to install Python 3.13, then try again.
3. In every command below, `<py>` means the interpreter you found, for example `python3.13`. Never use bare `python`, which may not exist or may be Python 2 or 3.9.

## Stage 1: is this worth it? (read-only)

1. Find the notes. Look for folders that contain `.obsidian`, or ask for the path to their markdown folder, and confirm which one if there's more than one.
2. Measure it:
   ```
   <py> kit.py assess "<vault path>" --json okf/reports/assess.json
   ```
   It needs only the standard library and writes nothing into the vault. If it says no markdown notes were found, the path is wrong: stop and ask again.
3. Tell the person the verdict in your own words.
   - The token count is an estimate (characters divided by four), and the monthly activity comes from file dates, which are only a proxy.
   - The outcome is based on size in tokens only: under about 200,000 skip, up to about 5 million optional, above that worth it if cost or speed matter.
   - Read out the benchmark line in the verdict: Mason's benchmark found browsing the most accurate way to search notes at every size, and injection cheaper and usually a few points behind, so the case for the kit is cost and speed. If browsing already works for them, it may be enough.
   - If the verdict is "skip", say their current setup is fine and ask whether they still want to try it. Don't talk them into stage 2.
4. Offer the mini test, which is optional. First run `<py> kit.py minitest estimate --harness claude` (or `codex`) and read them the time and usage range. It makes ten headless calls on their own subscription or API usage. If they say yes:
   ```
   <py> kit.py minitest sample "<vault path>" --out okf/reports/sample.json
   ```
   Read `sample.json`: five random notes with excerpts. Write one question per note that only that note answers, phrased as a person would ask it rather than copying the note's wording. Save them as `okf/reports/questions.json`:
   ```json
   {"version": 1, "cases": [{"id": "q1", "question": "...", "gold": ["relative/path.md"], "expect_none": false}]}
   ```
   Show the five questions and let them swap in their own. Then run:
   ```
   <py> kit.py minitest run "<vault path>" --questions okf/reports/questions.json --harness claude --out okf/reports/stage1-results.json
   ```
   The runs use isolation flags so their hooks, MCP servers and skills don't fire. Report the table it prints, and call five questions a sanity check, not a benchmark.

Stop here if the person doesn't want stage 2.

## Stage 2: build it (opt-in)

5. Set up. Ask which embeddings they want.
   - Ollama with `bge-m3` is free and local. The model is about 1.2 GB, so ask before pulling it.
   - OpenAI's `text-embedding-3-small` sends their note text and every prompt they type to OpenAI. Say that plainly before they choose it. They copy `.env.example` to `.env`, run `chmod 600 .env` and paste the key themselves.
   - Keyword-only needs neither.

   Edit `provider` in `config.toml` if they pick something other than Ollama. Then:
   ```
   <py> kit.py setup
   ```
   This creates `.venv` here and installs `pyyaml` and `numpy`. It ends with a check. If Ollama or the model is missing, say what to run (`ollama pull bge-m3`) and wait.
6. Plan the conversion:
   ```
   <py> kit.py migrate plan "<vault path>"
   ```
   It copies the notes into `okf/raw/notes/` as evidence and proposes a bundle (the top-level folder), a type (Meeting, Decision, Fix, SOP, Policy, Memory, Project, Client or Reference), an id, a description and tags for each. Rules made the first guess, not a model. Review `okf/staging/migration-plan.json` with the person, starting with the summary table and the flagged notes, and explain what a bundle and a type are before you ask them to approve anything.
   - Notes in journal, diary and similar folders are marked `scope: me`. Those are retrievable by default, so ask whether they want personal notes included. If not, remove `"me"` from `scopes` in `config.toml`.
   - Correct mistakes with `<py> kit.py migrate set "Journal/*" --bundle journal --type Memory --scope me`, or leave a folder out with `<py> kit.py migrate set "Archive" --exclude`.
   - The corpus folder must be new and separate from the notes. If the plan refuses it, don't point it at a parent of the notes.
7. Apply it:
   ```
   <py> kit.py migrate apply
   ```
   The linter runs as a gate, and if it fails nothing is written to the corpus. Read the problems, fix the plan and run it again. A note containing something shaped like a vendor API key is kept out of the corpus and listed, but it is still copied to `okf/raw/notes/`, and lines like `password: ...` aren't detected. Tell the person both. `<py> kit.py migrate revert` undoes the whole conversion.
8. Index it. The first index takes minutes: about two minutes per 1,000 notes with Ollama, roughly half an hour for 10,000. That is longer than your shell's default timeout, so run it in the background and poll:
   ```
   <py> kit.py reindex > .state/reindex.log 2>&1 &
   ```
   Check `.state/reindex.log` every half minute or so. It prints `embedded n/total sections` as it goes and a summary at the end. If a run was killed, the next one takes over its lock right away. Tell the person the expected time from the section count in the first lines.
9. Write a golden set so the settings get measured on their own notes.
   - Run `<py> kit.py eval sample --n 25` and read the excerpts.
   - Write about 25 questions into `okf/eval/golden.json`, in the mini test's format plus a `kind` (`direct`, `paraphrase` or `entity`). Use the original vault paths from the sample as `gold`.
   - Make at least a third of them unanswerable by this vault, with `"expect_none": true` and an empty `gold`. Those stop the thresholds from drifting toward "inject everything".
   - Show a handful to the person and let them correct any that are wrong.
   - The eval also adds 50 generic off-topic prompts that ship with the kit. You can add about 10 near-miss prompts that reuse words from their notes without being answerable from them, saved in the same format to `okf/eval/nearmiss.json` with `"expect_none": true` and `"kind": "nearmiss"`.
10. Run the eval and tune:
    ```
    <py> kit.py eval
    <py> kit.py eval tune
    ```
    Start with what the numbers mean. With the shipped settings, expect a passage to be injected for only about a third of the questions the notes answer, mostly the ones that share words with a note, and the agent's own browsing covers the rest. False injection (something added for a prompt the notes don't answer) is the cost they'd feel most, because it wastes tokens and can confuse the model.
    - The first command prints a table that ablates the optional parts (keyword only, semantic only, no thresholds, the shipped settings, the exact-title channel), with recall next to false injection.
    - The second prints the best recall at or under 10% false injection on every set.
    - Raising `min_cosine` stops more irrelevant injections and loses paraphrase matches. Set the values in `config.toml`, run the eval again to confirm, and keep an optional part only if the table shows it earns its place.

    Two optional parts are off by default, and neither is a clear win. The pointer tier (`pointers_enabled = true`) lists a few candidate notes for the agent to open, and hasn't been tested cleanly. The reranker (`<py> kit.py setup --with-reranker`, a multi-gigabyte download, about a second per prompt) raised recall to 55% in my tests but injected for more than half of near-miss prompts. Offer them only if the person's own eval shows a gain.
11. Install. Ask which tool they use, whether they want it everywhere or in one project, and which mode:
    - `inject`: a hook adds passages to prompts (the default).
    - `tool`: the agent gets a `search_notes` tool and decides when to call it.
    - `both`.

    In the kit's live test all three answered about equally well and used a quarter to two fifths fewer tokens than the agent browsing alone, so inject is the default (no long-running server). Then:
    ```
    <py> kit.py install --harness claude
    ```
    - Use `--harness codex` for Codex, and say it hasn't been tested live. Without `--harness` the installer uses whichever tool it finds.
    - Add `--mode tool` or `--mode both` for the search tool.
    - Add `--claude-settings <path>` or `--codex-hooks <path>` for a project-level hook file, and `--claude-mcp .mcp.json` or `--codex-config <path>` for the tool.
    - Tell them the backup file it prints. Claude Code needs a new session (restart it) to load the change. Codex asks them to run `/hooks` once to trust a hook.
    - The hook never blocks a prompt: on any error it injects nothing, and it stops waiting after a few seconds.
12. Verify. Rerun their five questions with retrieval on, then build the report:
    ```
    <py> kit.py verify "<vault path>" --questions okf/reports/questions.json --harness claude --out okf/reports/stage2-results.json
    <py> kit.py report --before okf/reports/stage1-results.json --after okf/reports/stage2-results.json
    ```
    Skip `--before` if they never ran the mini test. Open `report.html` for them, and tell them it contains excerpts of their notes, so it isn't something to share.
13. Hand over. Tell them how to add knowledge (`<py> kit.py remember problem|decision|preference|fact "..."`), how to see what retrieval would do for any text (`<py> kit.py search "..."`), and where the settings are (`config.toml`). Remind them that `<py> kit.py uninstall` removes the hook and search-tool entries from every file it was installed into, deletes the venv and index, keeps their converted notes unless they add `--remove-corpus`, and never touches their original notes.
