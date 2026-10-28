# Deterministic RAG kit

This kit puts a search step in front of your prompts. When you ask Claude Code or Codex something, a script searches your notes first and adds the passages that match, along with the path of each note, so the agent starts with them already in front of it.

It goes with my video "Deterministic RAG vs Agentic Knowledge Searches". It works on a folder of markdown notes, including an Obsidian vault, and it never writes to that folder.

## Get it

```
git clone https://github.com/Masonpfitz/deterministic-rag-kit
cd deterministic-rag-kit
```

Then open Claude Code or Codex in that folder and paste:

```
Read START.md and follow it.
```

The agent does the setup and asks before anything that takes time, uses your Claude or Codex usage, or downloads a lot. The folder also contains `CLAUDE.md` and `AGENTS.md`, which point an agent at `START.md`, so any agent you open here will start the setup. If you only want to read the code, open the folder in an editor instead.

## Do you need it?

The first thing the kit does is measure your vault and tell you whether any of this is worth it. Browsing a vault with an agent's own search tools is often the most accurate option. What this kit adds is lower cost and speed, so "your setup is fine" is a common answer. The verdict is based on size in tokens: under about 200,000 tokens your agent can read everything and you can skip it, from there to about 5 million it is optional, and above that it is worth it if cost, speed or usage limits matter. The lines are in `thresholds.json`, each with its source.

## What happens

Stage 1 only reads. `assess.py` counts your notes, words, tokens, links and tags, shows how many notes you touch per month, and gives a verdict. After that you can run an optional mini test: five questions from five of your own notes, each answered twice through the headless CLI, once by an agent browsing your vault and once from keyword search results. It tells you the time and usage before it starts.

Stage 2 only happens if you say yes:

1. The kit makes a virtual environment inside this folder and installs two packages, `pyyaml` and `numpy`.
2. It copies your notes into a new folder, `okf/raw/notes/`, as evidence, then writes a converted copy into `okf/knowledge/`: one document per note, with a type, a description and tags. A linter checks the result before anything goes live, and the whole conversion can be reverted.
3. It builds a search index and, if you use embeddings, a vector file next to it.
4. It connects the search to Claude Code or Codex (see "Three ways to use it"). The installer backs up your settings first and leaves your other hooks and servers alone.
5. It runs your five questions again with retrieval on, measures retrieval on questions it can check, and writes `report.html`.

## How it works

Each note is cut at its headings into sections. Every section goes into a SQLite file for keyword search, and its embedding goes into a numpy file for similarity search. Only files that changed get redone on the next run.

For a prompt, keyword search and similarity search each return their 30 best sections, and the two lists are merged by rank (reciprocal rank fusion, k=60). A relevance check then drops weak matches, and at most four passages (4,000 tokens, 9,000 characters) go to the agent. Deprecated notes and notes past their `stale_after` date are never searched, and a note's `scope` decides whether it can be retrieved at all.

No model writes anything in this path. With the same index and a responsive embedding service, the same prompt gives the same passages. If Ollama is slow the hook falls back to keyword search for that prompt, and the background reindex can change the index between prompts, so it is not a guarantee across every situation. Every setting is in `config.toml` with a comment, which means you can change one and measure what it did.

## What to expect

On my fictional test vault (bge-m3 embeddings, 194 prompts), with the shipped settings:

| Prompt | Passages injected |
|---|---|
| Question that shares words with its note | 86% of the time |
| Lookup by a name or code | 81% |
| Question phrased very differently from the note | 2% (1 of 51) |
| Generic coding, git, chat or general-knowledge prompt | 0 to 2% |
| Prompt that reuses your notes' words but isn't answered by them | about 25% |

So the kit stays quiet on most prompts, and when you ask in different words than the note used, it usually adds nothing and your agent searches on its own. A prompt that gets passages starts with a note telling the agent they may be incomplete and to search the notes itself if they don't clearly answer.

In a live test with Claude Code on Sonnet (30 questions, 2 runs each, 1,000 notes), an agent that browsed the vault answered 92% correctly, and the three kit modes answered 95%, which is within noise. The kit modes used 25 to 39% fewer tokens. That vault is about 170,000 tokens, which the stage 1 check calls small enough to skip, and the thresholds were tuned on the same fictional prompts, so a real vault will behave differently. `python3 kit.py eval tune` shows the tradeoff for your own notes.

Mason's benchmark for the video tested this on a much larger fictional vault (Claude Sonnet, 36 questions three times each, 100 to 50,000 notes). An agent browsing the vault was the most accurate at every size, from 92% at 100 notes to 64% at 50,000. Injecting passages scored 77% to 59% (the verify header on the injected block removed a failure where the agent answered from passages that lacked the answer and stopped searching). Injection used 41 to 64% fewer tokens, 27 to 50% with the verify header, and was faster up to 10,000 notes. A search tool the agent calls itself was no more accurate than browsing and was the slowest. No point where retrieval overtakes browsing on accuracy appeared, up to about 38 million tokens. So the reason to use the kit is cost and speed, and its answers are close to browsing's but usually a few points behind. The benchmark's results aren't public yet. The full tables for the kit's own tests are in `DEV-NOTES.md`.

## Three ways to use it

| Mode | What happens | Install |
|---|---|---|
| inject (default) | A hook runs on each prompt and adds passages when something matches well enough. | `python3 kit.py install` |
| tool | Your agent gets a `search_notes` tool and decides when to call it. Nothing is added to prompts. | `python3 kit.py install --mode tool` |
| both | Passages are injected on a clear match, and the agent can still search. | `python3 kit.py install --mode both` |

In the live test all three answered about equally well, so inject is the default because it needs no long-running server. Tool mode adds an MCP server that starts with your agent and uses the same index and settings. `install` picks whichever of Claude Code and Codex is on your machine, or you can pass `--harness claude` or `--harness codex`.

## Privacy: what leaves your machine

With Ollama, nothing does. The kit's search and index run locally.

If you set `provider = "openai"`, the kit sends your note text to OpenAI when it indexes and sends every prompt you type to OpenAI when the hook runs, including code and anything sensitive you paste. Use Ollama if that isn't what you want.

Either way, passages the kit adds to a prompt go to whichever model provider your agent uses. Notes in folders like journal or diary are marked `scope: me` and are retrievable by default. Remove `"me"` from `scopes` in `config.toml` to keep them out. Two more things to know: `report.html` contains excerpts of your notes, so don't share it, and a note containing something shaped like an API key is kept out of the converted corpus but is still copied to `okf/raw/notes/`. Only vendor key formats are detected, not lines like `password: ...`.

## What you need

- Python 3.11 or newer with SQLite's FTS5. A Mac's built-in `python3` is 3.9, so use Homebrew (`brew install python@3.13`) or python.org. `START.md` has the agent find a suitable one.
- For semantic search, [Ollama](https://ollama.com) with the `bge-m3` model (about 1.2 GB, `ollama pull bge-m3`), or an OpenAI key in a `.env` file you edit yourself (read the privacy note above). With neither, the kit still works with keyword search only.
- Claude Code or Codex for the hook. The mini test also runs one of them headless.
- Time and disk: indexing 1,000 notes took about two minutes with Ollama here, and 10,000 would take roughly half an hour (extrapolated, not measured). The raw copy of your notes uses about as much disk as the notes.

## What it can't do

- Four passages is a hard cap on breadth. A question that needs ten notes gets the best four.
- Obsidian wikilinks become ordinary links where the target note is found, and plain text where it isn't. Dataview and other plugin syntax isn't understood, and attachments aren't indexed.
- Semantic search without a threshold would add something to almost every prompt, so the shipped threshold trades recall for quiet. It is set for bge-m3. Run `python3 kit.py eval tune` after you change models.
- The hook adds a little time to every prompt. My numbers are in `DEV-NOTES.md`.
- The optional reranker is a multi-gigabyte download that needs a small local server, and in my tests it injected for more than half of the near-miss prompts. The optional pointer tier hasn't been tested cleanly.

## Where it has been tested

One Apple Silicon Mac running macOS, with Python 3.13, Claude Code 2.1.312 and Codex 0.164.0. The 3.11 minimum comes from the standard library features used and wasn't run on 3.11.

Codex: the hook and MCP setup follow `codex mcp add --help` and a working Codex config, and haven't been run live. Claude Code's hook and MCP server have.

Linux and Windows are untested. The code uses `pathlib` and no shell scripts. Known gaps: on Debian and Ubuntu, creating the virtual environment needs the `python3-venv` package (setup says so if it fails), and on Windows the hook command quoting and console behavior are unverified, although the hook reads and writes UTF-8 directly.

## Adding to it later

```
python3 kit.py remember problem "Build fails on Node 22" --solution "pin to Node 20"
python3 kit.py remember decision "Use Postgres" --why "we need transactions"
python3 kit.py remember fact "The invoice portal moved to portal.example.com"
```

Each one writes a note, runs it through the linter, and indexes it right away, so the next prompt can find it. A new or edited note in `okf/knowledge/` is picked up by a background reindex that the hook starts at most every five minutes. `python3 kit.py reindex` does it by hand.

## Uninstall

```
python3 kit.py uninstall
```

This removes the kit's hook and search-tool entries from every settings file it was installed into, including project-level ones, and leaves everything else in those files alone. It deletes the `.venv` and `.state` folders. It keeps `okf/` (your converted notes and anything you saved with `remember`) unless you add `--remove-corpus`, which deletes only the folders the kit created there, after checking for the kit's marker file. The backups it made (`*.bak-*`) stay where they are, because they can hold your settings. It never changes your original notes.

## Files

| File | What it does |
|---|---|
| `START.md` | The agent's playbook |
| `assess.py`, `thresholds.json` | Stage 1 measurement and verdict |
| `minitest.py` | The optional five-question test |
| `kit.py` | The command line for everything else |
| `config.toml` | Every retrieval setting, commented |
| `rag/` | The code, one module per job |
| `tests/` | Tests: `.venv/bin/python -m pip install pytest`, then `.venv/bin/python -m pytest tests` |
| `fixtures/make_vault.py` | Generates a fictional vault with questions that have known answers |

## License

MIT. See `LICENSE`.

Built by Mason Fitzgerald. There's more on setups like this in my Skool community: https://www.skool.com/ccu
