# Dev notes

Everything below was measured on one Mac (Apple Silicon, macOS, Python 3.13, Ollama with bge-m3, Claude Code 2.1.312, Codex CLI 0.164.0) on 2026-10-27. Linux and Windows were not run.

## Where the design comes from

The corpus format (bundles, types, scope, frozen heading anchors, a linter that gates writes, raw evidence that is never indexed) and the retrieval pipeline shape (keyword and semantic channels, reciprocal rank fusion at k=60, optional cross-encoder reranker, relative cutoff, abstain, passage and token caps, an injected block that names each source) are ported from Mason's production engine, in lighter form: SQLite and numpy instead of Postgres and pgvector, no graph hop, no query-envelope signals beyond a title match. Nothing was copied from the earlier lead-magnet kits.

Two things the brief expected did not exist: the PILOT.md with isolation flags, and a generated benchmark vault. I verified the flags myself and wrote a fixture generator (`fixtures/make_vault.py`).

## Decisions the brief did not specify

| Decision | Why |
|---|---|
| Vectors live in a numpy file next to SQLite, not in sqlite-vec | See "Vector store" below: numpy was 4 times faster at 30,000 and 75,000 vectors, and sqlite-vec cannot load at all in macOS's own `/usr/bin/python3` (`enable_load_extension` is missing). That fails the "installs cleanly" test. |
| Python 3.11 or newer | `tomllib` reads `config.toml` with no dependency. `kit.py` exits with a clear message on an older Python. |
| `config.toml` plus an identical `rag/defaults.toml` | The commented file is what people edit; the copy fills in any key they delete. A test asserts they match. |
| Bundles come from the vault's top-level folders; types come from keyword rules on path, title and tags | Rules are deterministic and cheap. The agent reviews the plan and overrides with `migrate set`. A model never rewrites note text. |
| Note text is copied, not summarized. Wikilinks become relative markdown links where the target resolves, plain text where it does not; fragments are dropped | Keeps the corpus a faithful copy and lets the linter's dead-link check apply. Dataview and other plugin syntax is not interpreted. |
| A note with a credential-shaped string is kept out of the corpus and listed | The linter gate would otherwise fail the whole migration over one note. |
| `okf/raw/notes/` holds a full copy of the originals | Evidence for `sources`, hashes for provenance. Costs disk space equal to the vault. |
| Revert deletes only files whose hash still matches what `apply` wrote | A note you edited after migrating is kept and listed. |
| Chunking: by heading, then paragraphs, cap 600 estimated tokens, no overlap | Matches the engine's section model; overlap was dropped to keep the code small. |
| Keyword channel is one OR query (porter stemming, first 32 content words, bm25 with title and heading weighted above body) | The engine runs an AND pass first; bm25 over an OR query ranks a full-phrase match first anyway. |
| Exact-title channel is built but off | It did not change any metric on the fixture (see ablation). |
| Reranker stays off | On the bigger fixture it raises recall from 35% to 55% but injects for 55 to 58% of near-miss prompts (24 to 27% without it), and costs a multi-gigabyte download and about a second per prompt. |
| Pointer tier is built but off | The live test showed equal accuracy and savings in tokens and time only (see the live table). |
| No-reranker abstain rule: a passage is injected only if its cosine is at least `min_cosine` (0.65) OR at least `min_term_overlap` (0.67) of the prompt's content words appear in it, and the prompt has at least `min_content_terms` (2) content words | Rank-only fusion always has a rank 1, so it cannot abstain by itself; an absolute signal is needed. Chosen by measurement (see the table above); the brief's fusion-score floor measured as noise and was dropped. |
| Injected block is also capped at 9,000 characters (`max_chars`) | Measured: Claude Code writes hook output over about 10,000 characters to a file and shows the model a 2,000-character preview. Four passages at 4,000 tokens would hit that. Whole passages are dropped, lowest ranked first. |
| The hook triggers a detached reindex at most every 5 minutes, and a lock file keeps writers apart | No cron or launchd, so it behaves the same on every OS. |
| The reranker runs in a small local server (`kit.py serve`) that the hook starts in the background; until it is up the hook uses the fusion rule | A hook is a new process per prompt and loading the model takes seconds. |
| Hook is a plain-stdout command with `timeout: 10`, and an internal 6-second budget | Both harnesses add plain stdout to context. |
| `kit.py hook` takes `--drag-kit <hash of kit path>` | That marker is how uninstall finds this kit's entries and no others. |
| `eval` and `tune` use the same golden file format as the mini test | One format to learn. |
| Query embeddings use Ollama's `keep_alive: 24h` | Ollama unloads idle models after a few minutes otherwise, and the next prompt pays a cold load. |

## Hook and MCP formats, and how each was checked (2026-10-27)

Claude Code: `hooks.UserPromptSubmit` is a list of groups, each with a `hooks` list of `{type: "command", command, timeout}` (seconds). The event has no matcher. The payload on stdin has `prompt`. Exit 0 with plain stdout adds the text as context; exit 2 blocks the prompt (the kit never exits 2). Output over 10,000 characters is saved to a file with a preview, which I confirmed with a live `claude -p` run: a 15,400-character hook output arrived as a 2,000-character preview plus a file path, and the model could not read the end of it.

Codex: it does have a prompt-submit hook, with the same JSON shape in `~/.codex/hooks.json` (or inline TOML in `config.toml`), the same `prompt` field and plain stdout added as developer context. Hooks need to be reviewed and trusted once (`/hooks`). Docs mention an `additionalContextLimit` of about 2,500 tokens, which the 9,000-character cap also stays under. The hook docs page was readable and the format matches what is already working in a real Codex install; the Codex MCP docs page returned 404, so the MCP format follows `codex mcp add --help` and an existing config (see the MCP section). I did not run a live Codex hook end to end: doing so needed a copy of the real Codex auth file and a trust bypass, and I was not given permission for that. The Codex path is covered by the installer tests and by the shared hook script, which a live Claude Code run did exercise.

Live Claude Code check with a temporary settings file (`--settings`, `--setting-sources project`, no user settings loaded): the hook injected the permit-timeline passage for "When is the Harbor sign permit approval expected?" and the model answered from it and named the file. Cost about $0.001 on Haiku.

## Measured numbers (fixture: a fictional vault from `fixtures/make_vault.py`, seed 7)

### Index time

| Notes | Sections | Embeddings | Index time | Migrate apply |
|---|---|---|---|---|
| 100 | 251 | bge-m3 via Ollama | 16.0 s | under 1 s |
| 1,000 | 2,312 | bge-m3 via Ollama | 116 s (an earlier 179 s was taken while other jobs ran) | 1.9 s |
| 10,000 | 22,822 | fake hash (1024-d), no network | 10.5 s | 15 s |
| 25,000 | 56,748 | fake hash (1024-d), no network | 26.8 s | 37 s |

With Ollama the time is almost all embedding, about 0.08 s per section here. I did not embed 10,000 notes with bge-m3 (it would take roughly 30 minutes), so the larger rows show the cost of everything except the model. State on disk at 25,000 notes: 318 MB SQLite plus 232 MB vectors.

### Hook latency (whole `kit.py hook` process, warm, 25 runs, median)

| Notes | Embedding | p50 | p95 |
|---|---|---|---|
| 100 | real Ollama query embedding | 229 ms | 261 ms |
| 1,000 | real Ollama query embedding | 246 ms | 263 ms |
| 10,000 | fake (no network) | 160 ms | 163 ms |
| 25,000 | fake (no network) | 201 ms | 209 ms |

The 10,000-note row has no network call in it. The 100 and 1,000 rows suggest a real Ollama embedding adds roughly 70 to 80 ms, so about 230 to 240 ms is a reasonable expectation at 10,000 notes, but that is an inference, not a measurement. Most of the time is Python startup plus importing numpy. In-process `retrieve()` took a median of 51 ms (100 notes), 59 ms (1,000 notes, real embeddings), 23 ms (10,000, fake) and 54 ms (25,000, fake). With the reranker on (in process, MPS, fp16) retrieval took a median of about 1.5 s (100 notes) and 0.8 s (1,000 notes).

### Vector store: numpy vs sqlite-vec (`fixtures/bench_vectors.py`, 1024-d, top 30)

| Vectors | numpy memory-mapped | sqlite-vec |
|---|---|---|
| 30,000 (about 10,000 notes) | 1.8 ms | 8.1 ms |
| 75,000 (about 25,000 notes) | 5.5 ms | 19.2 ms |

Both are fast enough. numpy wins on speed and on portability, so that is what the kit uses.

### Stage 1 and stage 2 mini test on the 100-note fixture (Claude Code, Haiku, 5 questions)

| Arm | Cited the right note | Time | Cost reported |
|---|---|---|---|
| Browse the vault (stage 1) | 5/5 | 23.9 s | $0.0048 |
| Keyword top 4 (stage 1) | 4/5 | 12.7 s | $0.0023 |
| Full hybrid injection (stage 2, shipped thresholds) | 3/5 | 11.2 s | not recorded |

The stage 1 verdict for this vault was "skip" (about 19,000 tokens), and browsing got every question right, so this fixture doesn't show retrieval beating browsing. The two hybrid misses are paraphrased questions that fall below the 0.65 cosine floor, so nothing is injected for them and the agent answers "not found". Five questions is too few to rank approaches.

A live Claude Code check with the hook installed in a temporary settings file: "When is the Harbor sign permit approval expected?" got the permit note injected and the model answered from it; "fix this failing test in test_parser.py" got no notes injected.

### Retrieval eval, false injection and ablation

Most prompts someone sends an agent aren't questions about their notes ("fix this failing test", "write a commit message", "thanks"), so the failure viewers hit most is injection where nothing should be injected. The eval scores four kinds of prompt separately. The fixture was grown after the first tuning pass, because 25 answerable questions (one question = 4 points) and 10 near-miss prompts were too few to choose between options. The earlier results in git history used that smaller set and were too optimistic.

| Set | Size | What it is |
|---|---|---|
| Answerable | 106 | 51 paraphrase (46 pass a strict "no shared distinctive words" check; the other 5 are original, frozen questions), 22 direct, 16 entity, 11 multi-note, 6 changed-fact. Scored as recall@4: a passage from the right note was injected. |
| No-answer | 5 | Questions that sound like the notes but aren't answered by them. |
| Off-topic | 50 | `rag/offtopic.json`, ships with the kit: coding, shell and git, error messages, general knowledge, chat. |
| Near-miss | 33 | `fixtures/nearmiss.json`: advice, drafting, coding and not-in-the-vault prompts that reuse the fixture's vocabulary. The hardest set. |

False injection (FI) is the share of prompts in a set that still got any passage injected. At 100 notes the fixture is 70 plant notes and 30 filler; at 1,000 it is the same 70 plus 930 filler.

**Baseline: the shipped defaults** (bge-m3, `min_cosine 0.65`, `min_term_overlap 0.67`, `min_content_terms 2`, no reranker, no pointers). The columns after recall are FI on no-answer / off-topic / near-miss:

| Variant | 100 notes | 1,000 notes |
|---|---|---|
| keyword only | 26%; 0 / 0 / 12 | 26%; 0 / 2 / 12 |
| semantic only | 35%; 0 / 0 / 24 | 34%; 0 / 2 / 27 |
| hybrid, no thresholds (always injects) | 73%; 100 / 100 / 100 | 80%; 100 / 100 / 100 |
| **hybrid, shipped thresholds** | **35%; 0 / 0 / 24** | **35%; 0 / 2 / 27** |
| hybrid + exact-title channel | 35%; 0 / 0 / 24 | 35%; 0 / 2 / 27 |
| reference: BAAI/bge-reranker-v2-m3 (opt-in, `rel_cutoff 0.35`, `abstain_floor 0.05`) | 55%; 0 / 0 / 55 | 55%; 0 / 2 / 58 |

Recall by question kind for the shipped defaults (identical at both sizes): direct 19 of 22, entity 13 of 16, multi-note 2 of 11, changed-fact 2 of 6, paraphrase 1 of 51. In other words the gate passes questions that share words with the note and almost never passes a true paraphrase.

Leaks (a deprecated, stale or out-of-scope passage injected) were 0 in every row. The right note is in the top 10 candidates for 92% (100 notes) and 89% (1,000 notes) of answerable questions, so retrieval finds the notes and the injection gate is what withholds them.

The big reranker is the only way I found to raise recall above 35% without injecting for most no-answer prompts, but on the larger near-miss set it injects for 55 to 58% of near-miss prompts, against 24 to 27% for the cheap gate. The 0% near-miss figure I reported for it earlier came from a 10-prompt set that was too easy. It takes about 0.9 to 1.0 s per prompt (20 candidates, in-process, MPS). It stays opt-in.

**The curve** (`python kit.py eval tune`; overlap 0.67, minimum two words; recall and FI off-topic / near-miss; FI no-answer was 0% from 0.60 up):

| min_cosine | 100 notes | 1,000 notes |
|---|---|---|
| 0.50 | 80%; 22 / 91 | 77%; 42 / 97 |
| 0.55 | 61%; 6 / 70 | 61%; 8 / 82 |
| 0.60 | 50%; 2 / 45 | 50%; 4 / 52 |
| **0.65 (ships)** | **35%; 0 / 24** | **35%; 2 / 27** |
| 0.70 | 30%; 0 / 15 | 30%; 2 / 15 |

No cosine floor gets near-miss false injection to the 5% I was aiming for, even at 0.70, where recall is 30%. Paraphrased questions score 0.50 to 0.65 against their notes and near-miss prompts score 0.42 to 0.71 against theirs, so an embedding score alone can't separate "close in meaning" from "answered by this note". The shipped 0.65 keeps off-topic and no-answer false injection at 0 to 2% and near-miss at about a quarter, at the cost of 35% recall.

**Signals tried** (over stored candidates; see the earlier pass for the smaller-set detail): the cosine floor is the main lever; raising the word-overlap bar from 0.5 to 0.67 and requiring two content words removed most off-topic injection; a fusion-score floor, a within-0.1-of-best-cosine rule, a 3-word minimum for the overlap route, a both-channels requirement and a margin gate added nothing or made things worse, and were dropped or not added.

**Small ONNX rerankers, dropped.** Option: a cross-encoder through `fastembed` and onnxruntime, no torch (184 MB of packages). Measured median time per prompt for 20 candidates at 100 notes: ms-marco-MiniLM-L-6 (80 MB, Apache-2.0) 443 ms, jina-reranker-v1-tiny-en (130 MB) 367 ms, jina-reranker-v1-turbo-en (150 MB) 538 ms, ms-marco-MiniLM-L-12 (120 MB) 855 ms. All are English-only and all are over the 300 ms budget, and the multilingual ones are either 570 MB (bge-reranker-v2-m3 int8, 2.4 s) or non-commercial (jina v2). I did not measure their recall once the latency and language limits ruled them out.

**Pointer tier** (`pointers_enabled`, off by default). Passages are injected only at a strict gate (`min_cosine 0.70`, `min_term_overlap 0.8`), and up to three more notes with cosine at least `pointer_min_cosine` (0.50) that were not injected appear as a short "possibly relevant notes" list: path and title, no text, about 500 characters (roughly 130 tokens) with the header, and the header tells the model they are candidates to open only if useful. Offline, pointer recall (right note in the passages or the pointers) was 76 to 80% at `pointer_min_cosine` 0.50, with roughly 22% (100 notes) to 42% (1,000 notes) of off-topic prompts getting a pointer line and nearly all near-miss prompts. Whether that matters depends on what the model does with the lines, so I tested it live.

**Correction to the live test below (found afterwards from the stored tool calls).** The "shipped defaults" and "pointers" arms ran from an empty folder inside a scratch directory that also held copies of the fixture vaults, other copies of the kit, the answer key (`sets.json`) and the fixture generator. In the paraphrase runs, 52 of the shipped-defaults arm's 98 tool calls targeted paths outside both the corpus and the vault (it often grepped the whole scratch directory), and 5 more went to unrelated fixture folders; only 6 touched `okf/raw/notes` and 35 `okf/knowledge`. The browse arm ran inside the vault folder, so it had none of that clutter. That is the real cause of the 21.2 s and 58.5k tokens against 8.3 s and 28.6k: the arms weren't comparable, and the clutter also let the agent find the answer key, so the correctness figures for those two arms may be inflated. Duplicate raw copies were a minor contributor (6 calls), and they are fixed separately (see "Search tools skip the raw copies"). Treat the table as a record of what ran, not as evidence about speed, tokens or accuracy. The rerun needs a clean working folder (an empty directory with nothing else under it) and waits for the go-ahead to use Ollama and Claude again.

**Live test: what the model does** (Claude Code on Haiku, real hook in a temporary settings file, `--tools Read,Grep,Glob`, 1,000-note fixture, 130 headless runs):

| Set | Arm | Correct | Pointer lines shown | Opened a pointed note | Tool calls | Time | Tokens (all kinds) |
|---|---|---|---|---|---|---|---|
| 20 paraphrased questions | no hook, browse the vault | 17/20 | - | - | 3.3 | 8.3 s | 28.6k |
| | shipped defaults (passages only) | 17/20 | - | - | 4.9 | 21.2 s | 58.5k |
| | strict passages + pointers | 16/20 | 18/20 | 12/18 | 3.5 | 13.6 s | 35.5k |
| 15 off-topic prompts that get pointers | shipped defaults | - | - | - | 1.7 | 6.1 s | 17.4k |
| | strict passages + pointers | - | 15/15 | 0/15 | 1.3 | 5.8 s | 16.4k |
| 15 near-miss prompts (highest similarity) | shipped defaults | - | - | - | 1.9 | 10.2 s | 25.2k |
| | strict passages + pointers | - | 15/15 | 8/15 | 1.3 | 8.4 s | 17.3k |

How it was graded. Paraphrase answers were checked mechanically against the key; an exact-substring check scored all three arms 11/20, but the misses were mostly wording ("November 14th" against "14th of November"), so I applied a looser token check and read every answer the two checks disagreed on (14 answers): all were correct, which gives 17, 17 and 16. Near-miss answers were flagged mechanically if they contained a number or code that occurs in the vault and not in the prompt (shipped defaults 7 of 15, pointers 5 of 15), and explicit "couldn't find" answers were counted (4 and 5). I then read all 30 near-miss answers by hand. None presented a note's fact as the answer to something the notes don't state. The flagged answers used a real note as context for a drafting or coding request (the plumber's booking in a plumber email, the staging port in a fetch example, an existing permit number) or declined while naming the closest note. Whether using a real note that way is wanted is a judgment call; I counted it as correct in both arms. The off-topic arms had their prompts picked so that pointers would appear (21 of the 50 got at least one), so the 15/15 isn't a rate; it is the opportunity for wasted opens.

What it shows:

- Accuracy did not improve in this (confounded) run; the clean rerun below replaces it. The agent answers most paraphrased questions by browsing whether or not the kit injects anything: at 1,000 notes (about 170,000 tokens, under the "skip" threshold) browsing alone got 17/20. With the shipped defaults only 1 of the 20 got a passage, so that arm is close to unaided browsing, and it needed more tool calls and tokens than browsing from the vault folder because it started with no hint where the notes were.
- Pointers appeared to cut work, not errors, but see the correction above: against the passages-only arm they took about 40% fewer tokens and 36% less time on paraphrases, and in 12 of 18 cases the model opened the note it was pointed to. Against browsing from inside the vault they were slower and used more tokens.
- No wasted opens on off-topic prompts (0 of 15) and no extra tokens; a pointer line costs roughly 130 tokens.
- No rise in wrong-note answers on near-miss prompts (by hand, none in either arm). The model opened a pointed note in 8 of 15, and then mostly used it appropriately or said the note didn't answer.

Decision: pointers stay off by default until a clean rerun. The condition was better correct answers on paraphrases, and this run shows the same accuracy (16 against 17 of 20, inside the noise of a 20-question sample), with apparent savings in tokens and time that the confound makes unreliable. The flag is there for anyone whose agent doesn't browse well or whose vault is much bigger than the 1,000 notes I could test live; I did not run the live test at 10,000 notes.

### Clean live comparison of the modes (1,000 notes, Sonnet)

The earlier live test was confounded (see the correction above), and a separate benchmark found that agents handed injected passages can anchor on them and stop searching. So I reran it with a clean setup. Everything below used Claude Code 2.1.312 with `--model sonnet`, tools limited to Read, Grep and Glob, the kit's real hook and its real MCP server (`search_notes`) through temporary `--settings` and `--mcp-config` files, and `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`. The run root held only the kit's code, the 1,000-note corpus (`okf/knowledge`) and its index: no fixtures, answer keys, tests, raw copies or other kits. The agent's working folder was `okf/knowledge` in every arm, so all five arms browsed the same files. Every run's tool paths were audited and any run that touched a path outside that folder was excluded: 14 of 360 (7 browse, 4 inject, 3 inject_verify), 13 of them reading Claude Code's own auto-memory folder for the working directory and one listing the home directory. None touched an answer key. Machine load during the runs averaged 5.8 because an unrelated benchmark was ingesting, so treat times as slow; accuracy and tokens are the main figures.

Arms: **browse** (no kit), **inject** (the older "may or may not be relevant" header), **inject_verify** (the new header: automatic top matches, may be incomplete, search the notes yourself if they don't clearly answer), **tool** (`search_notes` only, nothing injected) and **both** (verify injection plus the tool). 30 questions from the 1,000-note fixture (12 paraphrase, 6 direct, 5 entity, 4 multi-note, 3 changed-fact), 2 runs each, plus 12 off-topic coding and shell prompts per arm. Accuracy is the looser token check used before, which I checked by hand on the earlier set; times are for the whole `claude -p` call.

| Arm | Runs used | Correct | Mean tokens | Median tokens | Mean time | Tool calls | Cost per run |
|---|---|---|---|---|---|---|---|
| browse | 53 | 92% | 21.7k | 18.0k | 6.7 s | 2.4 | $0.0155 |
| inject (old header) | 56 | 93% | 16.2k | 11.9k | 6.0 s | 1.6 | $0.0102 |
| **inject_verify** | 57 | **95%** | 16.0k | 12.4k | 5.8 s | 1.6 | $0.0098 |
| tool | 60 | 95% | 14.7k | 13.4k | 5.3 s | 1.7 | $0.0104 |
| both | 60 | 95% | 13.2k | 13.3k | 5.3 s | 1.4 | $0.0068 |

Including the excluded runs, accuracy was 93, 93, 95, 95 and 95%. Percent correct by question kind (clean runs): direct and entity 100% in every arm; changed-fact 100% in every arm; multi-note 75% (browse, inject, inject_verify) and 100% (tool, both); paraphrase 92, 92, 96, 88 and 88%.

Off-topic prompts (12 per arm): no passages were injected for any of them, in any arm. Any notes search at all happened in 2 or 3 of the 12 prompts in every arm, including browse, and was the model looking for a file named in the prompt (for example `test_parser.py`); the tool arms made 0 `search_notes` calls on off-topic prompts. Mean tokens were 7.2k (browse), 6.6k (both inject arms) and 7.4k and 6.9k (tool, both), within noise of each other.

Anchoring, the failure the other benchmark found: passages were injected for 14 of about 57 answerable runs per injecting arm (25%, mostly direct and entity questions). In 2 of those 14 the injected passages did not contain the answer; in both the agent kept searching and answered correctly. The count of runs that answered from passages lacking the answer without searching further was 0 in every arm. The tool arm called `search_notes` in 60 of 60 answerable runs.

What this shows, and doesn't:

- **Accuracy:** no arm is distinguishable from browsing. The spread (92 to 95%) is smaller than the noise: only three of the 30 questions were missed by any arm (g023 by browse, inject and inject_verify; g035 by tool and both; g057 by most), so a handful of runs decides the ranking. Neither the new header nor the tool measurably changed accuracy here, and the header did not make injection worse than browsing.
- **Cost:** every kit mode used fewer tokens than browsing (means 25 to 39% lower, medians 25 to 34% lower) and was a bit faster, and `both` was cheapest per run in dollars. This matches the other benchmark's token finding.
- **Why it differs from the other benchmark:** that one found browsing most accurate at every size and injection 15 to 25 points worse. This fixture is smaller (1,000 notes), has plain factual questions, and my gate injects for only a quarter of prompts, so most questions were answered by the agent's own searching in every arm. The verify header may matter more when passages are injected more often or the vault is larger. This run can't tell.
- **Limits:** one vault, one model, 30 questions by 2 runs, a loaded machine, and a model that was not blind to the arm. One run is 1.7 points.

Decision: inject with the verify header stays the default (it matched browsing's accuracy with about a quarter fewer tokens, and it is the deterministic option), tool and both stay opt-in (`install --mode tool|both`). `both` had the lowest cost per run, but the gap to inject is within noise of a single-vault test, and it needs a long-running MCP server in addition to the hook.

### Live MCP start

`search_notes` ran in Claude Code through a temporary `--mcp-config` file with `--strict-mcp-config`: the init event listed the server `notes` as connected and the tool as `mcp__notes__search_notes`. A prompt asking when the Harbor sign permit approval was expected produced one `search_notes` call, whose result listed the permit note first with `match: strong`, and a correct one-sentence answer. In the comparison above the tool arm called it in 60 of 60 runs with no errors. Codex has not started the server.

### Batching speedup (real bge-m3, 1,000 notes, 2,208 sections)

Four fresh indexes of the same corpus, alternating old and new settings (load average 3.3 to 6.3): batch 16 with one request at a time took 116.4 s and 115.4 s; batch 32 with two in flight took 112.7 s and 113.6 s. That is about 2% faster, which is within run-to-run noise. Ollama serves embeddings one after another on this machine, so the time is the model, not the request overhead. The new defaults are harmless and tested, but they don't speed up Ollama; they may help with a hosted embedding API, which I did not measure. (The 179 s I recorded earlier was taken while other jobs were running.)

### Stage 1 isolation (verified)

Claude Code (prompt on stdin), used by the mini test:

    claude -p --safe-mode --strict-mcp-config --disable-slash-commands --permission-mode dontAsk \
      --no-session-persistence --output-format json [--model M] --tools "Read,Grep,Glob"   # browse arm, cwd = vault
      ... same with --tools ""                                                              # keyword and hybrid arms, cwd = empty temp dir

A canary project hook and a CLAUDE.md in a scratch folder: with no flags the hook fired and the init event listed 24 MCP servers, 211 skills and 370 tools; with these flags the hook did not fire and the init event listed 0 MCP servers, 0 skills, 0 slash commands and only Glob, Grep and Read. The prompt dropped from about 49k to about 5.5k tokens. `--bare` also isolates but requires an API key and ignores OAuth login, so subscription users can't use it. The init event still lists installed plugins as metadata, but nothing from them loaded. Not verified: that `Read` can't reach files outside the vault.

Codex (`codex exec`), with `CODEX_HOME` pointing at a temporary folder holding only a copy of the login file (deleted afterward):

    codex exec --json --ephemeral --skip-git-repo-check --sandbox read-only --ignore-user-config \
      --ignore-rules -c project_doc_max_bytes=0 -C <dir> [-m M] -

`--ignore-user-config` alone still let a global AGENTS.md through, which is why the temporary home is used. A positive control confirmed a project AGENTS.md was read with project docs on and ignored with `project_doc_max_bytes=0`. Not verified: hooks or MCP servers inside a real Codex config, since the temporary home has none to watch fire. Codex has no switch to disable all tools, so the keyword arm runs read-only in an empty folder and the tool-call count is recorded. The mini test warns when isolation is incomplete.

A 12-note vault run through both harnesses worked: Claude (Haiku) 4 runs in 13.4 s, Codex (default model) 2 runs in 13.5 s.

### Search tools skip the raw copies

The corpus folder holds `raw/notes/` (a full copy of your notes), `knowledge/` (the converted notes), and bookkeeping (`staging/`, `eval/`, `reports/`, `migration.json`). An agent searching `okf/` with Grep or Glob would hit each note twice and could read the eval answer key. The kit now writes `okf/.ignore` listing those folders. ripgrep, which Claude Code's Grep and Glob use, honors it: on the 100-note fixture `rg --files okf` went from 224 files across five folders to 113 under `knowledge/` plus two bookkeeping files, and a search for "Harbor" hit only `knowledge/`. Searching `okf/knowledge` directly is unaffected. I checked this with ripgrep 14.1.1 only; a Grep tool that doesn't use ripgrep wouldn't honor it. A test checks the file's contents, and the ripgrep behavior when `rg` is installed.

### Search-tool mode (MCP server)

`kit.py mcp-serve` is a stdio MCP server (newline-delimited JSON-RPC, standard library only) with one tool, `search_notes(query, limit)`. It returns up to 10 sections by fused rank, each with the note's path, section, a strong or weak label from the same relevance check the hook uses, and the text, capped at 9,000 characters; deprecated and stale notes are excluded like everywhere else. `kit.py install --mode inject|tool|both` writes the config, default `inject`. Config formats:

- Claude Code (live docs, code.claude.com/docs/en/mcp): `{"mcpServers": {"<name>": {"command": ..., "args": [...]}}}` in `.mcp.json` (project) or the top-level `mcpServers` of `~/.claude.json` (user). The kit merges into that key only, backs the file up first, and finds its own entry by a marker in the args.
- Codex: `[mcp_servers.<name>]` with `command` and `args` in `~/.codex/config.toml`. The Codex docs page I tried returned 404, so I checked the format against `codex mcp add --help` and the existing `[mcp_servers.*]` tables in a working config. The kit appends one marked comment block, checks the result parses, and removes exactly that block on uninstall.

Tests use temporary files only: merge keeps other servers and keys, install twice replaces, a name clash gets a suffixed name, uninstall leaves other kits' entries, invalid JSON or TOML is never touched, Windows-style paths quote correctly, and the protocol (handshake, tool list, call, errors, bad lines) works over real pipes. Not run live: neither Claude Code nor Codex has started this server yet.

### Embedding throughput

Sections are now sent to the embedding model in batches (`batch_size`, default 32) with `concurrency` (default 2) requests in flight. Only the main thread writes to SQLite. The first failed batch stops new ones and keeps finished work, and the next reindex picks up the rest. Unit tests use the fake embedder to check batch sizes, the concurrency limit, that each vector lands on its own section whatever order batches finish in, and failure recovery. The measured speedup is about 2%; see "Batching speedup" above.

### Release checklist results

A scan of every tracked and untracked file and the full git history for key and token shapes found only the three fake credentials in the tests that exercise the linter's secret check; they are now assembled from pieces at run time. Commits made before that still contain them in history, so squash or rewrite history before the first push or GitHub's secret scanning may flag them. No personal paths or emails are in the tree (the git history is a separate matter and is handled at release time). `.gitignore` covers `.venv/`, `.state/` (index and vectors), `okf/` (the user's corpus), `.env`, `report.html` (it embeds the user's notes), caches and `work/`.

## Tests

`python -m pytest tests`: 203 tests pass and 1 is skipped (ripgrep not on PATH), about 22 seconds, no network and no model. They cover the chunker, FTS search, fusion math, cutoff and abstain, deprecated/stale/scope filters, hook fail-open (bad JSON, corrupt database, crashing and slow embedders, a CLI run on garbage input), the settings merge and uninstall leaving foreign hooks intact, incremental reindex, migration (vault hash unchanged, revert, gate failure writing nothing), `remember`, the stage 1 scripts, the fixture generator, and both stages end to end on a generated 100-note vault. The installer and hook tests use temporary settings files and a temporary HOME; nothing in `~/.claude` or `~/.codex` was touched.

## What didn't work or isn't done

- A live Codex hook run (see above).
- A 10,000-note run with real embeddings, and any Linux or Windows run.
- The stage 1 verdict is now based on token size only (under about 200,000 skip, to about 5 million optional, above that recommended if cost or speed matter). The 5 million line is a judgment between two benchmark sizes (746k and 7.6M tokens), not a measured crossover; the benchmark found no accuracy crossover up to about 38M tokens. The benchmark's results aren't public, so the verdict describes them as Mason's benchmark for the video, with no link. The numbers live in one place (`BENCHMARK` in `assess.py`) plus one README paragraph.
- Time-based staleness is checked at query time, but a stale note stays in the index until someone deprecates or deletes it.
- Two fixture questions about a changed fact ("updated") only test that the newer note is retrievable; the kit has no recency weighting, so an old note with a stronger match can still win. Deprecating the old note with a `## Superseded by` link is the supported fix.


## Release review fixes

A second reviewer went through the kit before release. What changed, with the tests that lock it in (`tests/test_release_fixes.py` unless noted):

- **Data safety.** The corpus folder can't be the notes folder, inside it, a parent of it, the home folder, a shallow path like `/tmp`, or a folder that already holds someone else's files. The kit writes a `.kit-corpus` marker, and `uninstall --remove-corpus` refuses without it, refuses if the recorded vault overlaps the corpus, checks everything before deleting anything, and then removes only the folders and files the kit created. An early version of one of these tests wrote into the real `/tmp` before the shallow-path rule existed; I removed the five stray items and the rule now blocks it.
- **Uninstall coverage.** Every file installed into is recorded in `.kit-installs.json`, so a bare `uninstall` cleans project-level settings too. The marker is a stable id in `.kit-id`, so it survives moving the folder. Backups are left in place and the command says so.
- **Install.** Without `--harness` the installer uses whichever of `~/.claude` and `~/.codex` exists and never creates the other, and it refuses to install before `setup` has made the venv. A symlinked settings file is written through, not replaced. JSON is written without escaping non-ASCII text.
- **The hook.** The installed command uses the interpreter that ran `install`, and `kit.py` hands over to `.venv` itself; if the venv is gone the hook exits 0 without printing anything. The hook reads and writes UTF-8 bytes (Windows consoles), and exits with `os._exit(0)` so a timed-out search thread can't hold it open.
- **Indexing.** The lock records the owner's process id and is reclaimed at once if that process is gone; it is touched as work proceeds and a taken-over lock is never deleted by the old owner. The first index prints `embedded n/total sections`. A note moved to another path keeps its row and its embedding, and a section whose text matches one already indexed reuses that vector.
- **Search.** If the embedding model or vector size no longer matches the index, search falls back to keywords with a warning to reindex. Closing tags in a note are escaped case-insensitively, attributes are quoted, and the block header says the text is data.
- **Small ones.** `remember ""` is rejected, an empty vault stops with "check the path", plan fields that become file paths are validated, the reranker server accepts only small JSON bodies, the mini test doesn't hand `OPENAI_API_KEY` to the model, `setup` explains a venv failure, and dependencies have lower bounds (`pyyaml>=6`, `numpy>=1.26`).

Real-embedding sanity run after these fixes (100-note fixture copy in a clean scratch root, temporary HOME, bge-m3 via Ollama): setup, `migrate plan` and `apply`, a reindex with progress output (215 sections, 11 s), the hook installed with `install --mode both` and run through a shell exactly as an agent would, the MCP server answering a `search_notes` call, and a bare `uninstall` that cleaned both files. Only `~/.claude` existed in that home, and no `~/.codex` was created. The hook's warm median was 263 ms (12 runs, loaded machine), about 30 ms more than before because the installed command now starts the system Python and hands over to `.venv`.

Not changed: the verdict outcome names and the benchmark wording in `assess.py` (waiting on final benchmark numbers), the `scopes` default (documented instead), and `~/.claude.json` being rewritten while Claude Code may be running (the write is short and atomic, but a concurrent write by Claude Code could still lose one side).

## Automated security review fixes

- **Injected block.** Every value from a note that goes into an attribute or a pointer line (path, section, original path, title) now passes through one escaping rule (`_attr`: escape `&`, `"`, `<`, `>` and flatten newlines), and passage text has the block's own tags (`note`, `retrieved_notes`, `possibly_relevant_notes`, `retrieved_context`, opening or closing, any case) defanged without altering anything else. A test uses a note whose title and heading contain `"></retrieved_context>` and a newline and checks the block's line structure survives.
- **Secret check.** The linter and the migration gate now scan the whole rendered document (title, description, tags, aliases and body), not just the body. Tests cover a credential in a title, in tags and in aliases.
- **Local reranker server (opt-in).** It binds 127.0.0.1 only, requires a random token that is written to `.state/rerank.token` (mode 0600, replaced on each start) and sent by the kit's client in `X-Rerank-Token`, refuses any request with an `Origin` header or a non-loopback `Host`, requires the media type to be exactly `application/json` (a parameter such as `charset` is allowed, a substring match is not), and caps bodies at 1 MB. Tests run the real server with a fake scorer and try each refusal.
- **Reranker handshake (threat model).** The local reranker server is opt-in and off by default. The only attackers considered are other processes on the same machine: one that squats the port while the server is down, one that relays between ports, and a web page posting to localhost. The token never crosses the wire.
- **How it works.** The client sends a nonce and `HMAC(token, "client|nonce|127.0.0.1:dialed_port")` to `/hello`; the server refuses unless that matches its own port (so it never signs for anyone without the token, and a relay's proof is bound to the wrong port), then answers `HMAC(token, "server|nonce|127.0.0.1:server_port")`, which the client checks against the port it dialed. Only then does it send `/score`, authenticated per request by `HMAC(token, "score|nonce|time|port|sha256(body)")`, with a fresh nonce, a 30-second timestamp window and a server-side set of seen nonces against replay. Any failure means no reranking, with one warning when something answered wrongly; the server records its pid and port so `doctor` can report a mismatch.
- **Tests.** A squatter without the token, a relay forwarding to the real server on another port, a server proof made for another port, a client proof made for another port, replays, stale and future timestamps, tampered bodies, and the token never appearing in anything sent.
