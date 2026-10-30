# Board and wiki house style

For issues, comments, PRs, wiki pages, and ADRs in `RedHatResearch/aibom-security`. Board readers: intern and mentor. Wiki readers: external AI-security researchers. Both are public: translate private context into neutral technical language. (AGENTS.md) Every rule cites a source or is marked `[house]`; sources are listed at the end.

## 0. What renders where

| Affordance | Issues / PRs / comments | Wiki |
|---|---|---|
| Tables, fences, math, Mermaid | yes | yes (GitHub: diagrams) |
| `<details>` | yes | use headings; readers came for depth `[house]` |
| Task list | clickable; `#N` unfurls title and state | progress roll-up is issue-only; use a table `[house]` |
| Footnotes | yes | no (GitHub docs) |
| Alerts `> [!NOTE]` | yes | no documented support (github/docs#27919) |
| `#123`, `@user`, SHA autolinks | yes | no (GitHub: autolinks) |
| File and page links | full `https://github.com/.../blob/main/…` | `[x](Page-Name)`; repo links are full URLs |

Capability claims: GitHub Docs (basic syntax, math, autolinks, diagrams, tasklists); wiki exceptions per GitHub Docs and the linked issues.

## 1. Sentence rules

- One idea per sentence, split at a second "and which" or "because". (Google TW: Short sentences)
- Active voice, actor first; passive only when the actor is unknown; cut nominalizations ("causes the triggering of" → "triggers"). (Google TW; STE verbs)
- Cap length: 20 words for a step, 25 for description; one instruction per sentence, verb first; three or more parallel items become a list (numbered when order matters), parallel in grammar and punctuation. (STE 2.1, procedures; Google TW: Lists and tables)
- Paragraphs: 3–5 sentences, one topic; the first sentence carries the point. (Google TW: Paragraphs)
- Same word for the same thing; define or link a term at first use; keep noun clusters ≤3 nouns ("centered-residual LayerNorm cosine"). (Google TW: Words; STE noun clusters)
- Not adopted: STE's 900-word dictionary and one-part-of-speech rule. Domain terms need precision, and a second dictionary is a second spec to maintain. `[house]`

## 2. Issue bodies

### Hub issue: one page, stable
- Order: what/why (≤3 sentences) → the decision it feeds → sub-issues → scope → status → constraints. No body H1; the issue title is the heading. Budget ≤60 lines; link the wiki catalog, do not paste it; one Mermaid only for a real branch in the logic; logs go in `<details>` or a comment. (Google TW: Documents; AGENTS.md; `[house]`)
- Opening answers the issue-form prompt with the decision: ✅ "Decide which fingerprint survives quantization, so M2 ships one detector." ❌ "This issue is about fingerprints." (research issue form; Google TW: opening sentence)
- Native sub-issues, listed as `- [ ] #101` so GitHub unfurls title and state; check a box only when the child closes. (GitHub: sub-issues; AGENTS.md)
- Scope is a two-column `In | Out` table, one clause per cell. (Google TW: tables)
- Status is a rolling index, newest first: `- 2026-10-30: ≤20 words · [comment](url)`. It links the comment; it does not repeat it. `[house]`
- One-liners keep the schema `- [ ] Name (link) — signal ≤8 words · **verdict** · [findings](url) · path`. ✅ `Witness Overlap ([arXiv](…)) — update-delta overlap · **adopt** · [findings](…) · smokes/witness-overlap/` ❌ `Witness Overlap — promising, adopted for now, see comments`. (issue #26)

### Research sub-issue: question, test, done
- Title is the question or deliverable, ≤72 chars: ✅ "Do shallow, three-parent merges still get caught?" ❌ "Merge robustness". (issue #67; CL guide: first line stands alone)
- Paragraph 1 (2–3 sentences): the finding that makes this worth doing, with its number; then `What we want to check:` 3–6 testable variants, not topics. (Google TW: opening sentence; issue #67)
- `Done when:` one sentence naming the artifact that proves it: ✅ "Recover merge weight within ±0.05 on 5 shapes; the table says where it breaks." ❌ "Investigate robustness." End with `Refs #N`; no Status section; budget ≤30 lines. (KEP: Goals; AGENTS.md; `[house]`)

## 3. Issue comments

- Prefix first, then a lowercase clause: `Decision: centered cosine replaces raw cosine for LayerNorm pairs.` Five prefixes only; no chat scaffolding or agent ritual. A correction opens with the words `Correction —`, says what is withdrawn, and links the fix. (AGENTS.md)
- Bookkeeping comment: one paragraph ≤120 words; bullets only for ≥3 parallel items. (Google TW: Paragraphs)
- `Decision:` states what changed, what it rules out, and the next action; `Done:` names the artifact and link, not a thread summary. One decision per comment. (AGENTS.md: Closeout; `[house]`)
- Deep dive: `## Name — verdict`, metadata lines, `### Context`, `### Results` (table), `### Limits`; first table within one screen. (issue #26; `[house]`)
- ✅ `Decision: benchmark Witness Overlap on the 40-pair bench; adopt stays blocked on a genuine E=64 negative.` ❌ `Decision: After extensive analysis, we feel this is a really strong candidate going forward.`

## 4. PR description

- Title: `<area>: <imperative summary>`, ≤72 chars, no period: ✅ `smokes: calibrate routing overlap as a lineage signal` ❌ `Update files and fix issues`. (repo PRs; CL guide; cbea.ms)
- First body line: `Refs #N`, or `Fixes #N` only when the issue is fully done; set the PR milestone too. (AGENTS.md; PR template)
- Keep the template's three H2s (`What & why`, `Test plan`, `Out of scope / follow-ups`); under What & why, one sentence what, one sentence why, then the evidence table. (PR template; CL guide)
- Test plan: command plus observed result (`exits 0, 13/13 expectations held`), not "tests pass". (PR #77; `[house]`)
- Out of scope: the known gap and its issue number; state the weak negative. Budget ≤40 lines; raw logs in `<details>`. (CL guide: shortcomings; GitHub: collapsed sections; `[house]`)

## 5. Wiki pages

Wiki links are `[text](Page-Name)`. `#N` does not autolink there, so link the full issue URL. (GitHub: autolinks)

### Home: index, ≤30 lines
- One sentence on what the wiki holds, then a `Page` / `Question` table: ✅ `| [ProvenanceBench](ProvenanceBench) | How are pairs scored, and what counts as a pass? |` ❌ `| ProvenanceBench | WIP — see #36 |`. No status, dates, or owners; those live on the issue; order by reading path, and mirror the sections in `_Sidebar.md`. (Google TW: opening; AGENTS.md: board vs wiki; repo wiki; `[house]`)

### SOTA catalog snapshot: generated
- Header ≤3 lines: source, commit, date, row count, then `Generated; do not edit — open an issue.` ✅ `Source: survey data @ 4f2a1c9 · 2026-10-30 · 63 methods.` ❌ "A comprehensive, carefully curated overview of the evolving landscape." (Wikipedia: Promotional language; `[house]`)
- Fixed columns, one row per entry (method, engine, tool), stated sort, fixed verdicts (`adopt`, `benchmark`, `cite`, `reject`, `field case`); every row links its findings comment or paper; no prose in cells; no Mermaid. (Google TW: tables; issue #26; AGENTS.md)

### Authored writeup: 800–1500 words
- Title is the question or claim: ✅ "Does a universal weight interpreter exist?" ❌ "Notes on universal weight interpretation". (CL guide: first line)
- First 3 sentences: question, answer, why it matters. Then Context, Method, Results table, Limits, Reproduce, References. (Google TW; KEP: Summary/Motivation)
- One Mermaid maximum, only for a branch or flow; link the source issue and catalog row; date the evidence. No "Introduction" heading and no "Despite the challenges" section. (AGENTS.md; Wikipedia: Title heading, Outline-like conclusions)

## 6. ADR (MADR-lite)

- File `docs/decisions/NNNN-kebab-title.md`; numbers are never reused; H1 states the decision: ✅ `# ADR-0007: Compare LayerNorm pairs with centered cosine` ❌ `# ADR-0007: Fingerprinting`. (MADR; `[house]`)
- Header `Status: accepted · Date: 2026-10-30 · Supersedes: —`; `## Context and Problem Statement` is 2–3 sentences plus the issue link. (MADR metadata)
- `## Decision Outcome`: one sentence, active, "We …". (STE active voice; MADR)
- `## Consequences`: `Good, because …` / `Bad, because …`, each with an observable effect. (MADR)
- `## Considered Options`: 2–4, one line each plus why rejected. Budget ≤60 lines; supersede, never rewrite. (MADR; `[house]`)

## 7. Cross-cutting

### Links
- Link text names the target (`[HuRef repo](…)`, not `[here](…)`); one primary link per claim (a DOI or an arXiv id, not both plus a blog); link the number, so `0.858` gets the comment that computed it; in issues prefer `#N`, in wiki the full URL. (Google TW: Words; CL guide: durable links; GitHub: autolinks; `[house]`)

### Tables
- Use when ≥2 rows × ≥2 attributes, cells short (one clause where the column allows); for two items write a sentence. Introduce lists and tables with a colon; name the unit and direction; right-align numbers (`---:`); no cell over two sentences. (Google TW: tables, Lists and tables)

### Mermaid

| What you are showing | Type |
|---|---|
| Control flow, gates, pipelines, async handoff | `flowchart` / `graph` |
| Message or call order between actors | `sequenceDiagram` |
| States and transitions (job, run, test lifecycle) | `stateDiagram-v2` |
| Types, interfaces, plugin seams | `classDiagram` |
| Schema / store / metadata relationships | `erDiagram` |
| Component layout (services, pods, backends) | `block-beta` or `flowchart` with subgraphs |
| Timeline of milestones or phased rollout | `timeline` or `gantt` |
| Hierarchy of concepts or decision tree | `mindmap` or `flowchart` |
| Tradeoffs on two axes | `quadrantChart` |
| Simple share breakdown | `pie`, or a table if clearer |
| Branch/merge or release sequencing | `gitGraph` |

Pick by structure, never by habit: one diagram per artifact, ≤12 nodes, no styling, no Unicode math; check an unfamiliar type with the `mermaid info` block. (AGENTS.md; GitHub: diagrams)

### Symbols

| Mark | Use | Never |
|---|---|---|
| `—` | field separator in one-liners; one aside per paragraph in prose | two per sentence; drama |
| `·` | peer fields in a fixed schema | sentence punctuation |
| `⚠️` | a hazard that changes a decision (name collision, withdrawn number) | emphasis; max one per comment |
| `**bold**` | the verdict token; a first definition | every bullet lead-in; whole sentences |
| `✅` / `❌` | pass/fail column; good/bad examples | work status |
| badge | wiki Home or README, live signal only | issues, PRs, ADRs |

`→` only for derivation (`base → child`). Bold is a schema token, not emphasis; emoji is a status marker, not tone. (Wikipedia: Overuse of boldface, Inline-header vertical lists, Emoji as formatting; repo one-liners)

### Length budgets

| Artifact | Budget | Depth goes to |
|---|---|---|
| Issue body | hub ≤60 lines; sub-issue ≤30 | comments |
| Comment | bookkeeping ≤120 words; deep dive: first table within one screen | linked artifact, `<details>` |
| PR body | ≤40 lines | CI, `<details>` |
| Wiki Home | ≤30 lines | linked pages |
| Catalog snapshot | header ≤3 lines | generated rows |
| Writeup | 800–1500 words, ≤8 H2 | appendix |
| ADR | ≤60 lines | linked issue |

These are `[house]`, derived from Google TW's short-document rule. (Google TW: Short sentences)

### AI-tell checklist
- [ ] No "not just X but Y" / "It's not X, it's Y". (Wikipedia: Negative parallelisms)
- [ ] No trailing -ing rider ("highlighting", "underscoring") and no inflation ("pivotal", "testament"). (Wikipedia: Superficial analyses, Undue emphasis)
- [ ] No unnamed authority ("experts argue") and no forced triad. (Wikipedia: Vague attributions, Rule of three)
- [ ] No bold label on every bullet; bold only the verdict token. (Wikipedia: Inline-header vertical lists)
- [ ] No one-line closer that repeats the paragraph, and no "Despite the challenges…" ending. (Wikipedia: Outline-like conclusions)
- [ ] Prose em dashes ≤1 per paragraph; the one-liner `—` is schema, not prose. (Wikipedia: Overuse of em dashes, a historical indicator; repo sample)
- [ ] First sentences of paragraphs, read alone, form the argument; every number has a run or a link, and every method a source. (Google TW: Paragraphs; `[house]`)

## Sources

- [Google Technical Writing One](https://developers.google.com/tech-writing/one): [Short sentences](https://developers.google.com/tech-writing/one/short-sentences), [Paragraphs](https://developers.google.com/tech-writing/one/paragraphs), [Lists and tables](https://developers.google.com/tech-writing/one/lists-and-tables), [Words](https://developers.google.com/tech-writing/one/words), [Active voice](https://developers.google.com/tech-writing/one/active-voice); [ASD-STE100 Issue 9](https://www.asd-ste100.org/about_STE.html) (2025): 20/25-word limits, one topic per sentence, active voice, short noun clusters. STE is not adopted wholesale.
- GitHub Docs: [Basic syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) (footnotes, alerts), [Math](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions), [Autolinks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls), [Diagrams](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams), [Sub-issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues), [Tasklists](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/about-task-lists)
- Scope checks: [github/docs#27919](https://github.com/github/docs/issues/27919) (alerts), [community#56433](https://github.com/orgs/community/discussions/56433) (wiki footnotes). Practice: [Google CL descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html), [KEP template](https://github.com/kubernetes/enhancements/blob/master/keps/NNNN-kep-template/README.md), [MADR](https://adr.github.io/madr/), [cbea.ms/git-commit](https://cbea.ms/git-commit/), [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) and the humanizer skill derived from it.
