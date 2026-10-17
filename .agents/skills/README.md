# Agent skills

This is the single source of truth for every coding harness. `SKILL.md` files follow the open [Agent Skills](https://agentskills.io) format: frontmatter `name` matches the folder name, and the `description` stays short, because harnesses preload every description.

| Harness | Instructions | Skills |
| --- | --- | --- |
| Codex | `AGENTS.md` | `.agents/skills/` (native) |
| OpenCode | `AGENTS.md` | `.agents/skills/` (native). It also sees the `.claude/skills` symlink but lists each skill once (verified on 1.4.6). Don't set `OPENCODE_DISABLE_CLAUDE_CODE_SKILLS`: on 1.4.6 it hides these skills too |
| Claude Code | `CLAUDE.md` → imports `AGENTS.md` | `.claude/skills` → symlink to `.agents/skills` |
| Cursor, others | `AGENTS.md` | Read the paths listed in `AGENTS.md` |

Windows: clone with `git config --global core.symlinks true` (and Developer Mode on), or Claude Code won't see the skills.

## Writing skills
- Keep each `SKILL.md` around 3–4 KB of rules, defaults and gotchas an agent can't infer from the code. Put long reference material in `references/`.
- Keep descriptions ≤ ~250 chars: what it covers + trigger words.
- Don't restate the code. Point to the file.

## Sources
| Skill | Derived from | License |
| --- | --- | --- |
| `crusoe-inference` | [crusoe-developer-hub hackathon skill](https://github.com/crusoecloud/crusoe-developer-hub/blob/main/examples/hackathons/crusoe-cv-recursive-agents-hackathon/crusoe-cv-recursive-agents-hackathon-skill.md); verbatim copy in `references/` (fetched 2026-10-17) | MIT |
| `neon` | [neondatabase/agent-skills](https://github.com/neondatabase/agent-skills) `neon`, `neon-postgres`, `neon-object-storage`, `neon-postgres-branches`; verbatim copies in `references/upstream/` (commit 51c7d59) | Apache-2.0 |
| `modal-sandboxes` | Modal docs + SDK 1.6.2 source, see `references/full-guide.md` | - |

The condensed `SKILL.md` files are modified summaries of these sources. When upstream changes, re-fetch the verbatim copies and update the summaries.
