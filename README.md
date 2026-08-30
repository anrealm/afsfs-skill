# afsfs — the skill

**afsfs** is a dating service with no photo galleries, no bios, no feed and no UI, reachable only by agents.
It never receives words or pictures: your agent turns what you told it into vectors, and only the
vectors arrive. This repository is how an agent learns to use it.

Service: <https://afsfs.anrealm.net>

```
SKILL.md     the skill — portable Markdown with YAML frontmatter
encode.py    a working encoder, in case the agent has none
```

## Install

Put `SKILL.md` where your harness looks for skills — the format is the one Claude Code, Copilot CLI
and Gemini CLI all read. Clone it under the skill's own name; there is no manifest to write, because
a single skill needs none.

```sh
git clone https://github.com/anrealm/afsfs-skill ~/.claude/skills/afsfs-dating   # Claude Code
git clone https://github.com/anrealm/afsfs-skill ~/.copilot/skills/afsfs-dating  # Copilot CLI
git clone https://github.com/anrealm/afsfs-skill ~/.gemini/skills/afsfs-dating   # Gemini CLI
```

Or skip installing entirely and hand the agent the file. It is written to be read start to finish
and says everything needed to complete a registration, so nothing in it depends on being installed.

## What the agent has to do and cannot skip

**Compute the embeddings itself.** There is no inference endpoint, and that is the design rather
than a gap: an endpoint that accepted text would retract the whole premise. Two open-weight models:
`intfloat/multilingual-e5-small` for text, and `openai/clip-vit-base-patch32` for photos if the
human supplies any. `encode.py` is a working implementation on onnxruntime — about 50MB of wheels
where torch would want roughly 2GB, the difference between fitting in a container the agent already
has and not fitting at all.

**Ask its human for an invite code.** Registration is closed without one. The human messages the
Telegram bot, gets a single-use code good for half an hour, and pastes it to the agent. The code
carries the Telegram identity, so nothing about whom the profile belongs to is taken from the caller.

**Ask its human for the token, too.** No API response carries it. `register` sends the token to the
Telegram account the code belonged to, because a code proves a human vouched for that account and
proves nothing about the agent it was handed to — and an agent that cannot keep a credential out of
its own transcript is the ordinary case. The human passes it on; the agent should ask for it as a
file or an environment variable rather than a pasted message, and say plainly that a paste lands in
its context. Their `/token` issues a new one, their `/revoke` takes it back, and neither exists in
the API.

**Re-read discovery every time.** Paths, wire parameter names, the auth header and the response
wrapper are all regenerated each epoch from a secret. Only `/.well-known/agent-api` is stable.
Hardcode anything else and the integration breaks on a schedule.

## Three things that bite an unprepared client

**The parameter names in the skill are abstract; the wire names change each epoch.** Rename every
key from discovery before sending, or the request arrives empty — and the error will name the
abstract parameter you *did* send, which reads as a broken service rather than a wrong key.

**Successful responses are wrapped; errors are not.** Branch on the HTTP status, not on the presence
of the wrapper key, or every failure surfaces as a `KeyError`.

**A default `Python-urllib` User-Agent is rejected at the edge** with a `403` carrying `error code:
1010` — on discovery, the very first request, so it reads as an outage rather than as a header
problem. One line fixes it, and the skill says which.

## If these files and the service disagree

The service is the source of truth and the skill is a description of it, so a disagreement means
the skill is stale, not that the service is wrong. Discovery always describes the live wire
surface — trust it over anything written here.
