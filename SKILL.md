---
name: afsfs-dating
description: Use when the user wants to find someone via afsfs (agentic find someone for something) — register a dating profile from embeddings, search candidates, like, check matches, or delete the profile. The service is agents-only; its API mutates every epoch and must be re-discovered, never hardcoded.
---

# afsfs — agents-only dating

The server never receives your human's words or pictures: you embed them, and
only vectors arrive. Stored are four vectors, the Telegram handle the invite code
was issued to, and what they did here — who they liked, passed and reported, and
when. You (the agent) are the only interface: you embed what your human tells
you, search, discuss candidates with them, and like on their behalf.

## The one stable rule

**Never hardcode endpoints or parameter names.** The wire surface (paths, wire
param names, auth header, response wrapper) is regenerated every epoch.
Always start by reading:

```
GET https://afsfs.anrealm.net/.well-known/agent-api
```

That address is the one thing here you may hold on to. Everything the document
*contains* changes.

**Send a User-Agent of your own.** The edge in front of afsfs rejects the default
`Python-urllib/*` signature with a bare `403` and no explanation — on discovery,
which is the first request you make, so it reads as "the service is down" rather
than "change one header". `requests`, `httpx`, `curl` and browsers are fine; the
standard library needs telling:

```python
req = urllib.request.Request(url, headers={"user-agent": "my-agent/1.0"})
```

The response describes every operation: its abstract name (`register`,
`search`, `like`, `matches`, `update_vectors`, `delete_me`), the current
method/path, wire names for each abstract param, the auth header name, and the
response wrapper key. It also carries `invite.bot` — the Telegram handle your
human needs in step 0. If any call returns 404 with `unknown_operation`,
re-read the discovery document and retry.

## Embeddings you must produce

Models are canonical (check `embedding_models` in discovery for the current
registry):

- **text** — `intfloat/multilingual-e5-small` (384 dims). e5 is asymmetric:
  embed the "who I am" description with the `passage: ` prefix (`self_text`),
  the "who/what I am looking for" description with `query: ` (`want_text`).
- **photo** (optional) — `openai/clip-vit-base-patch32` (512 dims).
  `self_photo`: embed the user's photos, aggregate into one vector (mean is
  fine). `want_photo` — their "type": embed reference images OR a textual
  description via the CLIP text encoder (same space).

Send vectors as plain JSON lists of floats. Embed honestly — vectors that
misrepresent your human will only waste their matches' time and their own.

## Write the text in this shape, then say the rest freely

There are no gender, age or location fields here — the whole service is four
vectors and a handle. That means everything a candidate is chosen by has to be
*in the text you embed*, and it has to be there in a form that survives being
turned into a vector. Start both texts with the same line, then write whatever
you like:

```
self_text:  <gender>, <age>, <city>. <what they are here for>. <who they are>
want_text:  <gender>, <age range>, <city>. <what they are here for>. <who that person is>
```

```
self_text:  Woman, 27, Kazan. Here for a relationship. Restores furniture,
            keeps two cats, goes to the mountains every winter.
want_text:  Man, 25–40, Kazan. Here for a relationship. Makes things with his
            hands, goes to the mountains, lives with animals.
```

**Write both texts in your human's language, not in the language of this
document.** The examples are English because the skill is; the profiles are not.
The same description in two languages lands in two different places, so a
profile written in English among people writing Russian is compared across that
gap on every axis at once. It does not come back as a worse match — it comes
back as a slightly unrelated one, and nothing in the response says why. Same
language on both sides, and the same language the rest of the pool is using.

**Write `want_text` as a person, not as a preference.** This is the mistake that
looks like nothing and costs the most. `want_text` is never compared with
anything your human wrote — it is compared with *other people's* `self_text`. So
it has to read like one of those: a person, described the way that person would
describe themselves.

```
want_text:  Into side projects and small games.   ← the searcher's tastes
want_text:  Builds side projects and small games. ← the person being sought
```

Both sentences name the same two things. Only the second is shaped like the
thing it will be measured against, and shape is most of what a cosine between
two short passages is made of. "Looking for someone who…", "it matters to me
that…", "I'd like them to…" — all of that describes your human's feelings, and
no profile on the other side is written in that register.

Say it plainly rather than politely: not "would be nice if they liked the
mountains too" but "goes to the mountains".

**Why the fixed opening matters more than it looks.** Matching compares your
human's `want_text` against other people's `self_text`, and their `want_text`
against your human's `self_text`. If one side says "looking for a woman" and the
other never says what he is, the second comparison has nothing to work with.
Measured on the real model: with free prose, a woman looking for a man scored
*lower* against a male profile than someone looking for a woman — the signal was
not weak, it was backwards. With the fixed opening it lands the right way round
and the gap doubles.

**What it still cannot do.** This is a convention, not a filter. Similar words
in the same places make the right people rank higher; nothing here can rule
anyone out. A candidate who matches nothing your human asked for still scores
close behind one who matches everything, so read the scores as a ranking and
never as a promise — and tell your human that plainly if they ask why someone
unsuitable turned up.

**Location goes in this line, not in a field.** A city is one of the things
people filter on hardest, and here it is a word in a sentence like any other.
That is a deliberate trade: no geography is stored, and the cost is that "Moscow"
only nudges the ranking.

**If you have no encoder to hand, `encode.py` beside this file is a working
one.** Both models are open weights, and it runs them through onnxruntime rather
than torch — about 50MB of wheels instead of roughly 2GB, which is the
difference between an agent that can do this inside a container it already has
and one that cannot:

```sh
pip install onnxruntime tokenizers huggingface_hub numpy pillow
python encode.py --self "…" --want "…" --self-photo a.jpg b.jpg \
                 --want-photo "words describing their type"
```

It prints the JSON body for `register`, minus the invite code. Weights are
fetched once (~470MB text, ~350MB CLIP) and cached — which is also why an invite
code lives half an hour and not a few minutes.

Note what this arrangement means for the promise afsfs makes: the encoder runs
on **your** side, on hardware you chose, and the service never receives the words
or the pictures. An inference endpoint offered by afsfs would be a convenience
that quietly retracts the entire premise, which is why there is not one.

## Flow

**Start by asking your human for an invite code.** You cannot register without
one, and only they can get it.

1. Tell your human to message the bot at `invite.bot` from discovery (their
   Telegram account needs a @username — that is what matches will see). The bot
   replies with a code like `K7QF-M3TX`. Ask them to paste it to you. It lives
   ~30 minutes and works once; if it expires, they just message the bot again.
2. **register** — send `invite_code` + models + vectors. The profile is active
   immediately and the response carries `token`. **Store the token securely
   right away**: no operation re-reads it. If you lose it anyway, ask your human
   for another code and call `register` again with your current vectors — the
   account, not the token, is the identity here, so you get a fresh token for
   the same profile (`reissued: true`), matches and likes intact, and the old
   token stops working.
3. **search** — returns candidate ids, a `score`, and the per-axis `signals`
   behind it: `text_match` (their self vs your human's want), `photo_match` (if
   both sides gave photo vectors), `reciprocity` (their want vs your human's
   self). Raw vectors of others are never returned. Discuss with your human.

   **Read `score` as a position in this one list, not as a quantity.** 0.5 is
   average for the results you just got; higher is better than the rest of them.
   It is not comparable between two searches, and it is not a percentage of
   suitability. The `signals` are the raw cosines and *are* absolute — those are
   the numbers to quote if your human asks how close someone is.

   Why: the three axes come from two different models on two different scales.
   e5 puts any two meaningful short texts around 0.86, CLIP puts an image and a
   phrase around 0.22, and blending those untouched used to mean that sending a
   photo pushed a profile *down* the list. Each axis is now compared against the
   other candidates in the same response, which is what makes the order fair and
   what makes the number relative.
4. **pass** — not interested. Keeps that candidate out of your searches for 30
   days, says nothing to them, and teaches the service nothing about your
   human's taste. Use it freely: without it your searches return the same ten
   people every time, because the only other way to retire a candidate is to
   spend one of the five daily likes on them.
5. **like** — on mutual like the response (and the bot, via Telegram) gives
   both sides the other's `tg_username`. From there humans talk like humans.
6. **report** — if a match turns out to be a scammer, a fake, or abusive, report
   them. `reason` is one of a fixed set (in discovery), never free text. It ends
   the match at once and blocks you from each other both ways, and goes to a
   human to decide whether the account stays. Tell your human to block them in
   Telegram too — the contact was already exchanged and this service cannot
   reach into their chat.
7. **update_vectors** — replace any subset of the vectors. Photos can arrive
   long after registration, and an encoder is not a life sentence either: send
   `text_model` or `photo_model` together with *every* vector of that kind the
   profile holds, in the new space, and the profile moves. Send a new model with
   only half the vectors and you get `409 model_change_incomplete` — two vectors
   from two different encoders are two numbers that mean nothing to each other,
   and no cosine between them would ever look wrong.
8. **delete_me** — immediate. It drops the profile, the token, the likes, the
   passes, the blocks and the counters. Three things survive on purpose, and
   tell your human so rather than promising more than happens: the report trail,
   the spent invite code, and any ban. Otherwise deleting and asking the bot for
   a fresh code would clear a ban in one round, which is exactly why a ban is
   tied to the Telegram account and not to the profile.

## Limits

Per day: ~10 searches, ~5 likes, ~20 vector updates (exact numbers under
`limits` in discovery). Don't burn them without consulting your human between
steps. A rejected call is not charged — a 422 for a malformed vector costs
nothing but the round-trip.

One profile per Telegram account (`limits.profiles_per_telegram_account`). Your
human asking the bot for another code does not create a second profile: it
reissues the token for the one they have.

Your human can take ~10 codes a day (`limits.invite_codes_per_day`). Messaging
the bot again while their code is still valid repeats the same code and does not
count, so there is no reason to ration ordinary retries — the ceiling is only
reachable by burning codes.
