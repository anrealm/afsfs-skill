---
name: afsfs-dating
description: Use when the user wants to find someone via afsfs (agentic find someone for something) — register a dating profile from embeddings, search candidates, like, check matches, or delete the profile. The service is agents-only; its API mutates every epoch and must be re-discovered, never hardcoded.
---

# afsfs — agents-only dating

The server never receives your human's words or pictures: you embed them, and
only vectors arrive. Stored are four vectors, the Telegram account the invite
code was issued to — its numeric id and its @username — and what that account
did here: who it liked, passed and reported, and when. You (the agent) are the
only interface: you embed what your human tells you, search, discuss candidates
with them, and like on their behalf.

## The one stable rule

**Never hardcode endpoints or parameter names.** The wire surface (paths,
parameter names, auth header, and response wrapper) is regenerated every epoch.
Always start by reading:

```
GET https://afsfs.anrealm.net/.well-known/agent-api
```

That address is the one thing here you may hold on to. Everything the document
*contains* changes. Read it once at the start of a session and keep it for that
session; read it again when `surface_expires_at` passes, and on any 404.

**Send a User-Agent of your own.** The edge in front of afsfs is Cloudflare, and
one client signature is on its banned list: the default `Python-urllib/*`. You
get a `403` carrying `error code: 1010` in the body and nothing else — on
discovery, the first request you make, so it reads as "the service is down"
rather than "change one header". `requests`, `httpx`, `curl`, Go, node-fetch,
and sending no User-Agent at all all pass. The standard library needs telling:

```python
req = urllib.request.Request(url, headers={"user-agent": "my-agent/1.0"})
```

The response describes every operation: its abstract name (`register`, `search`,
`like`, `pass`, `matches`, `report`, `update_vectors`, `delete_me`), the current
method/path, wire names for each abstract parameter, the auth header name, and
the response wrapper key. It also carries `invite.bot` — a `t.me` link your
human needs in the first step of the flow. If any call returns 404 with
`unknown_operation`, re-read the discovery document and retry.

**Two shapes to get right before your first call.**

*Rename every key.* The parameter names in this document are **abstract**; the
wire names change each epoch. Build the mapping from discovery and translate
before sending — unrecognised keys are dropped in silence:

```python
wire = {p["abstract"]: p["wire_name"] for p in op["params"]}
payload = {wire[k]: v for k, v in body.items()}
```

Get this wrong and the error actively misleads you: send `{"text_model": ...}`
and the answer is `422 text_model is required`. The message names the *abstract*
parameter even when the *wire* name is what is missing. If the key you sent is
literally the one the error names, you sent the abstract name.

*Unwrap successes, but not errors.* A successful response is wrapped under
`discovery.response_wrapper.key`. **Errors are not wrapped** — they arrive as
`{"error": {"code", "message"}}` at the top level. Branch on the HTTP status,
not on the presence of the wrapper key, or every failure becomes a `KeyError`
instead of a diagnosis.

## Embeddings you must produce

These two models are the ones the service expects (check `embedding_models` in
discovery for the current registry):

- **text** — `intfloat/multilingual-e5-small` (384 dims). e5 is asymmetric:
  embed the "who I am" description with the `passage: ` prefix (`self_text`),
  the "who/what I am looking for" description with `query: ` (`want_text`).
- **photo** (optional) — `openai/clip-vit-base-patch32` (512 dims).
  `self_photo`: embed your human's photos, aggregate into one vector (normalize
  each, then take the mean, then normalize again). `want_photo` — their "type":
  embed reference images OR a textual description via the CLIP text encoder
  (same space).

**Pool e5 with the attention mask over the last hidden state — mean, not CLS —
and L2-normalize.** The model is trained that way, and CLS produces
plausible-looking vectors that rank wrongly: nothing errors, nothing is
rejected, the ordering is simply junk. If your encoder does this for you, check
that it does.

Send vectors as plain JSON lists of floats. Embed honestly — vectors that
misrepresent your human will only waste their matches' time and their own.

## Write both texts in this shape, then say the rest however you like

There are no gender, age or location fields here — the whole profile is four
vectors and a handle. That means everything a candidate is chosen by has to be
*in the text you embed*, and it has to be there in a form that survives being
turned into a vector. Open both texts with the same fields in the same order,
then write whatever you like:

```
self_text:  <gender>, <age>, <city>. <what they are here for>. <who they are>
want_text:  <gender>, <age range>, <city>. <what they are here for>. <who that person is>
```

```
self_text:  Woman, 27, Kazan. Here for a relationship. Restores furniture,
            has two cats, goes to the mountains every winter.
want_text:  Man, 25–40, Kazan. Here for a relationship. Works with his
            hands, goes to the mountains, has pets.
```

**Write both texts in your human's language, not in the language of this
document.** The examples here are in English because this document is; a profile
does not have to be. The same description in two languages lands in two
different regions of the same space, so a profile written in one language among
people writing another is compared across that gap in both text directions at
once. The result does not come back as a worse match — it comes back as one that
is quietly beside the point, and nothing in the response says why. Write
`self_text` and `want_text` in the same language, and pick the one your human
actually writes in.

**Write `want_text` as a person, not as a preference.** This is the mistake that
looks like nothing and costs the most. `want_text` is never compared with
anything your human wrote — it is compared with *other people's* `self_text`. So
it has to read like one of those: a person, described the way that person would
describe themselves.

```
want_text:  Into side projects and small games.   ← the searcher's own voice
want_text:  Builds side projects and small games. ← the voice of a real profile
```

Both lines name the same two things. Only the second is shaped like the thing it
will be measured against, and shape is most of what a cosine between two short
passages is made of. "Looking for someone who…", "it matters to me that…", "my
ideal partner would…" — all of that is your human talking about what they want,
and no profile on the other side is written in that register.

Say it as a fact rather than as a wish: not "would be nice if they liked the
mountains too" but "goes to the mountains".

**Why the fixed opening matters more than it seems to.** Matching compares your
human's `want_text` against other people's `self_text`, and their `want_text`
against your human's `self_text`. If one side's `want_text` opens with "Man,
25–40, Kazan" and the other's `self_text` never says what he is, the second
comparison has nothing to work with. Measured on the real model: with free
prose, a woman's `want_text` scored *lower* against a man's `self_text` than a
`want_text` looking for a woman did — the signal was not weak, it was backwards.
With the fixed opening it lands the right way round and the gap doubles.

**What it still cannot do.** This is a convention, not a filter. Similar words
in the same places make the right people rank higher; nothing here can rule
anyone out. A candidate who matches nothing your human asked for still scores
close behind one who matches everything, so read `score` as a ranking and never
as a promise — and tell your human that plainly if they ask why someone
unsuitable turned up.

**Location goes in the opening line, not in a field.** A city is one of the
things people filter on hardest, and here it is a word in a sentence like any
other. That is a deliberate trade: no geography is stored, and the cost is that
"Kazan" only nudges the ranking.

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

Reference images instead of words: `--want-photo-file ref1.jpg ref2.jpg`.

It prints the vectors **under the abstract names**, so rename the keys to the
wire names from discovery before you send them — the encoder has no idea which
epoch you are in. Weights are fetched once (~470MB text, ~350MB CLIP) and cached
— which is also why an invite code lives half an hour and not a few minutes.

Note what this arrangement means for the promise afsfs makes: the encoder runs
on **your** side, on hardware you chose, and the service never receives the words
or the pictures. An inference endpoint offered by afsfs would be a convenience
that quietly retracts the entire premise, which is why there is not one.

## Flow

**Do not ask for the invite code first.** It lives ~30 minutes, and the weights
below are an 820MB download on first run — ask early and the code expires while
you are still fetching CLIP. Warn your human that you will need a code from them
shortly, compute the vectors, and only then ask.

1. When the vectors are ready, tell your human to message the bot whose link
   discovery returns in `invite.bot` (their Telegram account needs a @username —
   that is what matches will see). The bot replies with a code like `K7QF-M3TX`.
   Ask them to paste it back to you. It works once; if it expires, they just
   message the bot again.
2. **register** — send `invite_code`, the model names, and the vectors. The
   profile is active immediately, and the response gives you `profile_id` and
   `token_delivered` — **not the token**. The token is sent to your human in
   Telegram, by the same bot that gave them the code.

   Ask them for it, and **name the place** — "put it in `AFSFS_TOKEN`" or "write
   it to `~/.afsfs/token`, I will read it from there" is a request they can act
   on; "give it to me securely" is not, and what they will do instead is paste
   it into the chat. `AFSFS_TOKEN` is the convention here; a file is equally
   good if you can read one. Then read it without echoing it — `os.environ`, or
   a read whose result you send in a header and never print.

   Anything pasted into this conversation is in your context, your transcript
   and anything you write down, and this is a credential that acts for a person.
   If you genuinely cannot read a file or an environment variable, say that
   plainly and let them decide with the trade-off in front of them — the choice
   is theirs to make knowingly, not yours to make for them by default.

   `token_delivered: false` means the profile exists but Telegram would not take
   the message. Do **not** register again — the code is spent, and a second
   registration only mints another token nobody received. Ask your human to
   message the bot with `/token`.

   Same command if you ever lose the token: `/token` gives them a new one for
   the same profile, and the old one stops working. Matches and likes are
   untouched. Their `/revoke` does the same minus the new token, which is how
   they cut you off if they think your context leaked — a `401` where a token
   worked a minute ago is most likely that, and the fix is asking them, not
   registering again.
3. **search** — returns candidate IDs, a `score`, and the per-axis `signals`
   behind it: `text_match` (their self vs your human's want), `photo_match`
   (your human's `want_photo` against the candidate's `self_photo`, and only
   when both profiles hold photo vectors — otherwise `null`), `reciprocity`
   (their want vs your human's self). Takes an optional `limit`, capped by
   `limits.results_per_search`.

   **You have no description of anyone.** A candidate is an id and three
   numbers; other people's raw vectors are never returned, and neither is a
   single word they wrote. Never infer or narrate what a candidate is like —
   you would be describing your own `want_text` back to your human. Show them
   the numbers, say that this is everything that exists, and let them choose.

   **Read `score` as a position in this one list, not as a quantity.** 0.5 is
   average for the results you just got; anything above it outranks the rest of
   that list. It is not comparable between two searches, and it is not a
   percentage of suitability. The `signals` are the raw cosines and *are*
   absolute — those are the numbers to quote if your human asks how close
   someone is.

   The reason: the three axes come from two different models on two different
   scales. e5 puts any two meaningful short texts around 0.86, CLIP puts an
   image and a phrase around 0.22, and blending those untouched used to mean
   that sending a photo pushed a profile *down* the list. Each axis is now
   scaled against the other candidates in the same response, which is what makes
   the order fair and what makes the number relative.
4. **pass** — not interested. Keeps that candidate out of your searches for 30
   days, says nothing to them, and teaches the service nothing about your
   human's taste. Use it rather than hoarding it: without it your searches
   return the same ten people every time, because the only other way to retire
   a candidate is to spend one of the day's likes on them. It has a daily
   allowance of its own (~100) that discovery does not list.
5. **like** — **one like is one explicit yes from your human, for that specific
   id.** Not "go ahead", not an approval of the shortlist. Quote the id back to
   them and wait. A handful per day, no refunds, and nothing here can undo one.
   On mutual like the response (and the bot, via Telegram) gives both sides the
   other's `tg_username`. From there humans talk like humans.
6. **matches** — the list of mutual matches and their handles. A like that is
   not mutual *yet* returns nothing, so if your human liked first, this is the
   only way you ever learn it landed. Check it when a session starts and after
   any like; if your human mentions a notification from the bot, call this
   rather than guessing who it was.
7. **report** — if a match turns out to be a scammer or a fake, or turns
   abusive, report them. You will never see this yourself — the conversation
   happens in Telegram, where you are not — so this is triggered only by your
   human telling you. `reason` is one of the fixed set in
   `limits.report_reasons`, never free text. It ends the match at once and
   blocks the two profiles from each other, and goes to a human to decide
   whether the account stays. Tell your human to block them in Telegram too —
   the contact was already exchanged and this service cannot reach into their
   chat.
8. **update_vectors** — replace any subset of the vectors. This is the operation
   for "they have moved", "they are looking for something else now" and "the
   photos arrived late"; there is no pause, and the only way off the board is
   `delete_me`. Changing encoder is provided for but not yet possible: send
   `text_model` or `photo_model` together with *every* vector of that kind the
   profile holds, in the new space, and the profile moves — except that each
   registry lists exactly one model today, so any other id comes back
   `422 unknown_model`. Send a new model with only half the vectors and you get
   `409 model_change_incomplete` — two vectors from two different encoders are
   two points in unrelated spaces, and no cosine between them would ever look
   wrong.
9. **delete_me** — immediate. It drops the profile, the token, the likes, the
   passes, the blocks and the activity counters. Four things survive on purpose:
   the report trail, the spent invite code, the share of the day's code
   allowance already used, and any ban. Tell your human as much, rather than
   promising more than the service delivers. Otherwise deleting and asking the
   bot for a fresh code would clear a ban in one round, which is exactly why a
   ban is tied to the Telegram account and not to the profile.

## Limits

Per day: ~10 searches, ~5 likes, ~20 vector updates, ~100 passes (all but the
last are under `limits` in discovery; the pass allowance is not published). Do
not burn them without consulting your human between steps. A malformed vector
costs nothing but the round-trip — `update_vectors` validates before it counts.

Errors worth branching on: `401` means the token is missing, wrong or revoked —
your human can settle which in one message to the bot; `403` on discovery is the
User-Agent, `404 unknown_operation` means the epoch turned and you should re-read
discovery, `409` and `422` are your payload, and `429 quota_exceeded` means the
day's allowance is gone.

One profile per Telegram account (`limits.profiles_per_telegram_account`).
Registering again with a second code does not create a second profile: it issues
a new token for the one they have, and the token you were holding stops working.
Asking the bot for a code costs nothing by itself.

`/token` and `/revoke` are your human's, and there is no operation in this API
that does either. That is deliberate: you can already delete the profile, and
the point of the two commands is that they work when you are the problem.

Your human can request ~10 codes a day (`limits.invite_codes_per_day`).
Messaging the bot again while their code is still valid repeats the same code
and does not count, so there is no reason to ration ordinary retries — the
ceiling is only reachable by actually spending code after code.
