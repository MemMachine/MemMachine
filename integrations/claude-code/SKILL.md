---
name: memmachine-timeline
description: >
  Use when you need to recall what happened in an earlier session -- a
  decision and why it was made, an approach that was already tried, how
  something was actually done -- or when the user refers to earlier work
  ("do you remember", "we discussed this", "where did we leave it"). Also use
  before re-deriving something the project has probably settled already.
  Provides search_timeline, expand_timeline and outline_timeline.
---

# Recalling from the timeline

Three tools, and they answer different questions. Reaching for the wrong one
is the usual reason a search comes back empty.

- `search_timeline` — *when was this talked about?* Ranks moments by a cue.
- `expand_timeline` — *what happened around this moment?* Reads the stored
  conversation on either side of an address, in order, including the tool
  calls and their output. This is the only way to reach anything that is not
  a message.
- `outline_timeline` — *where in the conversation was it?* Lists a session's
  events with their sizes, so you can see its shape without reading it.

The normal loop is search, then expand. A search result is a *pointer*: it
tells you where to read, and the reading is what actually answers the
question. Answering from search snippets alone is the commonest way to get a
confident wrong answer, because a snippet shows what was said and not what was
then decided.

## Writing a cue

A cue re-evokes the context a memory was formed in, not just the name of the
thing. Give it enough to pin one specific moment:

- Good: `the user asked why the nightly deploy started failing`
- Good: `User: can we just drop the retry loop entirely?`
- Poor: `deploy` — too diffuse; it pulls every mention.

An event description, or a verbatim line with its speaker, both work. The
user's own wording is often a fine cue. Adding *why* it came up helps.

When you cannot pin the target, search for what surrounded it — what was being
discussed, roughly when — and then expand from a nearby hit to reach it. If
even that is too vague, ask the user for the surrounding context rather than
guessing cues.

Following a lead is just another search with the new cue. There is no separate
tool for it.

## When to stop

Stop when the results are about a different *kind* of thing than the cue was
— that is the signal that the store does not have it, and rewording will not
help. Two rewrites is usually enough. Judge that from what is on the screen,
not from how many results were new.

## Addresses

Every result carries a short `handle`. Pass it back verbatim to
`expand_timeline` or `outline_timeline`. It is an abbreviation that was
unambiguous when it was printed; if it later names more than one thing you
will get a 404 listing the candidates, and a longer prefix resolves it.

## Choosing an expansion unit

`unit="segments"` (the default) is a flat budget: every call costs about the
same, and it is the only way to read *inside* one long entry. `unit="events"`
counts whole turns and tool calls, for when you want a fixed number of
exchanges either side and the length of what is in the way should not decide
how far you get. Reach for `events` when a segment window keeps landing inside
one enormous tool result.

`filter` selects what the window is spent on. In a stretch full of long
output, `filter` to messages; to read what a command actually returned, do not
filter at all. It never hides the moment you asked for.

## What is searchable, and what is not

Only messages are indexed. Tool calls, their results, reasoning, and text the
harness injected into a turn are all *stored* and reachable by expanding, but
they are not ranked by search. So: search for the message that discussed
something, then expand to reach the command that did it.
