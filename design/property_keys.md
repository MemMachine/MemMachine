# Property keys

An event carries properties: keys with scalar values that a search can
filter on. The caller and MemMachine's own components both write them,
into one space of names that the event memory's event memory store and its
vector store both hold. This document fixes how that space is divided
between them.

## The rule

A key that begins with `memmachine_` belongs to MemMachine. A component
builds each of its keys as `memmachine_<component>_<field>`, where the
component token names the code that writes the key: `em` for the event
memory, and `ltm` for the long-term memory's event backend. Every other
key belongs to the user: an identifier under the vector stores' naming
rule, stored and filtered exactly as written, so any user key can be
declared to a vector store as written. The reservation covers property
keys only; a value, including a session or producer id a caller names,
may begin with the prefix.

A user key is checked against the rule where it enters, before any store
is written, so a request with an illegal key fails whole and no store
keeps part of it. The event memory refuses a key under its own token,
since it writes those keys itself; a key under another component's token
belongs to the component that embeds the memory and is kept as given. In
the long-term memory's event backend, a filter names a component's field
by its plain name, and a filter that names a reserved key is refused.

## Why one reserved prefix

Three parties write keys into the same space: the end user, whose keys
are an open, untrusted set; the application that embeds the memory, whose
keys are a closed set it defines; and the memory's own components. A
scheme that gives one party a privileged tier leaves the others sharing a
space that stays disjoint only by luck, and a key the memory adds later
can collide with one an application already writes.

A leading underscore was rejected as the reservation. Underscore names
suit a space one party owns, where a few exact names are reserved. Shared
by several, a leading underscore spends the one obvious prefix on a
single component and leaves a second component or an embedding
application no sanctioned way to reserve names of its own. Shared spaces
are divided by a qualified prefix instead, the way a database reserves
its catalog's names.

Reserving characters outside the user alphabet was rejected: MemMachine's
own keys go to the same stores as user keys, whose naming rule admits
only the user alphabet. Widening it would take a review of every backend
for a character each one accepts within its length and other limits.

The prefix is the product's name, distinctive enough that users are
unlikely to want keys under it, and a reader of stored data can tell at
once who owns a key without knowing the codebase. A short top-level
prefix per memory system was rejected, since each system added would
reserve another short family of names users do want. The prefix is
reserved as a whole and checked as a prefix, so a field a component adds
later is protected without a list to update. Within it, each component
writes under its own token, named for the lowest level that writes the
key, the store wrapper or the memory type, rather than for a generic data
model, so no central list of keys is needed. Tokens are short, so keys
stay within the identifier rule's length with room to spare.

The reservation also leaves the API free in how a caller reaches a
component's field. Whether the field is a typed search parameter or a
name a filter routes to the reserved key, no caller can write the key,
so neither choice can collide with a user's.

## Keys are validated, not repaired

A key outside the rule is refused rather than repaired: input is
validated, not sanitized, since encoding a key into the alphabet
lengthens it against the rule's length, so a long key overflows it or,
truncated, collides with another. Reserving a length instead of a
prefix, with MemMachine's keys padded to that length, was rejected too,
since padded keys are unreadable in stored data.

## Where the rule applies

An event's properties are a shared space because the vector stage
filters on them: a vector record carries its properties as one map, and
a field a search narrows by at that stage must be among them. Keeping
MemMachine's fields out of the vector record was rejected, since it
leaves them filterable only after the vector search, which costs
selectivity on every query that filters by them.

The rule is needed only where MemMachine's keys and user keys share one
space of names. A store that keeps its own fields in their own columns or
attributes, and user keys in a separate map or column, is separated by
structure and needs no reservation. It takes up the prefix if its fields
ever move into a shared space.

A store's physical names follow the opposite rule: the store keeps its
own fields under names no property key can take. A store either gives
every declared key, MemMachine's included, a prefix internal to the store
that none of its own field names begins with, or names its own fields
outside the identifier rule and writes declared keys as they are. Either
way the rule stays intact, since callers write and filter keys as
written, only the store sees the physical name, and a filter on a
property never reaches one of the store's own fields.

A layer above MemMachine that offers keys outside the rule keeps its own
mapping from its keys to legal ones.
