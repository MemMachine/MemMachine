# Property keys

An event carries properties: keys with scalar values that a search can
filter on. The caller and MemMachine's own components both write them,
into one space of names that the event memory's segment store and its
vector store both hold. This document fixes how that space is divided
between them.

## The rule

A key that begins with `memmachine_` belongs to MemMachine. A component
builds each of its keys as `memmachine_<component>_<field>`, where the
component token names the code that writes the key: `em` for the event
memory, and `ltm` for the long-term memory's event backend. Every other
key belongs to the user: an identifier within the bound that every
supported backend accepts for a field name, stored and filtered exactly
as written. The bound is the vector stores' naming rule, so any user key
can be declared to a vector store as written. The reservation covers
property keys only; an id a caller names may begin with the prefix,
since ids are not keys.

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
keys are a closed set it stamps on every event; and the memory's own
components. A scheme that gives one party a privileged tier leaves the
others sharing a space that stays disjoint only by luck, and a key the
memory adds later can collide with one an application already writes.

A leading underscore was rejected as the reservation. It suits a space
one party owns. Shared by several, it spends the one obvious prefix on a
single component, leaves a second component or an embedding application
no sanctioned way to reserve names of its own, and takes the whole family
of underscore names away from callers. Shared spaces are divided by a
qualified prefix instead, the way a database reserves its catalog's
names.

The prefix is the distribution name, so the package registry keeps it
unique, and a reader of stored data can tell at once who owns a key. It
is reserved as a whole and checked as a prefix, so a field a component
adds later is protected without a list to update. Within it, each
component writes under its own token, named for the code that writes the
key rather than for a data model, so a key can be attributed without
knowing the codebase and no central list of keys is needed. Tokens are
short, because every key shares the bound the backends agree on.

The reservation also leaves the API free in how a caller reaches a
component's field. Whether the field is a typed search parameter or a
name a filter routes to the reserved key, no caller can write the key,
so neither choice can collide with a user's.

## Reservation, never rewriting

Leaving every name to the caller and rewriting user keys with a prefix on
the way to storage was rejected. The key a caller writes is the key the
caller filters on; a rewriting undone on every read is a round trip that
cancels; and a rewritten user space can still be made to address a
system key through the filter language. Reserving a length instead of a
prefix was rejected too, since it pads every system key to the bound and
makes stored data unreadable.

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
outside the identifier rule and writes declared keys as they are.
Rewriting at that level escapes the objections above, since callers write
and filter keys as written, only the store sees the physical name, and a
filter on a property never reaches one of the store's own fields.

A layer above MemMachine that offers keys outside the rule keeps its own
mapping from its keys to legal ones.
