# Semantic sets and their owners

## Problem

Semantic features live in sets. Each set belongs to an org. A set either
belongs to one project of that org or is shared by all of them, and within
that scope the set metadata (tag values such as a producer id) picks it out.

Set ids used to be strings that joined the org, the project and the set
metadata with underscores. Org and project ids may contain underscores, so
two owners could be given the same string: the project-level user set of
org `acme`, project `web`, and the org-level user set of org
`acme_project_web` got one id, and so shared one set. The id string was
also the only record of a set's owner, so operations that took an id could
not check that the caller owned it, and deleting a project could not find
the project's sets.

## Design

### A registry of sets

The config store keeps one row per set: its id, org, project (none for a
set of the whole org) and set metadata. It is unique per org, project and
set metadata. The owner is data, and checks compare it for equality. Ids
carry no meaning, and new sets get random UUIDs.

### Resolving the sets of a request

The sets a message or query reaches come from the org's set types that the
request's metadata covers, plus the default sets: the org's set, the
project's set, and the producer's set when the metadata names a producer.

- Writes (adding a message, asking for a set id) add the sets that are
  missing, then read them back. Writers racing to add one set keep a single
  row.
- Reads (search, listing features, listing sets) use only sets that exist,
  so a read never creates state. A set nothing has written to has nothing
  to read.

### What an operation reaches

An operation acts for one project of an org. It reaches that project's sets
and the org-level sets of the org, which every project of the org shares.
Set types and their template categories belong to the org. A feature
belongs to its set, a category to its set or set type, and a tag to its
category. Because a template category is the org's, deleting it deletes its
features from every set of its type, in every project of the org, except
sets that override it.

Anything outside that scope behaves as if it does not exist. Reads return
nothing, deletes skip it, and writes into it are refused as not found. A
response therefore never shows that another tenant's id exists.

### Deleting a project

Deleting a project deletes the project's sets, tagged or not, with their
features, queued history and per-set configuration, and then their registry
rows. Org-level sets stay. A project created later with the same ids gets
new, empty sets.

### Existing set ids

Data written before the registry stays under its legacy ids, which become
the ids of registered sets; nothing is rewritten. When the semantic service
is built, it registers every unregistered legacy id found among stored
features, history and set configuration.

Registration reads each id back into an org, project and set metadata, and
keeps a reading only if the legacy format turns it back into exactly that
id. Some ids read more than one way. Every project-level set with tags also
reads as an org-level set of an org named `<org>_project_<project>`. For
such ids, a reading is kept only if its org has a set type of that level
and those tag keys (any set type, for a reading without tags), since every
tagged set came from a set type and every producer set comes with the org's
user set type. An id still left with several readings belonged to sets
that two owners already shared. It stays unregistered, because no single
owner can be given data that is already mixed, and no operation reaches
it. The evidence can only pick wrongly between two orgs whose ids nest
(`<org>_project_<project>`), when the true owner has no set type of the
id's shape. That id was already shared by both orgs.

Registering at every start also covers ids that servers of an earlier
version write during an upgrade.

### One service

`SemanticService` serves the operations for an org and project and also owns
the set-keyed work beneath them: ingestion, each set's embedder, language
model and categories, and search across sets. Ownership is checked in one
place, at the operations; the set-keyed work below receives only sets that
already passed.

## Alternatives considered

- **Check owners by parsing ids on each request.** The format is ambiguous,
  and even an exact parse cannot tell apart owners that were given the same
  id. The format would also stay the only record of ownership.
- **Encode owners unambiguously in ids**, for example as escaped
  hierarchical names. Every existing id would need rewriting in every store
  that holds one: features, history, configuration and vector records. Ids
  would carry tag values such as producer ids, and the owner check would
  still be a parse.
- **Record owners beside the existing ids**, keeping how ids are made. That
  needs the same table, registration and checks while keeping ids that
  collide. Letting the first writer claim an id refuses the second owner
  instead of separating the two.
- **Restrict org and project ids** so that the old format becomes
  unambiguous. The rules are intricate, and they reject ids already in use.

## Consequences

- Set ids are opaque. The same org or project, scope level and metadata
  always give the same set id.
- Listing sets returns the sets that exist.
- A legacy id that two owners shared is reached by neither.
- Registration reads every stored set id at each start.
