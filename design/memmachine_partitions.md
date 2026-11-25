# MemMachine partitions: one identity for everything a project owns

MemMachine keeps each project of the v2 API as a partition of its own
partition namespace. A partition is an isolated memory with its own settings
and lifecycle: it holds its episodes and the memories built from them. It is
not a conversation: one partition holds every conversation an application
writes into it. Its name is the API's `org_id/project_id`. What the API and
its documentation call a partition is open (#1812); this document uses the
registry's word at every layer.

## Scope

This design changes two things: how a partition is identified, and how its
creation and deletion are sequenced. Its one aim is that a creation or a
deletion that stops part way is never visible, with as little change to the
server as that takes.

Every component is left as it is: the session data manager, the episode
store, the episodic memory manager with its instance cache, short-term and
long-term memory, and semantic memory keep their schemas, their interfaces,
and their division of work. They are kept because the aim does not need them
changed, not because their design is settled, and nothing here depends on
how a component holds a partition's data beyond the one key it takes.
[What this design leaves open](#what-this-design-leaves-open) lists what
that excludes.

## Problem

A partition's data is spread over several components: its record, its
episodes, short-term memory's summary, and a long-term memory in event
memory or in declarative memory. When each component holds the data under
the partition's name, a creation or a deletion that stops part way is
visible, whatever sequences its steps:

- **A partition being deleted is a third state.** Its deletion is a mark on
  its record and then steps that empty each component, so it is neither
  present nor absent until the last step. Every request has to answer for
  it, and a creation under its name waits, is refused, or fails for as long
  as any step keeps failing.
- **A lease on the deletion does not confine it.** A lease keeps two
  processes from running the steps at once. The steps delete by name, and
  no component checks the lease, so a deleter that outlives its lease
  deletes whatever holds the name by then.
- **A name created again inherits what a deletion left behind.** Data stored
  under the name, in any component the deletion did not reach, belongs to
  whichever partition holds the name.
- **A late write lands in the next partition.** A request that read its
  partition before the deletion and writes after it writes under the name.
- **A creation that stops leaves a partition every request fails on.** Its
  record says it exists, and its long-term storage does not.

## Design

MemMachine owns the `memmachine` partition namespace, in a
[partition registry](partition_registry.md). The registry mints a UUID for
each partition created under a name. MemMachine resolves a request's name to
its partition's UUID, and gives every component the UUID's string as the key
the component holds the partition's data under. The components are
unchanged: each already holds a partition's data under one opaque key.

Creating and deleting a partition are the registry's transactions. A
partition is answered only once its reservation is confirmed, and it is
unreachable, with its name free, when its deletion commits. A partition
created under the name afterwards has another UUID, so no component finds
the deleted partition's data under its key: what a deletion has yet to
remove is invisible, and a partition manager's purge rounds remove it.

### What a component holds under the key

- The session data manager holds the partition's record, with its
  description and settings, and short-term memory's summary.
- The episode store holds the partition's episodes.
- Event memory holds the event memory partition named by the key, and
  declarative memory holds the collections named by the key.
- Each process's instance cache holds the partition's episodic memory
  instance, and its episode count cache holds the partition's count, so a
  partition created again under a name gets its own.
- Semantic memory holds nothing under the key. Its sets are named by
  organization, project, and tags, and its history and citations by episode
  UUID (see [Semantic memory](#semantic-memory)).

The API's listed episodes carry the key they are stored under as
`session_key`.

### Stores

MemMachine builds its partition manager over three stores. Each is a
`PartitionedStore` over one component's existing operations by key, so no
component's schema or interface changes, and a component that later holds
partitions by UUID itself replaces its store here alone.

| Store over | Create | Purge | Drop |
|---|---|---|---|
| The session data manager | Writes the partition's record. | Finds nothing. | Deletes the record, and the summary by cascade. |
| The episode store | Nothing. | Deletes one batch of the key's episodes, after the batch's history and citations in semantic memory. | Nothing. |
| Long-term memory | Creates the event memory partition named by the key, when the deployment configures event memory. | Deletes that event memory partition by name, and the declarative nodes of a partition whose record selects declarative memory. | Nothing. |

A purge round drops a partition only after no store's purge finds anything,
so the record a purge reads is deleted last. Event memory keeps its own
namespace and registry: deleting its partition by name is one transaction
there, and its own purge rounds remove the records after its own retention.
The key is a UUID no other partition carries, so deleting that name deletes
exactly that partition's event memory.

The stores follow what the deployment configures, never whether a memory is
enabled, so a deletion reaches data written while a memory was enabled.

### Creation

The partition manager reserves the name, creates the partition in every
store, and confirms the reservation. The session data manager's store writes
the record under the key, with the partition's description and settings,
and the long-term memory store creates the event memory partition, when the
deployment configures event memory. Both happen while the partition is
pending, so every live partition has its record and its event memory
partition.

A creation under a name that a live partition holds creates nothing. It
compares its settings with the stored record's: equal settings answer the
partition, and different ones raise `SessionAlreadyExistsError`.

A creation that stops before it confirms leaves the partition pending, with
or without its record. A pending partition is not answered: getting it finds
none, and creating a partition under its name raises the registry's
`PartitionPendingError`, which carries when the partition was reserved.
Whether that creation was abandoned is its caller's decision: the caller
deletes the partition and creates it again, and the deleted partition's
purge drops whatever record it had.

An add or a search on a name that no partition holds creates the partition
the same way, with the deployment's default settings, and waits a bounded
number of times for a partition another creator holds pending.

### Requests and fencing

A request resolves its partition's name once, when it starts, and calls the
components with the key. With no partition under the name, a request finds
no session: a listing or a count answers nothing, and an update raises
`SessionNotFoundError`.

A request that writes checks the registration again after its writes, and
raises `SessionNotFoundError` when the partition was deleted meanwhile. A
write is therefore acknowledged only for a partition that was still live
after it. What the request wrote stays under the deleted partition's key,
where no request reads it, until the purge deletes it. A request that only
reads is not checked afterwards: a deleted partition's data stays until its
purge, so a read in flight when the deletion commits answers what it would
have answered a moment earlier.

### Deletion and purge

Deleting a partition resolves the name, and raises `SessionNotFoundError`
when no partition holds it. For a pending or a live partition it deletes the
features of the project's semantic sets (see
[Semantic memory](#semantic-memory)), then unregisters the name: one
transaction, after which no request resolves the partition, the name is
free, and the partition's tombstone is queued.

Once the namespace's retention has passed, the partition manager's purge
rounds purge the partition from every store until no store finds anything,
then drop it. A server runs the namespace's purge rounds from the time it
starts, so any server finishes any deletion, under the registry's leases,
backoff, and dead-lettering.

### Listing

Listing an organization's partitions is the registry's
[listing by name prefix](partition_registry.md#operations): the partitions
whose names start with the organization's id and the delimiter. A partition
is listed from the reservation of its name to its deletion, with whether it
is live, so a partition left pending is listed, as pending. No component is
read.

The registry caps how many partitions one call answers. MemMachine reads an
organization's listing a page at a time, and answers all of it.

### Semantic memory

Semantic memory names its sets by organization, project, and tags, and some
sets span every project of an organization. It takes an organization and a
project where every other component takes one key, and it keeps no record
of the sets created under a project: a request reaches a set by building
the set's id again from its own organization, project, and tags (#2004). A
purge by UUID has nothing to find a partition's sets by. So semantic memory
is not one of the stores, and its sets are shared by every partition
created under a name. A
partition's deletion deletes the features of the project's sets by name
before it unregisters the name, while the name still names the partition
being deleted: a deletion by name that ran after the name was free would
delete the sets of a partition created under the name afterwards. Features
extracted from episodes added while a deletion runs are read by the next
partition under the name.

### Retention

The registry is in the session database, and the session manager's
configuration names the namespace's retention. The retention covers every
write a request makes under its partition's registration. The event memory
partition of a deleted MemMachine partition is deleted when a purge round
reaches it, so its records are purged once the namespace's retention and
then event memory's have passed.

## Decisions

- **One UUID for everything a partition owns, minted by the registry.** A
  name created again starts empty in every component, and a late write or a
  late purge reaches only the partition it was meant for. Outside the
  registry, nothing a partition owns holds its name, so renaming a
  partition rewrites no component's records, a store's naming rules meet
  the UUID and never a caller's text, and what waits for a purge is held
  without the name.
- **The registry decides whether a partition exists.** A partition is
  pending, live, or absent in the registry, and a deleted partition is
  absent, so no request answers for a partition being deleted and a creation
  under a deleted name succeeds at once.
- **A listing is the registry's.** The registry knows every partition,
  pending ones included, and an organization is the start of its
  partitions' names, so MemMachine keeps no index of its own for listing.
- **The change is confined to MemMachine and three stores.** The UUID's
  string is the key each component already takes, and each store is a thin
  layer over the component's existing operations. A component that comes to
  hold partitions by UUID itself removes its store, and nothing else here
  changes with it.
- **A memory system keeps its own lifecycle.** MemMachine names event
  memory's partition by its own partition's UUID and deletes that name.
- **A pending partition is answered as pending.** When to give up on a
  creation is the caller's policy: the registry offers the deletion, and the
  server holds no timeout.
- **Stores follow configuration, not the enabled flags.**

## Alternatives considered

- **The partition's row as the registry**, with provisioning, active, and
  deleting states on the row and a job table for deletion. It is a second
  catalog and a second claim protocol beside the partition registry's, and
  it keeps the deleting state.
- **A deletion queue in each process**, with a mark on the partition's
  record, a lease on each deletion, and a rescan for marked partitions. The
  mark keeps the deleting state, and the name stays unusable until the last
  step. The lease's contract is that whatever writes checks its fencing
  token. Components are addressed by name and take no token, so an expired
  lease does not stop a stalled deleter from deleting the next partition
  under the name, and a token for every component to check changes each
  one. A purge addressed by UUID is safe without a lease, and the
  registry's claim on a tombstone keeps two purges apart.
- **The delete request deleting the record**, so that a listing reads the
  records. The records would be a second source for which partitions exist.
- **A listing from the records, each checked against the registry.** Every
  record would hold its partition's name and organization, a listing would
  read the registry once for each record, and a partition left pending
  before its record was written would be listed nowhere.
- **A creation replacing a partition left pending**, after a wait. How long
  a creation may take before it counts as abandoned is a policy the server
  would hold for every caller.
- **The project's semantic sets deleted in the purge round, by name.** The
  round runs after the name is free, so it would delete the sets of a
  partition created under the name afterwards.
- **The UUID as semantic memory's project.** Project-level sets would be
  under the key. Set ids are returned by the API and would change at each
  creation, and a purge still could not find them, since nothing records a
  partition's sets.

## What this design leaves open

- **Components holding partitions by UUID themselves**, with a UUID column or
  parameter in place of the session key. It changes each component's schema
  and interface, which this design avoids, and it removes the stores here
  one component at a time.
- **Semantic sets owned by a partition.** They are shared by every partition
  created under a name (see [Semantic memory](#semantic-memory)).
- **Whether a request may create a partition.** An add or a search on a
  name no partition holds creates one.
- **How the components are divided and composed**: the session data manager
  holding short-term memory's summary, the instance cache in each process,
  and MemMachine writing to each memory in turn.
- **What a pending partition answers to requests other than a creation**,
  and whether anything deletes one that no caller deletes.
- **Renaming a partition.** The registry offers no rename, and semantic
  sets, named by organization and project, would not follow one.
- **Paging a listing.** A listing answers all of an organization's
  partitions at once.
- **Creating everything a partition has before it is confirmed.**
  Declarative memory creates a partition's collections and indexes at the
  partition's first write, and semantic memory creates a set at the set's
  first write.
- **What the partition is called** in the API and its documentation (#1812).

## Consequences

- A request reads the registry once before its work, and a write reads it
  once more.
- A listing of an organization reads the registry once for each page of
  the organization's partitions, and no component.
- A listing shows a partition left pending, which getting it does not
  answer.
- A deleted partition's data is unreachable from the deletion and is
  physically deleted after the namespace's retention, and its event memory
  records after event memory's retention as well.
- A name is usable again as soon as its partition's deletion commits.
- Components store a partition's UUID where they stored its name, and the
  API's listed episodes carry it as `session_key`.
- A project whose name the registry refuses cannot be created.
- Semantic sets are shared by every partition created under a name.
