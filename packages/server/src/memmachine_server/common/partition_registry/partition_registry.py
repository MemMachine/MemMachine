"""
Abstract base classes for a partition registry, its reservations, registrations, and tombstone claims.

A partition registry is the catalog of one partition namespace. A partition
has a name, free for a later partition once the partition is deleted, and a
UUID, which the registry mints for it alone. The registry records each
partition's name, its UUID, and whether it is pending or live. Its calls are
arbitrated across every process sharing it: a reservation mints a UUID no
registered or queued partition carries, a deletion makes the partition
unreachable when it returns, and a tombstone claim holds a deleted
partition's tombstone under a lease.

A reservation holds a new partition's name until the reservation is
confirmed or canceled. Resolving a name answers a `Registration`, whose
`require_current` checks that the partition is still registered. A tombstone
claim holds a deleted partition's tombstone until its holder ends it with
`complete`, `progress`, or `fail`. Each acts on one partition alone.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from .data_types import (
    ListedPartition,
    PartitionAlreadyConfirmedError,
    PartitionDeletedError,
    validate_partition_name,
)

logger = logging.getLogger(__name__)

# The tasks canceling reservations, each held until done so the garbage
# collector cannot drop a task whose caller stopped awaiting it.
_reservation_cancellations: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class Reservation(ABC):
    """
    A pending partition's hold on its name.

    The partition is pending until the reservation is confirmed. `confirm`
    and `cancel` act only on a pending partition. A cancel may follow a
    confirmation that raised after it committed, and may race a deletion by
    name.
    """

    partition_name: str
    partition_uuid: UUID

    @abstractmethod
    async def confirm(self) -> None:
        """
        Confirm this reservation, marking the pending partition live.

        A concurrent deletion of the partition either follows the
        confirmation or makes this call raise.

        Raises:
            PartitionAlreadyConfirmedError:
                If the reservation was confirmed already.
            PartitionDeletedError:
                If the partition was deleted.
        """
        raise NotImplementedError

    async def cancel(self) -> None:
        """
        Cancel this reservation, unregistering its pending partition and queuing the partition's UUID for purge.

        The partition is unreachable when this returns. A partition reserved
        since under the same name is another partition and stays.

        The cancellation runs to completion in a shielded task of its own,
        even if the caller is canceled, and the task logs any failure other
        than PartitionAlreadyConfirmedError and PartitionDeletedError, since a
        canceled caller stops awaiting it.

        Raises:
            PartitionAlreadyConfirmedError:
                If the reservation was confirmed, since only a deletion by
                name ends a live partition.
            PartitionDeletedError:
                If the partition was canceled already or deleted by name.
            Exception:
                Whatever else `_cancel` raised. A partition whose cancellation
                committed nothing stays pending until it is deleted by name.
        """
        cancellation = asyncio.create_task(self._cancel())
        _reservation_cancellations.add(cancellation)

        def finish(task: asyncio.Task[None]) -> None:
            _reservation_cancellations.discard(task)
            if task.cancelled():
                return
            error = task.exception()
            if error is not None and not isinstance(
                error, (PartitionAlreadyConfirmedError, PartitionDeletedError)
            ):
                logger.error(
                    "Could not cancel the reservation of partition %r; it stays "
                    "pending until deleted",
                    self.partition_name,
                    exc_info=error,
                )

        cancellation.add_done_callback(finish)
        await asyncio.shield(cancellation)

    @abstractmethod
    async def _cancel(self) -> None:
        """
        Unregister this reservation's partition while it is pending, and queue its UUID for purge, atomically.

        Called by `cancel`, in a task of its own. Conditional on the
        partition being pending under this reservation's UUID, so a confirmed
        partition, or one reserved since under the same name, stays. A call
        that fails for a reason not listed below either committed both writes
        or neither.

        Raises:
            PartitionAlreadyConfirmedError:
                If the reservation's partition is live.
            PartitionDeletedError:
                If no partition holds the reservation's UUID.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Registration(ABC):
    """
    A live partition's registration.

    Its fields are fixed as the partition was resolved; `require_current`
    checks that the partition is still registered.
    """

    partition_name: str
    partition_uuid: UUID

    @abstractmethod
    async def require_current(self) -> None:
        """
        Raise unless this partition is still the one registered under its name.

        One read of the registry decides the outcome, so the call raises for a
        deletion committed before the read and returns despite a deletion
        committed after it.

        Raises:
            PartitionDeletedError:
                If the partition was deleted, whether or not another was
                registered under the name since.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class TombstoneClaim(ABC):
    """
    A purge round's claim on a deleted partition's tombstone, held under a lease.

    The tombstone is due: the retention has passed since the partition's
    deletion. Its holder ends it with `complete` when the partition's purge is
    complete, `progress` when the round purged records, or `fail` with a
    raised round's error. An end takes effect only while the claim is the
    latest on its tombstone and has not ended; any other end changes nothing.
    Each claim counts an attempt without progress on the tombstone. A claim
    never ended keeps its attempt, and its tombstone is claimable again once
    the claim's lease, and then the backoff, have passed, so a tombstone may
    be claimed again while an earlier round on it runs.
    """

    partition_uuid: UUID

    @abstractmethod
    async def complete(self) -> None:
        """
        Record that the partition's purge is complete.

        The registry deletes the tombstone, freeing the partition's UUID.

        Raises:
            TombstoneClaimStaleError:
                If the claim is not the tombstone's latest, unended claim.
        """
        raise NotImplementedError

    @abstractmethod
    async def progress(self) -> None:
        """
        Record that the round purged some of the partition's records.

        The tombstone stays due and is claimable at once, with its attempts
        without progress reset.

        Raises:
            TombstoneClaimStaleError:
                If the claim is not the tombstone's latest, unended claim.
        """
        raise NotImplementedError

    @abstractmethod
    async def fail(self, error: Exception) -> None:
        """
        Record that the round raised.

        The tombstone is claimable again after a backoff that grows with each
        attempt without progress. A round that raised on the registry's last
        attempt is reported, since claims skip its tombstone from then on. An
        implementation may note the round's attempt on the error.

        Args:
            error (Exception):
                What the round raised.

        Raises:
            TombstoneClaimStaleError:
                If the claim is not the tombstone's latest, unended claim.
        """
        raise NotImplementedError


class PartitionRegistry(ABC):
    """
    The partition registry of one partition namespace.

    A deleted partition's UUID waits on a queue as a tombstone, which comes
    due once the registry's retention has passed since the deletion. The
    registry deletes the tombstone, freeing its UUID, once a purge round
    finds none of the partition's records left.
    """

    @abstractmethod
    async def prepare(self) -> None:
        """
        Create the registry's durable resources, starting from none.

        The caller runs it before `startup`, while no other process uses the
        registry. Run again on what it created, it changes nothing. One call
        serves every registry object over the same resources.
        """
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """Start the registry on the durable resources `prepare` created."""
        raise NotImplementedError

    async def reserve(self, partition_name: str) -> Reservation:
        """
        Reserve a partition name for a new pending partition under a freshly minted UUID.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Reservation: The new partition's reservation.

        Raises:
            ValueError:
                If the partition name is invalid.
            PartitionAlreadyExistsError:
                If the name is taken, by a live or a pending partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free name failed.
        """
        validate_partition_name(partition_name)
        return await self._reserve(partition_name)

    @abstractmethod
    async def _reserve(self, partition_name: str) -> Reservation:
        """
        Reserve a partition name for a new pending partition under a freshly minted UUID.

        Called by `reserve` with a valid name, concurrently with any other
        call on any process. The partition name is arbitrated across
        processes, and the UUID is one no registered or queued partition
        carries.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Reservation: The new partition's reservation.

        Raises:
            PartitionAlreadyExistsError:
                If the partition name is taken, by a live or a pending
                partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free partition name failed.
        """
        raise NotImplementedError

    async def resolve(self, partition_name: str) -> Registration | None:
        """
        Resolve a partition name to the live partition registered under it.

        One read of the registry decides the outcome.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Registration | None:
                The live partition's registration, or None when no partition
                holds the name.

        Raises:
            ValueError:
                If the partition name is invalid.
            PartitionPendingError:
                If the partition registered under the name is pending.
        """
        validate_partition_name(partition_name)
        partition_uuid = await self._resolve(partition_name)
        if partition_uuid is None:
            return None
        return self._registration(partition_name, partition_uuid)

    @abstractmethod
    async def _resolve(self, partition_name: str) -> UUID | None:
        """
        Resolve a partition name to the UUID of the live partition registered under it, or None when no partition holds the name.

        Called by `resolve` with a valid name, concurrently with any other
        call on any process. One read of the registry decides the outcome,
        reflecting every reservation, confirmation, and deletion committed
        before the read. It only reads, so a call that raises changes nothing.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            UUID | None:
                The live partition's UUID, or None when no partition holds
                the name.

        Raises:
            PartitionPendingError:
                If the partition registered under the name is pending.
        """
        raise NotImplementedError

    @abstractmethod
    def _registration(self, partition_name: str, partition_uuid: UUID) -> Registration:
        """
        Build the registration of a live partition.

        Called by `resolve` with the UUID `_resolve` returned. Reads and
        writes nothing, so the registration's `require_current` reports a
        partition deleted since.

        Args:
            partition_name (str): The name the partition is registered under.
            partition_uuid (UUID): The partition's UUID.

        Returns:
            Registration: The partition's registration.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_list_limit(self) -> int:
        """The most partitions one `list_partitions` call may ask for."""
        raise NotImplementedError

    async def list_partitions(
        self,
        *,
        prefix: str = "",
        after: str | None = None,
        limit: int,
    ) -> list[ListedPartition]:
        """
        List the registered partitions, pending and live, whose names start with a prefix.

        Partitions are listed in ascending order of their names' UTF-8 bytes,
        from the first name after `after`. One read of the registry decides
        the outcome.

        A listing continues from the last name a call returned, and has ended
        once a call returns fewer partitions than its limit. A partition
        registered throughout a listing is listed once; one registered or
        deleted between two of its calls may or may not be listed.

        Args:
            prefix (str):
                What a listed partition's name starts with: a string valid
                as a partition name, or the empty string, which lists every
                partition (default: "").
            after (str | None):
                The name to list after, which need not be registered. None
                lists from the first name with the prefix (default: None).
            limit (int):
                The most partitions to return, at most `max_list_limit`.

        Returns:
            list[ListedPartition]:
                The partitions, in ascending order of their names' bytes.

        Raises:
            ValueError:
                If the prefix is neither empty nor valid as a partition
                name, `after` is invalid as a partition name, or the limit
                is not between 1 and `max_list_limit`.
        """
        # Every non-empty start of a partition name is a valid partition name.
        if prefix:
            validate_partition_name(prefix)
        if after is not None:
            validate_partition_name(after)
        if not 1 <= limit <= self.max_list_limit:
            raise ValueError(
                f"Limit {limit} must be between 1 and {self.max_list_limit}"
            )
        return await self._list_partitions(prefix, after, limit)

    @abstractmethod
    async def _list_partitions(
        self, prefix: str, after: str | None, limit: int
    ) -> list[ListedPartition]:
        """
        List the registered partitions, pending and live, whose names start with a prefix.

        Called by `list_partitions` with a prefix that is empty or valid as
        a partition name, an `after` that is None or valid as a partition
        name, and a limit between 1 and `max_list_limit`, concurrently with
        any other call on any process. One read of the registry decides the outcome,
        reflecting every reservation, confirmation, and deletion committed
        before the read. Names are compared and ordered by their UTF-8
        bytes. It only reads, so a call that raises changes nothing.

        Args:
            prefix (str):
                What a listed partition's name starts with.
            after (str | None):
                The name to list after, or None to list from the first name
                with the prefix.
            limit (int):
                The most partitions to return.

        Returns:
            list[ListedPartition]:
                The first `limit` partitions after `after` whose names start
                with the prefix, in ascending order of their names' bytes.
        """
        raise NotImplementedError

    async def unregister(self, partition_name: str) -> None:
        """
        Delete the partition registered under a name, pending or live, and queue its UUID for purge.

        The partition is unreachable when this returns: confirming its
        reservation and calling its registrations' `require_current` raise
        PartitionDeletedError. Idempotent.

        Args:
            partition_name (str): The name of the partition.

        Raises:
            ValueError:
                If the partition name is invalid.
        """
        validate_partition_name(partition_name)
        await self._unregister(partition_name)

    @abstractmethod
    async def _unregister(self, partition_name: str) -> None:
        """
        Unregister the partition under a name, pending or live, and queue its UUID for purge, atomically.

        Called by `unregister` with a valid name, concurrently with any other
        call on any process. The tombstone comes due once the retention has
        passed since the deletion. Idempotent: racing deletions of one
        partition queue one tombstone, and a name no partition holds is left
        as it is. A call that raises either committed both writes or neither.

        Args:
            partition_name (str): The name of the partition.
        """
        raise NotImplementedError

    @abstractmethod
    async def claim_tombstone(self) -> TombstoneClaim | None:
        """
        Claim the claimable tombstone that came due first.

        A tombstone is due once the retention has passed since its partition's
        deletion. A due tombstone is claimable unless it has had the
        registry's maximum of attempts without progress, or it waits out its
        latest claim's lease and then the backoff, or the backoff after its
        latest round raised. Safe to call from several processes at once.

        Returns:
            TombstoneClaim | None:
                The claim, or None when no tombstone is claimable.
        """
        raise NotImplementedError
