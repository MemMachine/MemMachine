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
claim holds a deleted partition's tombstone until the claim is finished or
failed. Each acts on one partition alone.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from .data_types import validate_partition_name

logger = logging.getLogger(__name__)

# The tasks canceling reservations, each held until done so the garbage
# collector cannot drop a task whose caller stopped awaiting it.
_reservation_cancellations: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class Reservation(ABC):
    """
    A pending partition's hold on its name.

    The partition is pending until the reservation is confirmed. A cancel may
    follow a confirmation that raised after it committed, and may race a
    deletion by name.
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
            PartitionDeletedError:
                If the partition was deleted, or the reservation was
                confirmed already.
        """
        raise NotImplementedError

    async def cancel(self) -> None:
        """
        Cancel this reservation, unregistering its partition while it is pending and queuing the partition's UUID for purge.

        The pending partition is unreachable when this returns. Canceling
        leaves a confirmed reservation's partition live, since only a deletion
        by name ends a live partition. A partition reserved since under the
        same name is another partition and stays. Idempotent.

        The cancellation runs to completion in a shielded task of its own,
        even if the caller is canceled, and the task logs its failure, since a
        canceled caller stops awaiting it.

        Raises:
            Exception:
                Whatever `_cancel` raised. A partition whose cancellation
                committed nothing stays pending until it is deleted by name.
        """
        cancellation = asyncio.create_task(self._cancel())
        _reservation_cancellations.add(cancellation)

        def finish(task: asyncio.Task[None]) -> None:
            _reservation_cancellations.discard(task)
            if not task.cancelled() and task.exception() is not None:
                logger.error(
                    "Could not cancel the reservation of partition %r; it stays "
                    "pending until deleted",
                    self.partition_name,
                    exc_info=task.exception(),
                )

        cancellation.add_done_callback(finish)
        await asyncio.shield(cancellation)

    @abstractmethod
    async def _cancel(self) -> None:
        """
        Unregister this reservation's partition while it is pending, and queue its UUID for purge, atomically.

        Called by `cancel`, in a task of its own. Conditional on the
        partition being pending under this reservation's UUID, so a confirmed
        partition, or one reserved since under the same name, stays.
        Idempotent. A call that raises either committed both writes or
        neither.
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
    deletion. Its holder ends it once: `finish` with the round's outcome, or
    `fail` with a raised round's error. Each claim counts an attempt without
    progress on the tombstone. A claim never ended keeps its attempt, and its tombstone is claimable
    again once the claim's lease, and then the backoff, have passed, so a
    tombstone may be claimed again while an earlier round on it runs.
    """

    partition_uuid: UUID

    @abstractmethod
    async def finish(self, found_records: bool) -> None:
        """
        Record the round's outcome.

        For a round that found none of the partition's records left, the
        registry deletes the tombstone under any claim, freeing the
        partition's UUID. For a round that found records, the tombstone stays
        due; if the claim is the latest on the tombstone, the claim ends and
        the tombstone's attempts without progress reset.

        Args:
            found_records (bool):
                Whether the round found any of the partition's records.
        """
        raise NotImplementedError

    @abstractmethod
    async def fail(self, error: Exception) -> None:
        """
        End the claim of a round that raised, if the claim is the latest on its tombstone.

        The tombstone is claimable again after a backoff that grows with each
        attempt without progress. A round that raised on the registry's last
        attempt is reported, since claims skip its tombstone from then on. An
        implementation may note the round's attempt on the error.

        Args:
            error (Exception):
                What the round raised.
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
    async def startup(self) -> None:
        """
        Make the registry's durable resources ready for use, creating them idempotently.

        One call serves every registry object over the same resources.
        """
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
