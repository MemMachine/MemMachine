"""A partition manager: one partition namespace's partitions, created and purged across every store that holds them."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Iterable, Sequence
from uuid import UUID

from memmachine_server.common.metrics_factory import OperationTracker

from .data_types import ListedPartition, TombstoneClaimStaleError
from .partition_registry import PartitionRegistry, Registration
from .partitioned_store import PartitionedStore

logger = logging.getLogger(__name__)


class PartitionManager:
    """
    The partitions of one partition namespace, in every store that holds them.

    The registry names the namespace's partitions and mints their UUIDs, and
    every store holds a partition's records under its UUID. Each step of a
    partition's lifecycle runs on every store at once: every store finishes,
    each error is logged, and the first failing store's error is raised, so
    the stores that answer keep their progress while another is down.

    Every partition manager of a namespace, in any process, must be given the
    same stores, since any of the managers may run the namespace's purge
    rounds, and a round purges only the stores its manager was given.
    Processes that share partitions must use the same namespace, since the
    same name in two namespaces is two separate partitions.
    """

    def __init__(
        self,
        partition_registry: PartitionRegistry,
        partitioned_stores: Iterable[PartitionedStore],
        *,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with the registry of the namespace, the stores that hold its partitions, and the tracker that times its operations."""
        self._partition_registry = partition_registry
        self._partitioned_stores = tuple(partitioned_stores)
        self._tracker = tracker

    async def create_partition(self, partition_name: str) -> None:
        """
        Create a partition under a name: reserve the name, create the partition in every store, and confirm the reservation.

        The partition is pending, holding its name, until its reservation is
        confirmed. A creation that raises or is canceled cancels its
        reservation, freeing the name, and the partition's purge rounds
        reclaim whatever the stores created; a confirmation that committed
        before its failure or cancellation was observed stands. A partition
        whose reservation cannot be canceled stays pending until it is
        deleted.

        Args:
            partition_name (str): The name of the partition.

        Raises:
            ValueError:
                If the partition name is invalid.
            PartitionAlreadyExistsError:
                If the name is taken, by a live or a pending partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free name failed.
            PartitionDeletedError:
                If the partition was deleted while the stores created it.
            Exception:
                Whatever a store raised while creating the partition.
        """
        async with self._tracker("create_partition"):
            reservation = await self._partition_registry.reserve(partition_name)
            try:
                await _on_every_store(
                    "create",
                    reservation.partition_uuid,
                    [
                        partitioned_store.create_partition(reservation.partition_uuid)
                        for partitioned_store in self._partitioned_stores
                    ],
                )
                await reservation.confirm()
            except BaseException:
                # The cancel logs an unexpected failure; the creation's error is
                # raised.
                with contextlib.suppress(Exception):
                    await reservation.cancel()
                raise

    async def resolve_partition(self, partition_name: str) -> Registration | None:
        """
        Resolve a partition name to the live partition registered under it.

        One read of the registry decides the outcome. The partition's records
        are under the registration's UUID in every store.

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
        async with self._tracker("resolve_partition"):
            return await self._partition_registry.resolve(partition_name)

    @property
    def max_list_limit(self) -> int:
        """The most partitions one `list_partitions` call may ask for."""
        return self._partition_registry.max_list_limit

    async def list_partitions(
        self,
        *,
        prefix: str = "",
        after: str | None = None,
        limit: int,
    ) -> list[ListedPartition]:
        """
        List the namespace's partitions, pending and live, whose names start with a prefix.

        Partitions are listed in ascending order of their names' UTF-8 bytes,
        from the first name after `after`. One read of the registry decides
        the outcome.

        A listing continues from the last name a call returned, and has ended
        once a call returns fewer partitions than its limit. A partition
        registered throughout a listing is listed once; one registered or
        deleted between two of its calls may or may not be listed.

        Args:
            prefix (str):
                What a listed partition's name starts with. The empty prefix
                lists every partition (default: "").
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
                If the prefix or `after` is invalid, or the limit is not
                between 1 and `max_list_limit`.
        """
        async with self._tracker("list_partitions"):
            return await self._partition_registry.list_partitions(
                prefix=prefix, after=after, limit=limit
            )

    async def delete_partition(self, partition_name: str) -> None:
        """
        Delete the partition registered under a name, pending or live.

        The partition is unreachable when this returns, and purge rounds
        reclaim its records from every store once the retention has passed.
        Idempotent.

        Args:
            partition_name (str): The name of the partition.

        Raises:
            ValueError:
                If the partition name is invalid.
        """
        async with self._tracker("delete_partition"):
            await self._partition_registry.unregister(partition_name)

    async def run_purge_round(self) -> bool:
        """
        Run one purge round on the claimable tombstone that came due first.

        The round claims the tombstone, purges the deleted partition from every
        store and, once no store's purge finds records, drops it from every
        store, then records on the claim whether the purge made progress or is
        complete. A round that raises fails the claim and raises. An end of a
        claim that has gone stale is logged, since a later claim decides the
        tombstone. A canceled round leaves its claim open: its tombstone is
        claimable again once the claim's lease, and then the backoff, have
        passed.

        Returns:
            bool:
                Whether a round ran: False when no tombstone is claimable.
        """
        async with self._tracker("run_purge_round"):
            claim = await self._partition_registry.claim_tombstone()
            if claim is None:
                return False
            try:
                found_records = await self._purge_partition(claim.partition_uuid)
            except Exception as error:
                await _end_claim(claim.fail(error))
                raise
            await _end_claim(claim.progress() if found_records else claim.complete())
            return True

    async def _purge_partition(self, partition_uuid: UUID) -> bool:
        """Purge a deleted partition from every store and, once no store's purge finds records, drop it from every store; return whether any store found records."""
        found = await _on_every_store(
            "purge",
            partition_uuid,
            [
                partitioned_store.purge_partition(partition_uuid)
                for partitioned_store in self._partitioned_stores
            ],
        )
        if any(found):
            return True
        await _on_every_store(
            "drop",
            partition_uuid,
            [
                partitioned_store.drop_partition(partition_uuid)
                for partitioned_store in self._partitioned_stores
            ],
        )
        return False


async def _end_claim(end: Awaitable[None]) -> None:
    """Await the end of a tombstone claim, logging the end of a stale claim instead of raising."""
    try:
        await end
    except TombstoneClaimStaleError as stale:
        logger.warning("%s; the purge round ended no claim", stale)


async def _on_every_store[T](
    step: str, partition_uuid: UUID, calls: Sequence[Awaitable[T]]
) -> list[T]:
    """Run one step of a partition's lifecycle on every store at once; once every store has finished, log each error and raise the first."""
    values: list[T] = []
    errors: list[BaseException] = []
    for result in await asyncio.gather(*calls, return_exceptions=True):
        if isinstance(result, BaseException):
            errors.append(result)
        else:
            values.append(result)
    for error in errors:
        logger.error(
            "A store failed to %s partition %s",
            step,
            partition_uuid,
            exc_info=error,
        )
    if errors:
        raise errors[0]
    return values
