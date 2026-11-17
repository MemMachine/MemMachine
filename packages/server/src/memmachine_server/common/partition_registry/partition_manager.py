"""A partition manager: one partition namespace's partitions, created and purged across every store that holds them."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Sequence
from uuid import UUID

from .partition_registry import PartitionRegistry, Registration
from .partitioned_store import PartitionedStore

logger = logging.getLogger(__name__)


class PartitionManager:
    """
    The partitions of one partition namespace, in every store that holds them.

    The registry names the namespace's partitions and mints their UUIDs, and
    every store holds a partition's records under its UUID. Creating a
    partition creates it in every store between reserving its name and
    confirming it. A purge round purges a deleted partition from every store
    and, once no store's purge finds records, drops it from every store. Each
    step runs on every store at once: every store finishes, each error is
    logged, and the first failing store's error is raised, so the stores that
    answer keep their progress while another is down.

    Every partition manager of a namespace, in any process, must be given the
    same stores. Any of them may run the namespace's purge rounds, and a round
    removes a deleted partition's records only from the stores it was given,
    so records in a store that the others lack would never be removed.
    Processes that should share partitions must use the same namespace: the
    same name in two namespaces is two separate partitions.
    """

    def __init__(
        self,
        partition_registry: PartitionRegistry,
        partitioned_stores: Sequence[PartitionedStore],
    ) -> None:
        """Initialize with the registry of the namespace and the stores that hold its partitions."""
        self._partition_registry = partition_registry
        self._partitioned_stores = tuple(partitioned_stores)

    async def create_partition(self, partition_name: str) -> None:
        """
        Create a partition under a name: reserve the name, create the partition in every store, and confirm the reservation.

        The partition is pending, holding its name, until it is confirmed. A
        creation that raises or is cancelled cancels its reservation, so the
        name is freed and the partition's purge rounds reclaim whatever the
        stores created. The cancel acts only on a pending partition, so a
        confirmation that committed before its failure or cancellation was
        observed stands. A partition whose reservation cannot be cancelled
        stays pending until it is deleted.

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
            # The cancel logs its own failure; the creation's error is raised.
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
        return await self._partition_registry.resolve(partition_name)

    async def delete_partition(self, partition_name: str) -> None:
        """
        Delete the partition registered under a name, pending or live. Idempotent.

        The partition is unreachable when this returns, and purge rounds
        reclaim its records from every store once the retention has passed.

        Args:
            partition_name (str): The name of the partition.

        Raises:
            ValueError:
                If the partition name is invalid.
        """
        await self._partition_registry.unregister(partition_name)

    async def run_purge_round(self) -> bool:
        """
        Run one purge round on the tombstone that came due first, and return whether it ran one.

        The round claims the tombstone, purges the deleted partition from
        every store and, once no store's purge finds records, drops it from
        every store, then finishes the claim with whether any store found
        records. A round that raises fails the claim and raises; a cancelled
        round releases the claim.

        Returns:
            bool:
                Whether a round ran: False when no tombstone is claimable.
        """
        claim = await self._partition_registry.claim_tombstone()
        if claim is None:
            return False
        try:
            found_records = await self._purge_partition(claim.partition_uuid)
        except Exception as error:
            await claim.fail(error)
            raise
        except BaseException:
            # The release logs its own failure; the cancellation is raised.
            with contextlib.suppress(Exception):
                await claim.release()
            raise
        await claim.finish(found_records)
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


async def _on_every_store[T](
    step: str, partition_uuid: UUID, calls: Sequence[Awaitable[T]]
) -> list[T]:
    """Run one step of a partition's lifecycle on every store at once, and raise the first error once every store has finished."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    for error in errors:
        logger.error(
            "A store failed to %s partition %s",
            step,
            partition_uuid,
            exc_info=error,
        )
    if errors:
        raise errors[0]
    return [result for result in results if not isinstance(result, BaseException)]
