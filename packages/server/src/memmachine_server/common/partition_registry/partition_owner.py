"""
What the owner of a partition namespace does through its registry.

The owner is the memory system whose stores hold a partition's records. It
reserves the partition's name, which mints the incarnation and leaves the
partition pending, prepares every store's storage under the incarnation,
and confirms the reservation, which makes the partition live. Only a live
partition is opened.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

from .data_types import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionIncarnationEndedError,
    PartitionPendingError,
    validate_partition_name,
)
from .partition_registry import PartitionRegistry, Registration, Reservation

logger = logging.getLogger(__name__)

# Attempts open-or-create makes, _OPEN_OR_CREATE_RETRY_DELAY_SECONDS apart,
# before it gives up on a partition that stays pending or a name it keeps
# losing.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10
_OPEN_OR_CREATE_RETRY_DELAY_SECONDS = 1

# Reservations cancelled after a failed preparation, held until done so the
# garbage collector cannot drop one whose creation was cancelled.
_reservation_cancellations: set[asyncio.Task[None]] = set()


async def open_or_create_partition(
    partition_registry: PartitionRegistry,
    partition_name: str,
    prepare_partition_incarnation: Callable[[UUID], Awaitable[None]],
) -> Registration:
    """
    Answer the registration of the live partition under a name, creating the partition if no partition holds the name.

    Read-then-create, retried a second apart: a live partition is answered;
    a pending one is another creator's, so the call waits for it; no
    partition means create; losing the reservation means another creator
    took the name meanwhile; and losing the confirmation means a deleter
    removed the partition while its storage was prepared, so the call
    creates again.

    Args:
        partition_registry (PartitionRegistry):
            The registry of the owner's partition namespace.
        partition_name (str):
            The name of the partition.
        prepare_partition_incarnation (Callable[[UUID], Awaitable[None]]):
            Prepares every store's storage for the incarnation of a
            partition this call reserved, before the call confirms it. It
            returns once every store's preparation has finished, and raises
            if any failed; the reservation is then cancelled, and the
            cancelled incarnation's storage is purged like a deleted one's.

    Returns:
        Registration:
            The live partition's registration.

    Raises:
        ValueError:
            If the partition name breaks the registry's rule.
        PartitionPendingError:
            If the partition's creation, by another caller, did not complete
            within the attempts to open it.
        PartitionCreateAttemptsExhaustedError:
            If every attempt lost the name to another creator or deleter.
        Exception:
            Whatever `prepare_partition_incarnation` raises.
    """
    validate_partition_name(partition_name)
    pending_error: PartitionPendingError | None = None
    lost_race: PartitionAlreadyExistsError | PartitionIncarnationEndedError | None = (
        None
    )
    for attempt in range(_MAX_OPEN_OR_CREATE_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_OPEN_OR_CREATE_RETRY_DELAY_SECONDS)
        try:
            registration = await partition_registry.resolve(partition_name)
        except PartitionPendingError as err:
            pending_error = err
            continue
        pending_error = None
        if registration is not None:
            return registration
        try:
            reservation = await partition_registry.reserve(partition_name)
        except PartitionAlreadyExistsError as err:
            lost_race = err
            continue
        try:
            return await _prepare_and_confirm_or_cancel(
                reservation, prepare_partition_incarnation
            )
        except PartitionIncarnationEndedError as err:
            lost_race = err
            continue
    # The last lookup found the partition pending.
    if pending_error is not None:
        raise pending_error
    raise PartitionCreateAttemptsExhaustedError(
        f"Opening or creating partition {partition_name!r} made no progress "
        f"after {_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
    ) from lost_race


async def _prepare_and_confirm_or_cancel(
    reservation: Reservation,
    prepare_partition_incarnation: Callable[[UUID], Awaitable[None]],
) -> Registration:
    """Prepare a reserved partition's storage and confirm the reservation, cancelling it if either raises or the creation is cancelled.

    The cancel acts only on a pending partition, so a confirmation that
    committed before its failure or cancellation was observed stands. A
    partition whose reservation the registry cannot cancel stays pending
    until it is deleted.
    """
    try:
        await prepare_partition_incarnation(reservation.incarnation)
        return await reservation.confirm()
    except BaseException:
        # Shielded, so a cancelled creation still frees the name. The task
        # reports its own failure, since a creation cancelled again stops
        # awaiting it before it ends.
        cancellation = asyncio.create_task(reservation.cancel())
        _reservation_cancellations.add(cancellation)

        def finish(task: asyncio.Task[None]) -> None:
            _reservation_cancellations.discard(task)
            if not task.cancelled() and task.exception() is not None:
                logger.exception(
                    "Could not cancel the reservation of partition %r after its "
                    "creation failed or was cancelled; it stays pending until "
                    "deleted",
                    reservation.partition_name,
                    exc_info=task.exception(),
                )

        cancellation.add_done_callback(finish)
        with contextlib.suppress(Exception):
            await asyncio.shield(cancellation)
        raise
