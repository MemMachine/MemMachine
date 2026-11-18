"""
Diagnostics for the server's asyncio event loop.

The loop's lag is reported as a metric. For debugging deadlocks and
long-running tasks, on Unix, send SIGUSR1 to the server process to dump all
asyncio task stacks and thread stacks to the log output:

    kill -SIGUSR1 <pid>

On all platforms, task and thread stacks are dumped automatically on shutdown.
"""

import asyncio
import logging
import signal
import sys
import traceback
from types import CoroutineType, FrameType

from memmachine_server.common.metrics_factory import MetricsFactory

logger = logging.getLogger(__name__)

_EVENT_LOOP_LAG_SAMPLE_INTERVAL_SECONDS = 0.1


def _format_task(task: asyncio.Task[object]) -> str:
    """Format a single task: header, stack frames, and what it's awaiting."""
    state = "cancelled" if task.cancelled() else ("done" if task.done() else "pending")
    parts = [f"Task {task.get_name()!r} [{state}]\n"]

    for frame in task.get_stack():
        parts.extend(traceback.format_stack(frame))

    # Walk the cr_await chain to find what the task is ultimately blocked on.
    obj = task.get_coro()
    while obj is not None:
        awaiting = getattr(obj, "cr_await", None)
        if awaiting is None:
            break
        if isinstance(awaiting, CoroutineType):
            frame: FrameType | None = awaiting.cr_frame
            if frame is not None:
                parts.extend(traceback.format_stack(frame))
            obj = awaiting
        else:
            parts.append(f"  -> awaiting: {awaiting!r}\n")
            break

    if len(parts) == 1:
        parts.append("  (no stack available)\n")

    return "".join(parts).rstrip()


def _format_thread(thread_id: int, frame: FrameType) -> str:
    """Format a single thread: header and stack frames."""
    parts = [f"Thread {thread_id:#x}:\n"]
    parts.extend(traceback.format_stack(frame))
    return "".join(parts).rstrip()


def dump_traceback() -> None:
    """Dump all asyncio task stacks and thread stacks to the logger."""
    try:
        loop = asyncio.get_running_loop()
        tasks = asyncio.all_tasks(loop)
    except RuntimeError:
        tasks = set()

    sections: list[str] = []

    # Async tasks.
    task_blocks = [
        _format_task(task) for task in sorted(tasks, key=lambda t: t.get_name())
    ]
    sections.append(f"=== Async task dump: {len(tasks)} task(s) ===")
    sections.append("\n\n".join(task_blocks))

    # Thread stacks.
    # _current_frames() is the only API for thread stacks; stable since Python 2.5
    thread_frames = sys._current_frames()  # noqa: SLF001
    thread_blocks = [_format_thread(tid, frame) for tid, frame in thread_frames.items()]
    sections.append(f"=== Thread stacks: {len(thread_frames)} thread(s) ===")
    sections.append("\n\n".join(thread_blocks))

    output = "\n\n".join(sections)
    logger.warning("\n%s", output)


def install_sigusr1_handler() -> None:
    """Register SIGUSR1 to dump traceback. No-op on Windows."""
    if not hasattr(signal, "SIGUSR1"):
        return
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, dump_traceback)


async def report_event_loop_lag(metrics_factory: MetricsFactory) -> None:
    """Report, until cancelled, how late the running loop runs a timed callback.

    Each sample schedules a wakeup _EVENT_LOOP_LAG_SAMPLE_INTERVAL_SECONDS out
    and observes how long past that time the loop took to run it.
    """
    lag = metrics_factory.get_histogram(
        "event_loop_lag_seconds",
        "Time the event loop took to run a timed callback past its due time",
    )
    loop = asyncio.get_running_loop()
    while True:
        due = loop.time() + _EVENT_LOOP_LAG_SAMPLE_INTERVAL_SECONDS
        await asyncio.sleep(_EVENT_LOOP_LAG_SAMPLE_INTERVAL_SECONDS)
        # The loop may run a timer up to its clock resolution early, and a
        # negative observation would shrink the histogram's sum.
        lag.observe(max(0.0, loop.time() - due))
