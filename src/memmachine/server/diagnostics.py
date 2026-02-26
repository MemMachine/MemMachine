"""
Diagnostics for debugging asyncio deadlocks and long-running tasks.

On Unix, send SIGUSR1 to the server process to dump all asyncio task stacks
and thread stacks to the log output:

    kill -SIGUSR1 <pid>

On all platforms, task and thread stacks are dumped automatically on shutdown.
"""

import asyncio
import logging
import signal
import sys
import traceback
from types import CoroutineType, FrameType

logger = logging.getLogger(__name__)


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
    loop = asyncio.get_running_loop()
    tasks = asyncio.all_tasks(loop)

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
