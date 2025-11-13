import asyncio
from datetime import datetime


class SemanticUpdateTracker:
    """Tracks feature update activity for a feature set.
    When a message is added to a feature set, this class keeps track of how many
    messages have been sent and when the first message was sent.
    This is used to determine when to trigger feature updates based
    on message count and time intervals.
    """

    def __init__(self, set_id: str, message_limit: int, time_limit_sec: float):
        self._set_id = set_id
        self._message_limit: int = message_limit
        self._time_limit: float = time_limit_sec
        self._message_count: int = 0
        self._first_updated: datetime | None = None

    def mark_update(self):
        """Marks that a new message has been added to the feature set.
        Increments the message count and sets the first updated time
        if this is the first message.
        """
        self._message_count += 1
        if self._first_updated is None:
            self._first_updated = datetime.now()

    def _seconds_from_first_update(self) -> float | None:
        """Returns the number of seconds since the first message was sent.
        If no messages have been sent, returns None.
        """
        if self._first_updated is None:
            return None
        delta = datetime.now() - self._first_updated
        return delta.total_seconds()

    def reset(self):
        """Resets the tracker state.
        Clears the message count and first updated time.
        """
        self._message_count = 0
        self._first_updated = None

    def should_update(self) -> bool:
        """Determines if a feature update should be triggered.
        A feature update is triggered if either the message count
        exceeds the limit or the time since the first message exceeds
        the time limit.

        Returns:
            bool: True if a feature update should be triggered, False otherwise.
        """
        if self._message_count == 0:
            return False
        elapsed = self._seconds_from_first_update()
        exceed_time_limit = elapsed is not None and elapsed >= self._time_limit
        exceed_msg_limit = self._message_count >= self._message_limit
        return exceed_time_limit or exceed_msg_limit


class SemanticUpdateTrackerManager:
    """Manages SemanticUpdateTracker instances for multiple feature sets."""

    def __init__(self, message_limit: int, time_limit_sec: float):
        self._trackers: dict[str, SemanticUpdateTracker] = {}
        self._trackers_lock = asyncio.Lock()
        self._message_limit = message_limit
        self._time_limit_sec = time_limit_sec

    def _new_tracker(self, set_id: str) -> SemanticUpdateTracker:
        return SemanticUpdateTracker(
            set_id=set_id,
            message_limit=self._message_limit,
            time_limit_sec=self._time_limit_sec,
        )

    async def mark_update(self, set_ids: list[str]):
        """Marks that a new message has been assigned to a feature set.
        Creates a new tracker if one does not exist for the feature set.
        """
        async with self._trackers_lock:
            for set_id in set_ids:
                if set_id not in self._trackers:
                    self._trackers[set_id] = self._new_tracker(set_id)
                self._trackers[set_id].mark_update()

    async def get_sets_to_update(self) -> list[str]:
        """Returns a list of sets whose features need to be updated.
        A feature update is needed if the set's tracker indicates
        that an update should be triggered.
        """
        async with self._trackers_lock:
            ret = []
            for set, tracker in self._trackers.items():
                if tracker.should_update():
                    ret.append(set)
                    tracker.reset()
            return ret
