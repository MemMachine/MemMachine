import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LONGMEMEVAL_S_PATH = Path("data/longmemeval_s_cleaned.json")
LONGMEMEVAL_M_PATH = Path("data/longmemeval_m_cleaned.json")
LONGMEMEVAL_ORACLE_PATH = Path("data/longmemeval_oracle.json")

ALL_DATA_PATHS = [LONGMEMEVAL_S_PATH, LONGMEMEVAL_M_PATH, LONGMEMEVAL_ORACLE_PATH]


# ---------- ENUMS ----------
class QuestionType(str, Enum):
    SINGLE_SESSION_USER = "single-session-user"
    SINGLE_SESSION_ASSISTANT = "single-session-assistant"
    SINGLE_SESSION_PREFERENCE = "single-session-preference"
    TEMPORAL_REASONING = "temporal-reasoning"
    KNOWLEDGE_UPDATE = "knowledge-update"
    MULTI_SESSION = "multi-session"
    UNKNOWN = "unknown"


class Turn(BaseModel):
    role: str
    content: str
    has_answer: bool = False

    # add-on
    index: int = 0
    timestamp: str | None = ""


class LongMemEvalItem(BaseModel):
    question_id: str
    question_type: QuestionType = QuestionType.UNKNOWN
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[str] = []
    haystack_dates: list[str] = []
    haystack_sessions: list[list[Turn]] = []
    answer_session_ids: list[str] = []

    # add-on
    abstention_question: bool = False
    session_id_map: dict[str, int] = {}
    turn_id_map: dict[str, dict[int, Turn]] = {}
    haystack_datetimes: list[datetime] = []
    answer_turn_indices: list[str] = []

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_answer_to_str(cls, v):
        if v is None:
            return ""
        return str(v)

    @model_validator(mode="after")
    def _post_init(self):
        # abstention
        self.abstention_question = self.question_id.endswith("_abs")

        # session ID map
        self.session_id_map = {
            sid: idx for idx, sid in enumerate(self.haystack_session_ids)
        }

        # turn maps
        self.turn_id_map = defaultdict(dict)
        for session_id in self.haystack_session_ids:
            session = self.get_session(session_id)
            session_date = self.get_session_date(session_id)
            session_datetime = get_datetime_from_timestamp(session_date)
            self.haystack_datetimes.append(session_datetime)
            for i, turn in enumerate(session):
                self.turn_id_map[session_id][i] = turn
                # assume the session starts at session_datetime and each turn is 1 second apart
                time_diff = timedelta(seconds=i)
                turn.timestamp = (session_datetime + time_diff).isoformat()
                turn.index = i
                if turn.has_answer:
                    self.answer_turn_indices.append(f"{session_id}:{turn.index}")
        return self

    def get_session(self, session_id: str) -> list[Turn]:
        idx = self.session_id_map.get(session_id)
        if idx is None or idx >= len(self.haystack_sessions):
            return []
        return self.haystack_sessions[idx]

    def get_session_date(self, session_id: str) -> str | None:
        idx = self.session_id_map.get(session_id)
        if idx is None or idx >= len(self.haystack_dates):
            return None
        return self.haystack_dates[idx]

    def get_turn(self, session_id: str, turn_index: int) -> Turn | None:
        return self.turn_id_map.get(session_id, {}).get(turn_index)


def load_longmemeval_dataset(file_path: str) -> list[LongMemEvalItem]:
    """
    Load the dataset from the given json file path and return a list of LongMemEvalItem objects
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [LongMemEvalItem(**item) for item in raw_data]


def get_datetime_from_timestamp(ts: str) -> datetime:
    """
    Convert timestamp string in the format "2023/04/10 (Mon) 23:07" to a datetime object
    """
    dt = datetime.strptime(ts, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=UTC)
    return dt
