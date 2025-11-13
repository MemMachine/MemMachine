from enum import Enum
from typing import Final, Protocol, runtime_checkable

from memmachine.semantic_memory.semantic_model import Resources, SetIdT


class IsolationType(Enum):
    """Isolation scopes supported when mapping activity to semantic-memory set_ids."""

    USER = "user_profile"
    ROLE = "role_profile"
    SESSION = "session"


ALL_MEMORY_TYPES: Final[list[IsolationType]] = [m for m in IsolationType]


@runtime_checkable
class SessionData(Protocol):
    """Protocol exposing the identifiers used to derive set_ids."""

    def user_profile_id(self) -> SetIdT | None:
        raise NotImplementedError

    def session_id(self) -> SetIdT | None:
        raise NotImplementedError

    def role_profile_id(self) -> SetIdT | None:
        raise NotImplementedError


@runtime_checkable
class SessionIdIsolationTypeChecker(Protocol):
    """Protocol for determining the isolation type associated with a set_id."""

    def set_id_isolation_type(self, set_id: SetIdT) -> IsolationType:
        raise NotImplementedError


class SessionIdManager:
    """Generates namespaced set_ids and reports which isolation scope they belong to."""

    def __init__(self):
        pass

    _SESSION_ID_PREFIX: Final[str] = "mem_session_"
    _USER_ID_PREFIX: Final[str] = "mem_user_"
    _ROLE_ID_PREFIX: Final[str] = "mem_role_"

    def generate_session_data(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        role_id: str | None = None,
    ) -> SessionData:
        class _SessionDataImpl:
            """Lightweight `SessionData` implementation backed by generated set_ids."""

            def __init__(self, *, _user_profile_id, _role_profile_id, _session_id):
                self._user_id = _user_profile_id
                self._role_id = _role_profile_id
                self._session_id = _session_id

            def user_profile_id(self) -> str | None:
                return self._user_id

            def role_profile_id(self) -> str | None:
                return self._role_id

            def session_id(self) -> str | None:
                return self._session_id

        return _SessionDataImpl(
            _user_profile_id=self._USER_ID_PREFIX + user_id if user_id else None,
            _session_id=self._SESSION_ID_PREFIX + session_id if session_id else None,
            _role_profile_id=self._ROLE_ID_PREFIX + role_id if role_id else None,
        )

    def set_id_isolation_type(self, _id: str) -> IsolationType:
        if self.is_session_id(_id):
            return IsolationType.SESSION
        elif self.is_producer_id(_id):
            return IsolationType.USER
        elif self.is_role_id(_id):
            return IsolationType.ROLE
        else:
            raise ValueError(f"Invalid id: {_id}")

    def is_session_id(self, _id: str) -> bool:
        return _id.startswith(self._SESSION_ID_PREFIX)

    def is_producer_id(self, _id: str) -> bool:
        return _id.startswith(self._USER_ID_PREFIX)

    def is_role_id(self, _id: str) -> bool:
        return _id.startswith(self._ROLE_ID_PREFIX)


class SessionResourceRetriever:
    """Resolves the `Resources` bundle for a set_id, falling back to isolation defaults."""

    def __init__(
        self,
        session_id_manager: SessionIdIsolationTypeChecker,
        default_resources: dict[IsolationType, Resources],
    ):
        self._session_id_manager = session_id_manager
        self._default_resources = default_resources

    def get_resources(self, set_id: str) -> Resources:
        custom_resources = self._get_set_id_resources(set_id)
        if custom_resources is not None:
            return custom_resources

        set_id_type = self._session_id_manager.set_id_isolation_type(set_id)

        return self._default_resources[set_id_type]

    @staticmethod
    def _get_set_id_resources(_: str) -> Resources | None:
        # TODO: Load set_id custom resources
        return None
