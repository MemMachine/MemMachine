"""Unit tests for the SessionManager class."""

import os

import pytest

from memmachine.episodic_memory.data_types import SessionInfo
from memmachine.episodic_memory.session_manager.session_manager import (
    SessionManager,
)


@pytest.fixture
def session_manager():
    """Pytest fixture to set up and tear down a SessionManager with a test
    database."""
    db_path = "test_sessions.db"
    config = {"uri": f"sqlite:///{db_path}"}
    # Ensure the db file doesn't exist from a previous failed run
    if os.path.exists(db_path):
        os.remove(db_path)

    manager = SessionManager(config)
    yield manager

    # Teardown
    del manager
    if os.path.exists(db_path):
        os.remove(db_path)


def test_create_session(session_manager: SessionManager):
    """Test creating a session."""
    # Create a session without group
    with pytest.raises(ValueError):
        session_manager.create_session(
            group_id="group1",
            session_id="session1",
            configuration={"key": "value"},
        )
    # create a new group first
    session_manager.create_new_group(
        group_id="group1",
        agent_ids=["agent1"],
        user_ids=["user1"],
        configuration={"key": "value"},
    )
    session_info = session_manager.create_session(
        group_id="group1",
        session_id="session1",
        configuration={"key": "value"},
    )
    assert isinstance(session_info, SessionInfo)
    assert session_info.group_id == "group1"
    assert session_info.agent_ids == ["agent1"]
    assert session_info.user_ids == ["user1"]

    # create a same session
    with pytest.raises(ValueError):
        session_manager.create_session(
            group_id="group1",
            session_id="session1",
            configuration={"key": "value"},
        )

    # create session with different ID
    session_info = session_manager.create_session(
        group_id="group1",
        session_id="session2",
        configuration={"key": "value"},
    )
    session = session_manager.open_session("group1", "session1")
    assert session is not None
    assert session.group_id == "group1"
    assert session.agent_ids == ["agent1"]
    assert session.user_ids == ["user1"]
    assert session.session_id == "session1"
    assert session.configuration == {"key": "value"}
    with pytest.raises(ValueError):
        session_info = session_manager.open_session("group1", "session4")
    # Verify it's in the DB
    sessions = session_manager.get_all_sessions()
    assert len(sessions) == 2
    assert sessions[0].session_id != sessions[1].session_id
    assert sessions[0].session_id in ["session1", "session2"]
    assert sessions[1].session_id in ["session1", "session2"]
    assert sessions[0].group_id == "group1"
    assert sessions[1].group_id == "group1"
    assert sessions[0].agent_ids == ["agent1"]
    assert sessions[1].agent_ids == ["agent1"]
    assert sessions[0].user_ids == ["user1"]
    assert sessions[1].user_ids == ["user1"]


def test_create_group(session_manager: SessionManager):
    """Test creating a group."""
    session_manager.create_new_group(
        group_id="group1",
        agent_ids=["agent1"],
        user_ids=["user1"],
        configuration={"key": "value"},
    )

    # Verify it's in the DB
    group = session_manager.retrieve_group("group1")
    assert group is not None
    assert group.group_id == "group1"
    assert group.agent_list == ["agent1"]
    assert group.user_list == ["user1"]
    assert group.configuration == {"key": "value"}
    assert len(session_manager.retrieve_all_groups()) == 1
    # Test create the same group
    with pytest.raises(ValueError):
        session_manager.create_new_group(
            group_id="group1",
            agent_ids=["agent1"],
            user_ids=["user1"],
            configuration={"key": "value"},
        )

    # create a group with different ID
    session_manager.create_new_group(
        group_id="group2",
        agent_ids=["agent1"],
        user_ids=["user1"],
        configuration={"key": "value"},
    )

    groups = session_manager.retrieve_all_groups()
    assert len(groups) == 2
    assert groups[0].group_id != groups[1].group_id
    assert groups[0].group_id in ["group1", "group2"]
    assert groups[1].group_id in ["group1", "group2"]
    session_manager.delete_group("group1")
    assert len(session_manager.retrieve_all_groups()) == 1
    # create a session for group2
    session_manager.create_session(
        group_id="group2",
        session_id="session1",
        configuration={"key": "value"},
    )
    with pytest.raises(ValueError):
        session_manager.delete_group("group2")
    session_manager.delete_session("group2", "session1")
    session_manager.delete_group("group2")
    assert len(session_manager.retrieve_all_groups()) == 0

    # Test createing a group with invalid parameter
    with pytest.raises(ValueError):
        session_manager.create_new_group(
            group_id="group",
            agent_ids=[],
            user_ids=[],
            configuration=None,
        )


def test_create_session_if_not_exist_new(session_manager: SessionManager):
    """Test creating a new session."""
    session_info = session_manager.create_session_if_not_exist(
        group_id="group1",
        agent_ids=["agent1"],
        user_ids=["user1"],
        session_id="session1",
        configuration={"key": "value"},
    )
    assert isinstance(session_info, SessionInfo)
    assert session_info.group_id == "group1"
    assert session_info.agent_ids == ["agent1"]
    assert session_info.user_ids == ["user1"]
    assert session_info.session_id == "session1"
    assert session_info.configuration == {"key": "value"}

    # Verify it's in the DB
    all_sessions = session_manager.get_all_sessions()
    assert len(all_sessions) == 1
    assert all_sessions[0].session_id == "session1"

    # Verify link tables
    session = session_manager.get_session_by_group("group1")
    assert len(session) == 1
    assert session[0].group_id == "group1"
    session = session_manager.get_session_by_agent("agent1")
    assert len(session) == 1
    assert session[0].group_id == "group1"
    assert "agent1" in session[0].agent_ids
    session = session_manager.get_session_by_user("user1")
    assert len(session) == 1
    assert session[0].group_id == "group1"
    assert "user1" in session[0].user_ids


def test_create_session_if_not_exist_existing(session_manager: SessionManager):
    """Test retrieving an existing session."""
    # Create a session first
    session_manager.create_session_if_not_exist(
        group_id="group1",
        agent_ids=["agent1"],
        user_ids=["user1"],
        session_id="session1",
        configuration={"key": "value"},
    )

    # Request it again
    session_info = session_manager.create_session_if_not_exist(
        group_id="group1",
        agent_ids=["agent1"],
        user_ids=["user1"],
        session_id="session1",
        configuration={"key": "value"},
    )

    assert isinstance(session_info, SessionInfo)
    assert session_info.session_id == "session1"

    # Verify there's still only one session in the DB
    all_sessions = session_manager.get_all_sessions()
    assert len(all_sessions) == 1


def test_get_all_sessions(session_manager: SessionManager):
    """Test retrieving all sessions."""
    session_manager.create_session_if_not_exist("g1", ["a1"], ["u1"], "s1")
    session_manager.create_session_if_not_exist("g2", ["a2"], ["u2"], "s2")

    sessions = session_manager.get_all_sessions()
    assert len(sessions) == 2
    session_ids = {s.session_id for s in sessions}
    assert session_ids == {"s1", "s2"}


def test_get_session_by_user(session_manager: SessionManager):
    """Test retrieving sessions by user ID."""
    session_manager.create_session_if_not_exist("g1", ["a1"], ["u1", "u2"], "s1")
    session_manager.create_session_if_not_exist("g2", ["a2"], ["u1"], "s2")
    session_manager.create_session_if_not_exist("g3", ["a3"], ["u3"], "s3")

    user1_sessions = session_manager.get_session_by_user("u1")
    assert len(user1_sessions) == 2
    session_ids = {s.session_id for s in user1_sessions}
    assert session_ids == {"s1", "s2"}

    user3_sessions = session_manager.get_session_by_user("u3")
    assert len(user3_sessions) == 1
    assert user3_sessions[0].session_id == "s3"


def test_get_session_by_agent(session_manager: SessionManager):
    """Test retrieving sessions by agent ID."""
    session_manager.create_session_if_not_exist("g1", ["a1", "a2"], ["u1"], "s1")
    session_manager.create_session_if_not_exist("g2", ["a1"], ["u2"], "s2")
    session_manager.create_session_if_not_exist("g3", ["a3"], ["u3"], "s3")

    agent1_sessions = session_manager.get_session_by_agent("a1")
    assert len(agent1_sessions) == 2
    session_ids = {s.session_id for s in agent1_sessions}
    assert session_ids == {"s1", "s2"}


def test_get_session_by_group(session_manager: SessionManager):
    """Test retrieving sessions by group ID."""
    session_manager.create_session_if_not_exist("g1", ["a1"], ["u1"], "s1")
    session_manager.create_session_if_not_exist("g1", ["a2"], ["u2"], "s2")
    session_manager.create_session_if_not_exist("g3", ["a3"], ["u3"], "s3")

    group1_sessions = session_manager.get_session_by_group("g1")
    assert len(group1_sessions) == 2
    session_ids = {s.session_id for s in group1_sessions}
    assert session_ids == {"s1", "s2"}


def test_delete_session(session_manager: SessionManager):
    """Test deleting a session."""
    session_info = session_manager.create_session_if_not_exist(
        "g1", ["a1"], ["u1"], "s1"
    )
    assert session_info is not None
    assert session_info.session_id == "s1"
    assert session_info.group_id == "g1"

    session_manager.create_session_if_not_exist("g2", ["a2"], ["u2"], "s2")

    assert len(session_manager.get_all_sessions()) == 2

    session_manager.delete_session(group_id="g1", session_id="s1")

    sessions = session_manager.get_all_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == "s2"

    # Verify link tables are also cleaned up
    sessions = session_manager.get_session_by_group("g1")
    assert len(sessions) == 0
    sessions = session_manager.get_session_by_group("g2")
    assert len(sessions) == 1
    assert sessions[0].session_id == "s2"
    sessions = session_manager.get_session_by_agent("a1")
    assert len(sessions) == 0
    sessions = session_manager.get_session_by_user("u1")
    assert len(sessions) == 0


def test_init_with_invalid_config():
    """Test initialization with invalid configuration."""
    with pytest.raises(ValueError):
        SessionManager(None)
    with pytest.raises(ValueError):
        SessionManager({})
    with pytest.raises(ValueError):
        SessionManager({"uri": ""})


def test_create_session_with_null_config(session_manager: SessionManager):
    """Test creating a session with null configuration."""
    session_info = session_manager.create_session_if_not_exist(
        group_id="g1",
        agent_ids=["a1"],
        user_ids=["u1"],
        session_id="s1",
        configuration=None,
    )
    assert session_info.configuration == {}
