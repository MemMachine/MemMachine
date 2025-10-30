"""Tests for SessionData and other request parameters in memmachine.server.app."""

from memmachine.server.app import NewEpisode, SessionData


def test_empty_session_data():
    data = SessionData()
    assert data.group_id == "default"
    assert data.session_id == "default"
    assert data.user_id == ["default"]
    assert data.agent_id == []


def test_user_id_only_session_data():
    data = SessionData(user_id=["tom"])
    assert data.user_id == ["tom"]
    assert data.group_id == "tom"
    assert data.session_id == "tom"
    assert data.agent_id == []


def test_multiple_user_ids_session_data():
    data = SessionData(user_id=["bob", "alice"])
    assert data.user_id == ["alice", "bob"]
    assert data.group_id == "5#alice3#bob"
    assert data.session_id == "5#alice3#bob"
    assert data.agent_id == []


def test_specify_group_user_and_session_ids():
    data = SessionData(
        group_id="group1",
        session_id="session1",
        user_id=["charlie"],
        agent_id=["agentX"],
    )
    assert data.group_id == "group1"
    assert data.session_id == "session1"
    assert data.user_id == ["charlie"]
    assert data.agent_id == ["agentX"]


def test_user_id_does_not_affect_group_and_session_if_specified():
    data = SessionData(
        group_id="group2",
        session_id="session2",
    )
    assert data.group_id == "group2"
    assert data.session_id == "session2"
    assert data.user_id == ["default"]


def test_default_producer_if_user_id_available():
    session = SessionData(user_id=["dave"])
    episode = NewEpisode(session=session)
    assert episode.producer == "dave"


def test_default_producer_with_multiple_user_ids():
    session = SessionData(user_id=["frank", "eve"])
    episode = NewEpisode(session=session)
    assert episode.producer == "3#eve5#frank"


def test_default_producer_with_empty_session():
    session = SessionData()
    episode = NewEpisode(session=session)
    assert episode.producer == "default"


def test_merge_session_data_with_default():
    session = SessionData(
        user_id=["dave", "carol"],
        group_id="group_a",
        session_id="session_b",
        agent_id=["my_agent", "another_agent"],
    )
    dft_session = SessionData()
    session.merge(dft_session)
    assert session.user_id == ["carol", "dave"]
    assert session.group_id == "group_a"
    assert session.session_id == "session_b"
    assert session.agent_id == ["another_agent", "my_agent"]


def test_no_default_agent_if_user_exists():
    session = SessionData(
        user_id=["dave", "carol"],
        group_id="group_a",
        session_id="session_b",
    )
    dft_session = SessionData()
    session.merge(dft_session)
    assert session.user_id == ["carol", "dave"]
    assert session.group_id == "group_a"
    assert session.session_id == "session_b"
    assert session.agent_id == []


def test_no_default_user_if_agent_exists():
    session = SessionData(
        group_id="group_a",
        session_id="session_b",
        agent_id=["my_agent", "another_agent"],
    )
    dft_session = SessionData()
    session.merge(dft_session)
    assert session.user_id == []
    assert session.group_id == "group_a"
    assert session.session_id == "session_b"
    assert session.agent_id == ["another_agent", "my_agent"]
