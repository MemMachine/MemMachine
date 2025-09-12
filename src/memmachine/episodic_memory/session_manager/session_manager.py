"""Manages database sessions for multi-agent and multi-user conversations."""

import json
import os
from typing import Annotated

from sqlalchemy import Integer, MetaData, String, Table, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..data_types import SessionInfo, GroupConfiguration


# Base class for declarative class definitions
class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """
    Base class for declarative class definitions.
    """


IntKeyColumn = Annotated[int, mapped_column(Integer, primary_key=True)]
IntColumn = Annotated[int, mapped_column(Integer)]
StringColumn = Annotated[str, mapped_column(String)]


class SessionManager:
    """
    Handles the lifecycle of conversation sessions, including creation,
    retrieval, and deletion.
    """

    class MemSession(Base):  # pylint: disable=too-few-public-methods
        """ORM model for a session."""

        __tablename__ = "sessions"
        id: Mapped[IntKeyColumn]
        timestamp: Mapped[IntColumn]
        group_id: Mapped[StringColumn]
        agent_ids: Mapped[StringColumn]
        user_ids: Mapped[StringColumn]  # JSON string of a list of user IDs
        session_id: Mapped[StringColumn]
        configuration: Mapped[StringColumn]

    class User(Base):  # pylint: disable=too-few-public-methods
        """ORM model for a user's association with a session."""

        __tablename__ = "users"
        id: Mapped[IntKeyColumn]
        user_id: Mapped[StringColumn]
        session_id: Mapped[IntColumn]  # Foreign key to sessions.id

    class Agent(Base):  # pylint: disable=too-few-public-methods
        """ORM model for an agent's association with a session."""

        __tablename__ = "agents"
        id: Mapped[IntKeyColumn]
        agent_id: Mapped[StringColumn]
        session_id: Mapped[IntColumn]  # Foreign key to sessions.id

    class Group(Base):  # pylint: disable=too-few-public-methods
        """ORM model for a group's association with a session."""

        __tablename__ = "groups"
        id: Mapped[IntKeyColumn]
        group_id: Mapped[StringColumn]
        session_id: Mapped[IntColumn]  # Foreign key to sessions.id

    class GroupInfo(Base):
        """ORM model for a group information."""

        __tablename__ = "group_info"
        id: Mapped[IntKeyColumn]
        group_id: Mapped[StringColumn]
        user_list: Mapped[StringColumn]
        agent_list: Mapped[StringColumn]
        configuration: Mapped[StringColumn]

    def __init__(self, config: dict):
        """
        Initializes the SessionManager.

        Args:
            config (dict): A configuration dictionary containing the database
                           connection URI.
                           Example: {"uri": "sqlite:///sessions.db"}

        Raises:
            ValueError: If the "uri" is not provided in the config.
        """
        if config is None:
            raise ValueError(f"""Invalid config: {str(config)}""")

        sql_path = config.get("uri")
        if sql_path is None or len(sql_path) < 1:
            raise ValueError(f"""Invalid sql path: {str(config)}""")
        if sql_path.find(":///") < 0:
            sql_path = "sqlite:///" + sql_path

        # create empty sqlite file if it does not exist
        if sql_path.startswith("sqlite:///"):
            file_path = sql_path.replace("sqlite:///", "")
            if os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")

        self._engine = create_engine(sql_path)
        self._session = sessionmaker(bind=self._engine)
        # Create all tables defined in the Base metadata if they don't exist
        Base.metadata.create_all(self._engine)

    def __del__(self):
        """Destructor to clean up database engine resources."""
        if hasattr(self, "_engine"):
            # Disposes of the connection pool
            self._engine.dispose()

    def create_new_group(
            self,
            group_id: str,
            agent_ids: list[str],
            user_ids: list[str],
            configuration: dict | None = None,
    ) -> tuple[bool, str | None]:
        """
        Creates a new group.
        If the group already exists, this function fails.

        Args:
            group_id (str): The ID of the group.
            agent_ids (list[str]): A list of agent IDs.
            user_ids (list[str]): A list of user IDs.
            configuration (dict | None): A dictionary for group
                                          configuration.
        Returns:
            tuple[bool, str | None]: The first value is if
            the function call successful. The second is the error
            message if the function failed.
        """
        if len(agent_ids) == 0 and len(user_ids) == 0:
            return False, "New group without users or agents"
        with self._session() as dbsession:
            # Query for an existing group with the same ID
            group = (
                dbsession.query(self.GroupInfo)
                .filter(self.GroupInfo.group_id == group_id)
                .first()
            )
            if group is not None:
                return False, f"""Group {group_id} already exists"""
            group = self.GroupInfo(
                group_id=group_id,
                user_list=json.dumps(user_ids),
                agent_list=json.dumps(agent_ids),
                configuration=json.dumps(
                    configuration if configuration is not None else {}
                ),
            )
            dbsession.add(group)
            dbsession.commit()
            dbsession.refresh(group)
            return True, None

    def retrieve_group(self, group_id: str) -> GroupConfiguration | None:
        """
        Retrieves a group by its ID.

        Args:
            group_id (str): The ID of the group.

        Returns:
            GroupConfiguration | None: A GroupConfiguration object if found,
                                        None otherwise.
        """
        with self._session() as dbsession:
            group = (
                dbsession.query(self.GroupInfo)
                .filter(self.GroupInfo.group_id == group_id)
                .first()
            )
            if group is None:
                return None
            return GroupConfiguration(
                group_id=group.group_id,
                agent_list=json.loads(group.agent_list),
                user_list=json.loads(group.user_list),
                configuration=json.loads(group.configuration),
            )

    def delete_group(self, group_id: str):
        """
        Deletes a group by its ID.

        Args:
            group_id (str): The ID of the group.
        """
        with self._session() as dbsession:
            dbsession.query(self.GroupInfo).filter(
                self.GroupInfo.group_id == group_id
            ).delete()
            dbsession.commit()

    def retrieve_all_groups(self) -> list[GroupConfiguration]:
        """
        Retrieves all groups.

        Returns:
            list[GroupConfiguration]: A list of GroupConfiguration objects.
        """
        with self._session() as dbsession:
            groups = dbsession.query(self.GroupInfo).all()
            result = []
            for group in groups:
                result.append(
                    GroupConfiguration(
                        group_id=group.group_id,
                        agent_list=json.loads(group.agent_list),
                        user_list=json.loads(group.user_list),
                        configuration=json.loads(group.configuration),
                    )
                )
            return result

    def create_session_if_not_exist(
        self,
        group_id: str,
        agent_ids: list[str],
        user_ids: list[str],
        session_id: str,
        configuration: dict | None = None,
    ) -> SessionInfo:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """
        Creates a new session if one with the specified group doesn't
        already exist.
        If a session exists, it returns the existing session information.

        Args:
            group_id (str | None): The ID of the group for this session.
            agent_ids (list[str] | None): A list of agent IDs in the session.
            user_ids (list[str] | None): A list of user IDs in the session.
            session_id (str | None): The unique identifier for the session.
            configuration (dict | None): A dictionary for session
            configuration.

        Returns:
            SessionInfo: An object containing the information of the created
            or found session.

        Raises:
            ValueError: If more than one session is found with the given
                        parameters.
        """
        agents = json.dumps(agent_ids)
        users = json.dumps(user_ids)
        config = json.dumps(configuration if configuration is not None else {})
        with self._session() as dbsession:
            # Query for an existing session with the same group id
            sess = (
                dbsession.query(self.MemSession)
                .filter(
                    self.MemSession.group_id == group_id,
                    self.MemSession.session_id == session_id,
                )
                .all()
            )

            if len(sess) > 1:
                raise ValueError(f"""More than one session found with same ID
                                  {group_id}: {session_id}""")

            if len(sess) == 0:
                # Check if group exists. If not, create one
                group = (
                    dbsession.query(self.GroupInfo)
                    .filter(self.GroupInfo.group_id == group_id)
                    .first()
                )
                if group is None:
                    ret, msg = self.create_new_group(group_id, agent_ids,
                                                     user_ids, configuration)
                    if not ret:
                        raise ValueError(msg)
                else:
                    agents = group.agent_list
                    users = group.user_list
                    config = group.configuration

                # Create a new session if it doesn't exist
                new_sess = self.MemSession(
                    timestamp=int(os.times()[4]),
                    group_id=group_id,
                    agent_ids=agents,
                    user_ids=users,
                    session_id=session_id,
                    configuration=config,
                )
                dbsession.add(new_sess)
                dbsession.commit()
                # Refresh the object to get the database-generated ID
                dbsession.refresh(new_sess)
                sess_data = new_sess
                group_link = self.Group(
                    group_id=group_id, session_id=new_sess.id
                )
                dbsession.add(group_link)
                for agent_id in agent_ids:
                    agent_link = self.Agent(
                        agent_id=agent_id, session_id=new_sess.id
                    )
                    dbsession.add(agent_link)
                for user_id in user_ids:
                    user_link = self.User(
                        user_id=user_id, session_id=new_sess.id
                    )
                    dbsession.add(user_link)
                dbsession.commit()
            else:
                sess_data = sess[0]

            # Return session information as a SessionInfo object
            return SessionInfo(
                id=sess_data.id,
                group_id=sess_data.group_id,
                agent_ids=json.loads(sess_data.agent_ids),
                user_ids=json.loads(sess_data.user_ids),
                session_id=sess_data.session_id,
                configuration=json.loads(sess_data.configuration),
            )

    def get_all_sessions(self) -> list[SessionInfo]:
        """
        Retrieves all sessions from the database.

        Returns:
            list[SessionInfo]: A list of objects containing session
                              information.
        """
        with self._session() as dbsession:
            sessions = dbsession.query(self.MemSession).all()
            result = []
            for session in sessions:
                # Convert each ORM object to a SessionInfo dataclass instance
                result.append(
                    SessionInfo(
                        id=session.id,
                        group_id=session.group_id,
                        agent_ids=json.loads(session.agent_ids),
                        user_ids=json.loads(session.user_ids),
                        session_id=session.session_id,
                        configuration=json.loads(session.configuration),
                    )
                )
        return result

    def get_session_by_user(self, usr_id: str) -> list[SessionInfo]:
        """
        Retrieves all sessions associated with a specific user ID.

        Args:
            usr_id (str): The ID of the user.

        Returns:
            list[SessionInfo]: A list of session information objects for
                               the given user.
        """
        with self._session() as dbsession:
            # Find all session links for the given user ID.
            # Note: This performs N+1 queries. For better performance, a JOIN
            # would be preferable.
            user_session_links = (
                dbsession.query(self.User)
                .filter(self.User.user_id == usr_id)
                .all()
            )
            result = []
            for link in user_session_links:
                # For each link, fetch the full session details
                sess = (
                    dbsession.query(self.MemSession)
                    .filter(self.MemSession.id == link.session_id)
                    .first()
                )
                if sess is None:
                    continue

                result.append(
                    SessionInfo(
                        id=sess.id,
                        group_id=sess.group_id,
                        agent_ids=json.loads(sess.agent_ids),
                        user_ids=json.loads(sess.user_ids),
                        session_id=sess.session_id,
                        configuration=json.loads(sess.configuration),
                    )
                )
        return result

    def get_session_by_group(self, group_id: str) -> list[SessionInfo]:
        """
        Retrieves all sessions associated with a specific group ID.

        Args:
            group_id (str): The ID of the group.

        Returns:
            list[SessionInfo]: A list of session information objects for the
                               given group.
        """
        with self._session() as dbsession:
            # Find all session links for the given group ID.
            # Note: This performs N+1 queries. For better performance, a JOIN
            #       would be preferable.
            group_session_links = (
                dbsession.query(self.Group)
                .filter(self.Group.group_id == group_id)
                .all()
            )
            result = []
            for link in group_session_links:
                # For each link, fetch the full session details
                sess = (
                    dbsession.query(self.MemSession)
                    .filter(self.MemSession.id == link.session_id)
                    .first()
                )
                if sess is None:
                    continue

                result.append(
                    SessionInfo(
                        id=sess.id,
                        group_id=sess.group_id,
                        agent_ids=json.loads(sess.agent_ids),
                        user_ids=json.loads(sess.user_ids),
                        session_id=sess.session_id,
                        configuration=json.loads(sess.configuration),
                    )
                )
        return result

    def get_session_by_agent(self, agent_id: str) -> list[SessionInfo]:
        """
        Retrieves all sessions associated with a specific agent ID.

        Args:
            usr_id (str): The ID of the agent.

        Returns:
            list[SessionInfo]: A list of session information objects for the
                              given agent.
        """
        with self._session() as dbsession:
            # Find all session links for the given agent ID.
            # Note: This performs N+1 queries. For better performance, a JOIN
            # would be preferable.
            agent_session_links = (
                dbsession.query(self.Agent)
                .filter(self.Agent.agent_id == agent_id)
                .all()
            )
            result = []
            for link in agent_session_links:
                # For each link, fetch the full session details
                sess = (
                    dbsession.query(self.MemSession)
                    .filter(self.MemSession.id == link.session_id)
                    .first()
                )
                if sess is None:
                    continue

                result.append(
                    SessionInfo(
                        id=sess.id,
                        group_id=sess.group_id,
                        agent_ids=json.loads(sess.agent_ids),
                        user_ids=json.loads(sess.user_ids),
                        session_id=sess.session_id,
                        configuration=json.loads(sess.configuration),
                    )
                )
        return result

    def delete_session(
        self,
        group_id: str | None,
        session_id: str,
    ):
        """
        Deletes a session and its associated user, agent, and group links.

        Args:
            group_id (str | None): The ID of the group for the session to
                                   delete.
            agent_ids (list[str] | None): A list of agent IDs for the
                                         session to delete.
            user_ids (list[str]): A list of user IDs for the session to delete.
            session_id (str): The unique identifier for the session to delete.
        """
        with self._session() as dbsession:
            # Find the session to delete
            sessions = (
                dbsession.query(self.MemSession)
                .filter(
                    self.MemSession.session_id == session_id,
                    self.MemSession.group_id == group_id,
                )
                .all()
            )
            if len(sessions) == 0:
                return  # Session not found

            session_to_delete = sessions[0]
            session_db_id = session_to_delete.id

            # Delete the main session entry
            dbsession.delete(session_to_delete)
            dbsession.commit()

            # Manually delete associated records in other tables.
            # A better approach would be to use SQLAlchemy relationships with
            # cascade deletes.
            metadata = MetaData()
            metadata.reflect(bind=self._engine)
            user_table = Table("users", metadata, autoload_with=self._engine)
            agent_table = Table("agents", metadata, autoload_with=self._engine)
            group_table = Table("groups", metadata, autoload_with=self._engine)
            dbsession.execute(
                user_table.delete().where(
                    user_table.c.session_id == session_db_id
                )
            )
            dbsession.execute(
                agent_table.delete().where(
                    agent_table.c.session_id == session_db_id
                )
            )
            dbsession.execute(
                group_table.delete().where(
                    group_table.c.session_id == session_db_id
                )
            )
            dbsession.commit()
