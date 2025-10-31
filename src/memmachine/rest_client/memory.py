"""
Memory management interface for MemMachine.

This module provides the Memory class that handles episodic and profile memory
operations for a specific context.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class Memory:
    """
    Memory interface for managing episodic and profile memory.
    
    This class provides methods for adding, searching, and managing memories
    within a specific context (group, agent, user, session).
    
    Example:
        ```python
        from memmachine import MemMachineClient
        
        client = MemMachineClient()
        memory = client.memory(
            group_id="my_group",
            agent_id="my_agent",
            user_id="user123",
            session_id="session456"
        )
        
        # Add a memory
        memory.add("I like pizza", metadata={"type": "preference"})
        
        # Search memories
        results = memory.search("What do I like to eat?")
        ```
    """
    
    def __init__(
        self,
        client,
        group_id: Optional[str] = None,
        agent_id: Optional[Union[str, List[str]]] = None,
        user_id: Optional[Union[str, List[str]]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Memory instance.
        
        Args:
            client: MemMachineClient instance
            group_id: Group identifier
            agent_id: Agent identifier(s)
            user_id: User identifier(s)
            session_id: Session identifier
            **kwargs: Additional configuration options
        """
        self.client = client
        self.group_id = group_id
        
        # Normalize agent_id and user_id to lists
        if agent_id is None:
            self.agent_id = None
        elif isinstance(agent_id, list):
            self.agent_id = agent_id if agent_id else None
        else:
            self.agent_id = [agent_id]
        
        if user_id is None:
            self.user_id = None
        elif isinstance(user_id, list):
            self.user_id = user_id if user_id else None
        else:
            self.user_id = [user_id]
        
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate required fields
        if not self.user_id or len(self.user_id) == 0 or not self.agent_id or len(self.agent_id) == 0:
            raise ValueError("Both user_id and agent_id are required and cannot be empty")
        
        # Ensure group_id is non-empty to avoid server defaulting issues
        if not self.group_id or len(self.group_id) < 1:
            self.group_id = self.user_id[0] if self.user_id else "default"
    
    def add(
        self,
        content: str,
        producer: Optional[str] = None,
        produced_for: Optional[str] = None,
        episode_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a memory episode.
        
        Args:
            content: The content to store in memory
            producer: Who produced this content (defaults to first user_id)
            produced_for: Who this content is for (defaults to first agent_id)
            episode_type: Type of episode (default: "text")
            metadata: Additional metadata for the episode
            
        Returns:
            True if the memory was added successfully
            
        Raises:
            requests.RequestException: If the request fails
        """
        # Set default producer and produced_for to match the session context
        # These must be in the user_id or agent_id lists sent in headers
        if not producer:
            producer = self.user_id[0] if self.user_id else "unknown"
        if not produced_for:
            produced_for = self.agent_id[0] if self.agent_id else "unknown"
        
        # Validate that producer and produced_for are in the session context
        # Server requires these to be in either user_id or agent_id lists
        user_id_list = self.user_id or []
        agent_id_list = self.agent_id or []
        
        if producer not in user_id_list and producer not in agent_id_list:
            raise ValueError(
                f"producer '{producer}' must be in user_id {user_id_list} or agent_id {agent_id_list}. "
                f"Current context: user_id={user_id_list}, agent_id={agent_id_list}"
            )
        if produced_for not in user_id_list and produced_for not in agent_id_list:
            raise ValueError(
                f"produced_for '{produced_for}' must be in user_id {user_id_list} or agent_id {agent_id_list}. "
                f"Current context: user_id={user_id_list}, agent_id={agent_id_list}"
            )
        
        episode_data = {
            "producer": producer,
            "produced_for": produced_for,
            "episode_content": content,
            "episode_type": episode_type,
            "metadata": metadata or {}
        }
        
        # Prepare session headers - these must match what the server expects
        # Important: The user_id and agent_id in headers must match what was used
        # when the session was created, or the session must be recreated
        headers = {}
        if self.group_id:
            headers["group-id"] = self.group_id
        if self.session_id:
            headers["session-id"] = self.session_id
        if self.agent_id:
            headers["agent-id"] = ",".join(self.agent_id)
        if self.user_id:
            headers["user-id"] = ",".join(self.user_id)
        
        # Log the request details for debugging
        logger.debug(
            f"Adding memory: producer={producer}, produced_for={produced_for}, "
            f"user_id={self.user_id}, agent_id={self.agent_id}, "
            f"group_id={self.group_id}, session_id={self.session_id}"
        )
        
        try:
            response = self.client._session.post(
                f"{self.client.base_url}/v1/memories",
                json=episode_data,
                headers=headers,
                timeout=self.client.timeout
            )
            response.raise_for_status()
            logger.info(f"Successfully added memory: {content[:50]}...")
            return True
        except requests.RequestException as e:
            # Try to get detailed error information from response
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = f" Response: {e.response.text}"
                except:
                    error_detail = f" Status: {e.response.status_code}"
            logger.error(f"Failed to add memory: {e}{error_detail}")
            raise
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for memories.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            filter_dict: Additional filters for the search
            
        Returns:
            Dictionary containing search results from both episodic and profile memory
            
        Raises:
            requests.RequestException: If the request fails
        """
        search_data = {
            "query": query,
            "filter": filter_dict,
            "limit": limit
        }
        
        # Prepare session headers
        headers = {}
        if self.group_id:
            headers["group-id"] = self.group_id
        if self.session_id:
            headers["session-id"] = self.session_id
        if self.agent_id:
            headers["agent-id"] = ",".join(self.agent_id)
        if self.user_id:
            headers["user-id"] = ",".join(self.user_id)
        
        try:
            response = self.client._session.post(
                f"{self.client.base_url}/v1/memories/search",
                json=search_data,
                headers=headers,
                timeout=self.client.timeout
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Search completed for query: {query}")
            return data.get("content", {})
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current memory context.
        
        Returns:
            Dictionary containing the context information
        """
        return {
            "group_id": self.group_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id
        }
    
    def __repr__(self):
        return f"Memory(group_id='{self.group_id}', user_id='{self.user_id}', session_id='{self.session_id}')"
