import logging
import os
from typing import Any, Dict, Optional

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger(__name__)


class SlackService:
    """
    Thin async wrapper around Slack Web API for the operations we need.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("SLACK_BOT_TOKEN")
        if not self.token:
            logger.warning("SLACK_BOT_TOKEN is not set; Slack operations will fail")
        self.client = AsyncWebClient(token=self.token)

    async def get_user_display_name(self, user_id: str) -> str:
        """
        Resolve a Slack user ID to a human-friendly display name.
        Fallback order: profile.display_name -> real_name -> user_id
        """
        try:
            resp = await self.client.users_info(user=user_id)
            user: Dict[str, Any] = resp.get("user") or {}
            profile: Dict[str, Any] = user.get("profile") or {}
            display = (profile.get("display_name") or "").strip()
            real = (profile.get("real_name") or "").strip()
            return display or real or user_id
        except SlackApiError as e:
            logger.error(
                f"[SLACK] users.info failed: {e.response['error'] if hasattr(e, 'response') else str(e)}"
            )
            return user_id
        except Exception:
            logger.exception("[SLACK] users.info unexpected error")
            return user_id

    async def post_message(
        self, channel: str, text: str, thread_ts: Optional[str] = None
    ) -> Optional[str]:
        """
        Post a message to a channel (optionally in a thread). Returns ts if successful.
        """
        try:
            resp = await self.client.chat_postMessage(
                channel=channel, text=text, thread_ts=thread_ts, mrkdwn=True
            )
            return resp.get("ts")
        except SlackApiError as e:
            logger.error(
                f"[SLACK] chat.postMessage failed: {e.response['error'] if hasattr(e, 'response') else str(e)}"
            )
            return None
        except Exception:
            logger.exception("[SLACK] chat.postMessage unexpected error")
            return None

    async def get_channel_history(
        self, channel: str, limit: int = 1000
    ) -> list[Dict[str, Any]]:
        """
        Fetch historical messages from a Slack channel.
        Returns list of message objects with metadata.
        """
        try:
            messages = []
            cursor = None

            while True:
                resp = await self.client.conversations_history(
                    channel=channel, limit=min(limit, 200), cursor=cursor
                )

                batch_messages = resp.get("messages", [])
                if not batch_messages:
                    break

                human_messages = [
                    msg
                    for msg in batch_messages
                    if not msg.get("bot_id") and not msg.get("subtype")
                ]

                messages.extend(human_messages)

                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor or len(messages) >= limit:
                    break

            logger.info(
                f"[SLACK] Fetched {len(messages)} historical messages from channel {channel}"
            )
            return messages

        except SlackApiError as e:
            logger.error(
                f"[SLACK] conversations.history failed: {e.response['error'] if hasattr(e, 'response') else str(e)}"
            )
            return []
        except Exception:
            logger.exception("[SLACK] conversations.history unexpected error")
            return []

    async def get_all_channels(self) -> list[Dict[str, Any]]:
        """
        Get list of all channels the bot has access to.
        """
        try:
            channels = []
            cursor = None

            while True:
                resp = await self.client.conversations_list(
                    types="public_channel,private_channel", cursor=cursor, limit=200
                )

                batch_channels = resp.get("channels", [])
                if not batch_channels:
                    break

                accessible_channels = [
                    ch
                    for ch in batch_channels
                    if ch.get("is_member", False) and not ch.get("is_archived", False)
                ]

                channels.extend(accessible_channels)

                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            logger.info(f"[SLACK] Found {len(channels)} accessible channels")
            return channels

        except SlackApiError as e:
            logger.error(
                f"[SLACK] conversations.list failed: {e.response['error'] if hasattr(e, 'response') else str(e)}"
            )
            return []
        except Exception:
            logger.exception("[SLACK] conversations.list unexpected error")
            return []
