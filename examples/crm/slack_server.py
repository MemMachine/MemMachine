import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from slack_service import SlackService

load_dotenv()

# Setup balanced logging - informative but not overwhelming
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() or "INFO"
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress only the most verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

router = APIRouter(prefix="/slack", tags=["Slack"])
slack_service = SlackService()

MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
CRM_SERVER_URL = os.getenv("CRM_SERVER_URL", "http://localhost:8000")

message_counters = {"processed": 0, "skipped": 0, "errors": 0}


def get_counter_status():
    """Get current counter status"""
    return {
        "processed": message_counters["processed"],
        "skipped": message_counters["skipped"],
        "errors": message_counters["errors"],
    }


def reset_counters():
    """Reset all counters to zero"""
    message_counters["processed"] = 0
    message_counters["skipped"] = 0
    message_counters["errors"] = 0


@router.post("/events")
async def slack_events(
    request: Request,
    x_slack_signature: Optional[str] = Header(default=None, alias="X-Slack-Signature"),
    x_slack_request_timestamp: Optional[str] = Header(
        default=None, alias="X-Slack-Request-Timestamp"
    ),
):
    signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")
    raw_body: bytes = await request.body()

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if payload.get("type") == "url_verification":
        return PlainTextResponse(content=payload.get("challenge", ""))

    if (
        not x_slack_signature
        or not x_slack_request_timestamp
        or not verify_slack_signature(
            signing_secret, x_slack_request_timestamp, x_slack_signature, raw_body
        )
    ):
        logger.warning("[SLACK] Invalid signature")
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    if payload.get("type") != "event_callback":
        return PlainTextResponse(content="ignored")

    event = payload.get("event") or {}
    event_type = event.get("type")
    subtype = event.get("subtype")
    bot_id = event.get("bot_id")
    channel = event.get("channel")
    user = event.get("user")
    text = event.get("text") or ""
    ts = event.get("ts")
    thread_ts = event.get("thread_ts")

    # Only log important events, not every message
    if event_type == "message" and not subtype and not bot_id:
        logger.debug(
            f"[SLACK] Processing message from user {user} in channel {channel}"
        )

    if event_type != "message" or subtype is not None or bot_id:
        return PlainTextResponse(content="ignored")

    # optional channel restrictions
    allow_channel = os.getenv("CRM_CHANNEL_ID")
    if allow_channel and channel != allow_channel:
        return PlainTextResponse(content="ignored")

    stripped = (text or "").lstrip()
    low = stripped.lower()

    if low.startswith("*q") or low.startswith("*q "):
        query_text = stripped[2:].lstrip()
        asyncio.create_task(
            process_query_and_reply(channel, ts, thread_ts, user, query_text)
        )
        return PlainTextResponse(content="ok")

    else:
        asyncio.create_task(process_memory_post(channel, ts, thread_ts, user, text))
        return PlainTextResponse(content="ok")


def verify_slack_signature(
    secret: str, timestamp: str, signature: str, body: bytes
) -> bool:
    try:
        req_ts = int(timestamp)
    except Exception:
        return False
    if abs(int(time.time()) - req_ts) > 300:
        return False

    base_string = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    computed = hmac.new(secret.encode("utf-8"), base_string, hashlib.sha256).hexdigest()
    expected = f"v0={computed}"
    return hmac.compare_digest(expected, signature)


async def process_memory_post(
    channel: str, ts: str, thread_ts: Optional[str], user: str, text: str
) -> bool:
    """Post all messages to memory system with efficient deduplication

    Returns:
        bool: True if message was skipped (already processed), False if successfully processed
    """
    logger.debug(f"[SLACK] Processing message from user {user}")

    (await slack_service.get_user_display_name(user) if user else (user or ""))

    slack_message_id = f"slack_{channel}_{user}_{ts}"

    crm_server_url = f"{CRM_SERVER_URL}/memory"

    params = {
        "user_id": user,
        "query": text,
        "slack_message_id": slack_message_id,
    }

    logger.debug(f"[SLACK] POST -> {crm_server_url}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(crm_server_url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "skipped":
                    logger.debug(f"[SLACK] Message {ts} already processed, skipped")
                    message_counters["skipped"] += 1
                    return True
                else:
                    logger.debug("[SLACK] Message posted to memory successfully")
                    message_counters["processed"] += 1
                    return False
            else:
                logger.warning(f"[SLACK] Failed to post to memory: {resp.status_code}")
                message_counters["errors"] += 1
                return False
    except Exception as e:
        logger.error(f"[SLACK] Error posting to memory: {e}", exc_info=True)
        message_counters["errors"] += 1
        return False


async def process_query_and_reply(
    channel: str, ts: str, thread_ts: Optional[str], user: str, query_text: str
):
    """Handle *Q queries by searching memory and using OpenAI chat completion"""
    logger.info(f"[SLACK] Processing query from user {user}: {query_text[:50]}...")

    (await slack_service.get_user_display_name(user) if user else (user or ""))

    search_url = f"{CRM_SERVER_URL}/memory"

    params = {"query": query_text, "user_id": user, "timestamp": str(int(time.time()))}

    logger.debug(f"[SLACK] GET -> {search_url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(search_url, params=params)
            logger.debug(f"[SLACK] memory search resp status={resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    formatted_query = data.get("formatted_query", "")

                    response_text = await generate_openai_response(
                        formatted_query, query_text
                    )
                else:
                    response_text = (
                        f"⚠️ Search failed: {data.get('message', 'Unknown error')}"
                    )
            else:
                response_text = f"⚠️ Search failed with status {resp.status_code}"

    except Exception as e:
        logger.error(f"[SLACK] Error searching memory: {e}")
        response_text = f"⚠️ Error searching memory: {str(e)}"

    await slack_service.post_message(
        channel=channel, text=response_text, thread_ts=thread_ts or ts
    )
    logger.info("[SLACK] Query response posted")


async def generate_openai_response(formatted_query: str, original_query: str) -> str:
    """Generate response using OpenAI chat completion"""
    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            return "⚠️ OpenAI API key not configured"

        client = openai.AsyncOpenAI(api_key=api_key)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful CRM assistant. Use the provided context to answer the user's question accurately and concisely.",
            },
            {"role": "user", "content": formatted_query},
        ]

        logger.debug(f"[OPENAI] Sending request with {len(formatted_query)} characters")

        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=1000, temperature=0.7
        )

        response_text = response.choices[0].message.content
        logger.debug(f"[OPENAI] Generated response: {len(response_text)} characters")

        return response_text

    except Exception as e:
        logger.error(f"[OPENAI] Error generating response: {e}")
        return f"⚠️ Error generating AI response: {str(e)}"


def get_user_input() -> int:
    """Get user input for number of messages to ingest"""
    try:
        print("\n" + "=" * 50)
        print("SLACK HISTORICAL MESSAGE INGESTION")
        print("=" * 50)
        print("Enter number of historical messages to ingest per channel.")
        print("Press Enter for default (5 messages), or type a number.")
        print("=" * 50)

        user_input = input("Messages per channel [5]: ").strip()

        if not user_input:
            return 5

        try:
            num_messages = int(user_input)
            if num_messages < 0:
                print("⚠️ Negative number not allowed, using default (5)")
                return 5
            return num_messages
        except ValueError:
            print("⚠️ Invalid input, using default (5)")
            return 5

    except (EOFError, KeyboardInterrupt):
        print("\n⚠️ Input interrupted, using default (5)")
        return 5


app = FastAPI(title="CRM Slack Integration")
app.include_router(router)


@app.get("/counters")
async def get_counters():
    """Get current message processing counters"""
    return get_counter_status()


@app.post("/counters/reset")
async def reset_counters_endpoint():
    """Reset all counters to zero"""
    reset_counters()
    return {"status": "success", "message": "Counters reset"}


async def main():
    """Main function that handles user input before starting server"""
    message_limit = get_user_input()

    config = uvicorn.Config(app, host="0.0.0.0", port=8001)
    server = uvicorn.Server(config)

    print("\nStarting Slack server on port 8001...")
    print("Server ready for real-time messages!")
    print(f"\n{'=' * 50}")
    print("HISTORICAL INGESTION")
    print(f"{'=' * 50}")
    print(f"Processing {message_limit} messages per channel...")
    print(f"{'=' * 50}")

    historical_task = asyncio.create_task(
        ingest_historical_messages_with_limit(message_limit)
    )

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("[SLACK] Shutting down...")
        if not historical_task.done():
            historical_task.cancel()
            try:
                await historical_task
            except asyncio.CancelledError:
                pass


async def ingest_historical_messages_with_limit(message_limit: int):
    """Ingest historical messages with the specified limit"""
    logger.info(
        f"[SLACK] Starting historical message ingestion with limit: {message_limit}"
    )

    reset_counters()

    enable_historical = os.getenv("SLACK_ENABLE_HISTORICAL", "true").lower() == "true"
    if not enable_historical:
        logger.info(
            "[SLACK] Historical ingestion disabled via SLACK_ENABLE_HISTORICAL=false"
        )
        return

    try:
        channels = await slack_service.get_all_channels()
        if not channels:
            logger.warning(
                "[SLACK] No accessible channels found for historical ingestion"
            )
            print("❌ No accessible channels found")
            return

        allow_channel = os.getenv("CRM_CHANNEL_ID")
        if allow_channel:
            channels = [ch for ch in channels if ch.get("id") == allow_channel]
            logger.info(f"[SLACK] Limited to specific channel: {allow_channel}")
            print(f"Channel filter: {allow_channel}")

        print(f"Channels to process: {len(channels)}")
        for i, channel in enumerate(channels, 1):
            channel_name = channel.get("name", "unknown")
            print(f"  {i}. #{channel_name}")

        print("\nProcessing messages...")

        for channel in channels:
            channel_id = channel.get("id")
            channel_name = channel.get("name", "unknown")

            logger.info(f"[SLACK] Processing channel: {channel_name} ({channel_id})")
            print(f"  #{channel_name}...", end=" ", flush=True)

            messages = await slack_service.get_channel_history(
                channel_id, limit=message_limit
            )

            if not messages:
                logger.info(f"[SLACK] No messages found in channel {channel_name}")
                print("⚠️ No messages found")
                continue

            channel_processed = 0
            channel_skipped = 0
            for msg in messages:
                user = msg.get("user")
                text = msg.get("text", "")
                ts = msg.get("ts")
                thread_ts = msg.get("thread_ts")

                if not user or not text or not ts:
                    continue

                was_skipped = await process_memory_post(
                    channel_id, ts, thread_ts, user, text
                )

                if was_skipped:
                    channel_skipped += 1
                else:
                    channel_processed += 1

                await asyncio.sleep(0.1)

            logger.info(
                f"[SLACK] Processed {channel_processed} messages, skipped {channel_skipped} messages from channel {channel_name}"
            )
            if channel_skipped > 0:
                print(f"✅ {channel_processed} processed, {channel_skipped} skipped")
            else:
                print(f"✅ {channel_processed} processed")

        counters = get_counter_status()
        print(f"\n{'=' * 50}")
        print("INGESTION COMPLETE")
        print(f"{'=' * 50}")
        print(
            f"Total: {counters['processed']} processed, {counters['skipped']} skipped"
        )
        if counters["errors"] > 0:
            print(f"❌ Errors: {counters['errors']}")
        print(f"{'=' * 50}")

        logger.info(
            f"[SLACK] Historical ingestion complete. Processed: {counters['processed']}, Skipped: {counters['skipped']}, Errors: {counters['errors']}"
        )

    except Exception as e:
        logger.error(f"[SLACK] Error during historical ingestion: {e}")
        print(f"❌ Error during historical ingestion: {e}")
        import traceback

        logger.error(f"[SLACK] Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())
