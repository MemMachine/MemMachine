from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Any, Optional
import tempfile, os
import json
import asyncio
import logging
import re
from sqlalchemy.orm import Session
from openai import OpenAI

from ..auth import get_current_user, get_db, get_user_api_keys
from ..config import settings
from ..schemas import ChatResponse
from ..models import Thread, ProjectMember, Message

from .. import memmachine_client
from ..utils.parser import sniff_and_read
from ..utils.memory import chunk_text
from ..utils.model_utils import get_model_display_name
from ..utils.tavily_client import TavilyClient

# Set up logging
logger = logging.getLogger(__name__)

# ✅ VERSION MARKER - Updated for V2
CHAT_FILE_VERSION = "V2-MEMMACHINE-2025-12-04"
print("=" * 80)
print(f"🚀 CHAT.PY LOADED - VERSION: {CHAT_FILE_VERSION}")
print("=" * 80)

router = APIRouter(prefix="/chat", tags=["chat"])


def _ensure_access(thread: Thread, user_id: int, db: Session):
    """Check whether the current user can access this chat thread."""
    if thread.project_id:
        member = (
            db.query(ProjectMember)
            .filter_by(project_id=thread.project_id, user_id=user_id)
            .first()
        )
        if not member:
            raise HTTPException(status_code=403, detail="No access to this project chat")
    elif thread.owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="No access to this personal chat")


def _get_project_owner_email(thread: Thread, db: Session) -> Optional[str]:
    """
    Get the project owner's email if this is a project thread.
    Returns None for personal threads.
    """
    if not thread.project_id or not thread.project:
        return None
    
    # Access through SQLAlchemy relationship
    if thread.project.owner:
        return thread.project.owner.email
    
    return None


def should_search_web(message: str) -> bool:
    """
    Determine if the message requires web search.
    Looks for indicators like current events, recent info, real-time data queries.
    """
    search_triggers = [
        # Explicit search requests
        r'\b(search|find|look up|google)\b',
        # Current events
        r'\b(latest|recent|current|today|now|this week|this month|2024|2025)\b',
        # Real-time data
        r'\b(price|stock|weather|news|score|result)\b',
        # Questions about specific things that change
        r'\b(what is|who is|when did|where is).*(now|today|currently|latest)\b',
    ]
    
    message_lower = message.lower()
    
    for pattern in search_triggers:
        if re.search(pattern, message_lower, re.IGNORECASE):
            logger.info(f"🔍 Web search triggered by pattern: {pattern}")
            return True
    
    return False


def extract_profile_facts(message: str, response: str) -> list[str]:
    """
    Extract profile facts from conversation that should be permanently remembered.
    Returns list of facts to store in profile memory.
    """
    facts = []
    
    # Patterns for extracting user information from their messages
    user_patterns = [
        # Location
        (r'\b(?:I live in|I\'m in|I\'m from|I am from|I am in|located in|based in)\s+([A-Z][A-Za-z\s,]+)', 'location'),
        (r'\b(?:my location is|my city is)\s+([A-Z][A-Za-z\s,]+)', 'location'),
        
        # Name
        (r'\b(?:my name is|I\'m|I am|call me)\s+([A-Z][a-z]+)', 'name'),
        
        # Job/occupation
        (r'\b(?:I work as|I\'m a|I am a|I work at|my job is)\s+([a-z\s]+)', 'occupation'),
        
        # Preferences
        (r'\b(?:I like|I love|I enjoy|I prefer|my favorite)\s+([a-z\s]+)', 'preference'),
        (r'\b(?:I don\'t like|I hate|I dislike)\s+([a-z\s]+)', 'dislike'),
    ]
    
    for pattern, fact_type in user_patterns:
        matches = re.finditer(pattern, message, re.IGNORECASE)
        for match in matches:
            value = match.group(1).strip()
            if len(value) > 2 and len(value) < 100:  
                if fact_type == 'location':
                    facts.append(f"User lives in {value}")
                elif fact_type == 'name':
                    facts.append(f"User's name is {value}")
                elif fact_type == 'occupation':
                    facts.append(f"User works as {value}")
                elif fact_type == 'preference':
                    facts.append(f"User likes {value}")
                elif fact_type == 'dislike':
                    facts.append(f"User doesn't like {value}")
    
    # Extract confirmed facts from AI responses
    if response:
        response_patterns = [
            r'(?:you live in|you\'re in|you\'re from|you are from|you are in)\s+([A-Z][A-Za-z\s,]+)',
            r'(?:your name is)\s+([A-Z][a-z]+)',
            r'(?:you work as|you\'re a)\s+([a-z\s]+)',
        ]
        
        for pattern in response_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                fact = match.group(0).strip()
                if len(fact) > 5 and len(fact) < 150:
                    facts.append(f"Confirmed: {fact}")
    
    return facts


@router.post("/stream")
async def chat_stream(
    thread_id: int = Form(...),
    message: str = Form(...),
    model: str = Form("gpt-4o-mini"),
    temperature: float = Form(1.0),
    system_prompt: str = Form(""),
    file: Optional[UploadFile] = File(None),
    user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Chat endpoint with streaming, memory integration, and multi-provider support."""
    
    # Load thread + access check FIRST
    thread = db.query(Thread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    _ensure_access(thread, user.id, db)
    
    # Get user's API keys for all providers
    user_api_keys = get_user_api_keys(user)
    
    # Determine which provider to use based on model
    provider = 'openai'  # default
    model_lower = model.lower()
    
    print(f"\n🔍 PROVIDER DETECTION:")
    print(f"   Model: {model}")
    print(f"   Model lower: {model_lower}")
    
    if 'claude' in model_lower:
        provider = 'anthropic'
        print(f"   ✅ Detected 'claude' in model name → provider = 'anthropic'")
    elif 'gemini' in model_lower:
        provider = 'google'
        print(f"   ✅ Detected 'gemini' in model name → provider = 'google'")
    elif 'gpt' in model_lower or 'o1' in model_lower:
        provider = 'openai'
        print(f"   ✅ Detected 'gpt' or 'o1' in model name → provider = 'openai'")
    else:
        print(f"   ⚠️ No keyword match, defaulting to: openai")
    
    print(f"   After detection: provider = '{provider}'")
    
    # Get the appropriate API key
    user_api_key = user_api_keys.get(provider)
    print(f"   API key for '{provider}': {bool(user_api_key)}")
    
    # Fallback to OpenAI if chosen provider has no key
    if not user_api_key and provider != 'openai':
        print(f"   ⚠️ NO API KEY for '{provider}', falling back to OpenAI")
        user_api_key = user_api_keys.get('openai')
        provider = 'openai'
        model = 'gpt-4o-mini'  # Default OpenAI model
    
    print(f"   FINAL: provider = '{provider}', has_key = {bool(user_api_key)}\n")
    
    if not user_api_key:
        raise HTTPException(
            status_code=400,
            detail=f"No {provider} API key configured. Please add your API key in settings."
        )

    # Build scope identifiers for V2
    uid = str(user.id)
    project_owner_email = _get_project_owner_email(thread, db)
    
    logger.info(f"=== CHAT REQUEST (Streaming) - V2 ===")
    logger.info(f"Thread: {thread.id}, User: {uid}, Provider: {provider}, Model: {model}")
    logger.info(f"Project: {thread.project_id if thread.project_id else 'Personal'}")
    if project_owner_email:
        logger.info(f"Project Owner: {project_owner_email}")
    logger.info(f"Using user's {provider} API key")
    logger.info(f"Message: {message[:100]}...")

    # Handle file upload
    uploaded_texts = []
    if file and file.filename:
        try:
            file_content = file.file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            try:
                parsed, _ = sniff_and_read(tmp_path, file.filename)
            except:
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    parsed = f.read()
            
            os.remove(tmp_path)
            
            uploaded_texts.append(f"[File: {file.filename}]\n{parsed}")
            
            logger.info(f"File uploaded: {file.filename}, size: {len(parsed)} chars")
            
            # ✅ V2: Store file upload as semantic memory
            try:
                memmachine_client.add_semantic_memory(
                    user_email=user.email,
                    content=f"File: {file.filename}\n\n{parsed}",
                    project_id=thread.project_id,
                    project_owner_email=project_owner_email,
                    metadata={
                        "type": "file_upload",
                        "filename": file.filename,
                        "size": len(parsed),
                        "thread_id": thread.id
                    }
                )
                logger.info(f"✅ File stored in semantic memory (V2)")
            except Exception as e:
                logger.error(f"❌ Failed to store file in memory: {e}")
            
        except Exception as e:
            logger.error(f"File processing error: {e}")
            uploaded_texts.append(f"[File: {file.filename}]\n[Error reading file]")

    # Get user location from database
    user_location = None
    try:
        if user.location_city:
            if user.location_state:
                user_location = f"{user.location_city}, {user.location_state}"
            else:
                user_location = user.location_city
            logger.info(f"📍 USER LOCATION (from database): {user_location}")
        else:
            logger.info("📍 No user location found in database")
    except Exception as e:
        logger.error(f"❌ Error loading user location: {e}")
    
    # Web search with Tavily
    web_search_results = ""
    tavily_key = user_api_keys.get('tavily') or settings.tavily_api_key
    if tavily_key and should_search_web(message):
        try:
            logger.info("🌐 Performing Tavily web search...")
            
            # Enhance search query with user's location if applicable
            search_query = message
            weather_keywords = ["weather", "temperature", "forecast", "climate", "local", "here", "today"]
            has_weather_keyword = any(keyword in message.lower() for keyword in weather_keywords)
            
            if user_location and has_weather_keyword:
                search_query = f"{message} in {user_location}"
                logger.info(f"✅ Enhanced search query with location: '{search_query}'")
            
            tavily = TavilyClient(tavily_key)
            logger.info(f"🌐 Searching Tavily with query: '{search_query}'")
            search_result = tavily.search(search_query, max_results=5)
            
            # Apply location filtering if we have a user location
            web_search_results = tavily.format_results_for_llm(
                search_result, 
                filter_location=user_location if has_weather_keyword else None
            )
            logger.info(f"✅ Web search complete: {len(web_search_results)} chars")
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            web_search_results = ""

    # ✅ V2: CROSS-THREAD MEMORY SEARCH
    logger.info("=== MEMORY SEARCH (CROSS-THREAD V2) ===")
    try:
        memory_results = memmachine_client.search_memories(
            user_email=user.email,
            query=message or "context",
            project_id=thread.project_id,  # Determines scope (personal vs project)
            project_owner_email=project_owner_email,
            thread_id=None,  # ← KEY: None = search ALL threads (cross-thread)
            limit=20,
            search_semantic=True,
            search_episodic=True
        )
        
        semantic_results = memory_results.get("semantic_results", [])
        episodic_results = memory_results.get("episodic_results", [])
        
        logger.info(f"Memory search returned: {len(episodic_results)} episodic, {len(semantic_results)} semantic")
        
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        semantic_results = []
        episodic_results = []

    # Separate current thread vs other threads
    current_thread_memories = []
    other_thread_memories = []
    profile_facts = []
    
    logger.info("=== CATEGORIZING MEMORIES ===")
    
    # Process semantic memories (profile facts, documents)
    for m in semantic_results:
        content = m.get("profile_content") or ""
        if content:
            profile_facts.append(content)
            logger.debug(f"Semantic memory: {content[:50]}...")
    
    # Process episodic memories
    for m in episodic_results:
        metadata = m.get("metadata", {})
        mem_thread_id = metadata.get("thread_id")
        content = m.get("episode_content") or ""
        
        if not content:
            continue
            
        if mem_thread_id == thread.id:
            current_thread_memories.append(content)
            logger.debug(f"Current thread memory: {content[:50]}...")
        else:
            other_thread_memories.append(content)
            logger.debug(f"Other thread memory (thread {mem_thread_id}): {content[:50]}...")
    
    logger.info(f"Categorized: {len(profile_facts)} semantic, {len(current_thread_memories)} current, {len(other_thread_memories)} other")

    # Build enriched system prompt
    current_model_name = get_model_display_name(model)
    
    base_instructions = """You are an intelligent AI assistant with persistent memory and web search capabilities.

**CRITICAL - CONTEXT AWARENESS:**
You have access to detailed information about this user in the USER PROFILE section below. This includes:
- Personal preferences and interests
- Location and demographic information
- Past experiences and background
- Skills, education, and professional information
- Likes, dislikes, and habits
- Any other facts the user has shared

**ALWAYS use this profile information when relevant to the query:**
- Location-based queries (weather, time, local events, restaurants, etc.) → Use their location
- Recommendation queries (movies, books, games, food, etc.) → Use their preferences and interests
- Personal questions ("what do I...", "where do I...", "when did I...") → Use profile facts
- Scheduling/planning queries → Use their timezone, schedule, location
- Shopping/product queries → Use their preferences, past purchases, interests
- Career/education queries → Use their background, skills, education
- ANY query where profile context is relevant → Use it proactively

**DO NOT:**
- Ask for information that's already in the profile
- Ignore profile context when it's relevant
- Provide generic answers when personalized answers are possible
- Say "I don't know" if the answer is in the profile

**Model Information:**
You are currently using the {model_name} model ({provider})."""
    
    # Add current date and time context
    from datetime import datetime
    current_datetime = datetime.now()
    
    system_prompt_parts = [
        system_prompt if system_prompt else base_instructions.format(model_name=current_model_name, provider=provider),
        f"\n\n⏰ **CURRENT DATE & TIME:**\n"
        f"Today is {current_datetime.strftime('%A, %B %d, %Y')} at {current_datetime.strftime('%I:%M %p')}.\n"
        f"**CRITICAL:** Use this date for all time-sensitive queries."
    ]

    # Add user location
    if user_location:
        system_prompt_parts.append(
            f"\n\n📍 **USER LOCATION:**\n"
            f"The user is located in: {user_location}\n"
            f"**CRITICAL:** When the user asks about 'local', 'here', 'my area', 'my city', or similar terms, "
            f"they are referring to {user_location}."
        )

    # Add profile facts
    if profile_facts:
        logger.info(f"Adding {len(profile_facts)} profile facts to prompt")
        system_prompt_parts.append(
            f"\n\n{'='*70}\n"
            f"👤 **USER PROFILE** - CRITICAL CONTEXT ABOUT THIS USER\n"
            f"{'='*70}\n"
            f"The following are established facts about this user:\n\n" + 
            "\n".join(f"✓ {fact}" for fact in profile_facts[:30]) +
            f"\n{'='*70}\n"
        )

    if current_thread_memories:
        logger.info(f"Adding {len(current_thread_memories[:20])} current thread memories")
        system_prompt_parts.append(
            f"\n\n📝 **CURRENT CONVERSATION CONTEXT:**\n" + 
            "\n".join(f"- {m}" for m in current_thread_memories[:20])
        )

    if other_thread_memories:
        logger.info(f"Adding {len(other_thread_memories[:15])} other thread memories")
        system_prompt_parts.append(
            f"\n\n💭 **PREVIOUS CONVERSATIONS:**\n" + 
            "\n".join(f"- {m}" for m in other_thread_memories[:15])
        )

    if uploaded_texts:
        system_prompt_parts.append(f"\n\n📄 **UPLOADED DOCUMENTS:**\n" + "\n\n".join(uploaded_texts))

    if web_search_results:
        is_location_query = any(kw in message.lower() for kw in ["weather", "temperature", "forecast", "climate"])
        system_prompt_parts.append(f"\n\n🌐 **WEB SEARCH RESULTS:**\n{web_search_results}")
        
        if is_location_query and user_location:
            system_prompt_parts.append(
                f"\n\n**CRITICAL:** The user asked: '{message}'. "
                f"Their location is {user_location}. Provide ONLY {user_location} information."
            )

    final_system_prompt = "".join(system_prompt_parts)
    logger.info(f"System prompt length: {len(final_system_prompt)} chars")

    # Load conversation history from database
    hist = db.query(Message).filter_by(thread_id=thread.id).order_by(Message.created_at.asc()).all()
    
    conversation_messages = []
    for h in hist:
        if h.sender == "user":
            conversation_messages.append({"role": "user", "content": h.content or "[file uploaded]"})
        else:
            conversation_messages.append({"role": "assistant", "content": h.content})
    
    conversation_messages.append({"role": "user", "content": message})
    logger.info(f"Conversation history: {len(hist)} previous messages")

    # Save user message to DB
    user_msg = Message(
        thread_id=thread.id, 
        sender="user", 
        content=message, 
        type="text", 
        model_used=model,
        provider_used=provider if hasattr(Message, 'provider_used') else None,
        user_email=user.email 
    )
    db.add(user_msg)
    db.flush()

    # ✅ V2: Store user message in MemMachine
    logger.info("Storing user message in MemMachine V2")
    try:
        memmachine_client.add_memory(
            user_email=user.email,
            content=message,
            thread_id=thread.id,
            project_id=thread.project_id,
            project_owner_email=project_owner_email,
            role="user",
            metadata={
                "sender": "user",
                "model": model,
                "provider": provider
            }
        )
        logger.info("✅ User message stored in V2")
    except Exception as e:
        logger.error(f"❌ Failed to store message: {e}")

    async def generate():
        full_response = ""
        
        try:
            logger.info(f"Starting {provider} streaming")
            
            # Call appropriate AI provider
            if provider == 'openai':
                client = OpenAI(api_key=user_api_key)
                
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": final_system_prompt}, *conversation_messages],
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                        await asyncio.sleep(0.01)
            
            elif provider == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=user_api_key)
                
                claude_messages = []
                system_message = final_system_prompt
                
                for msg in conversation_messages:
                    if msg["role"] == "system":
                        system_message += "\n\n" + msg["content"]
                    else:
                        claude_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_message,
                    messages=claude_messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        yield f"data: {json.dumps({'content': text})}\n\n"
                        await asyncio.sleep(0.01)
            
            elif provider == 'google':
                import google.generativeai as genai
                genai.configure(api_key=user_api_key)
                
                gemini_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": 4096,
                    }
                )
                
                prompt_parts = []
                if final_system_prompt:
                    prompt_parts.append(f"System Instructions:\n{final_system_prompt}\n\n")
                
                for msg in conversation_messages:
                    if msg["role"] == "user":
                        prompt_parts.append(f"User: {msg['content']}\n")
                    elif msg["role"] == "assistant":
                        prompt_parts.append(f"Assistant: {msg['content']}\n")
                
                full_prompt = "".join(prompt_parts)
                response = gemini_model.generate_content(full_prompt, stream=True)
                
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        full_response += chunk.text
                        yield f"data: {json.dumps({'content': chunk.text})}\n\n"
                        await asyncio.sleep(0.01)
            
            yield "data: [DONE]\n\n"
            logger.info(f"Response complete: {len(full_response)} chars")
            
            # Save assistant reply to DB
            asst_msg = Message(
                thread_id=thread.id, 
                sender="assistant", 
                content=full_response, 
                type="text", 
                model_used=model,
                provider_used=provider if hasattr(Message, 'provider_used') else None
            )
            db.add(asst_msg)

            # ✅ V2: Store assistant response
            logger.info("Storing assistant response in MemMachine V2")
            try:
                memmachine_client.add_memory(
                    user_email=user.email,
                    content=full_response,
                    thread_id=thread.id,
                    project_id=thread.project_id,
                    project_owner_email=project_owner_email,
                    role="assistant",
                    metadata={
                        "sender": "assistant",
                        "model": model,
                        "provider": provider
                    }
                )
                logger.info("✅ Assistant response stored in V2")
            except Exception as e:
                logger.error(f"❌ Failed to store response: {e}")

            # Extract and store profile facts
            logger.info("=== EXTRACTING PROFILE FACTS ===")
            profile_facts_extracted = extract_profile_facts(message, full_response)
            
            if profile_facts_extracted:
                logger.info(f"Found {len(profile_facts_extracted)} profile facts")
                for fact in profile_facts_extracted:
                    logger.info(f"Storing profile fact: {fact}")
                    try:
                        memmachine_client.add_semantic_memory(
                            user_email=user.email,
                            content=fact,
                            project_id=thread.project_id,
                            project_owner_email=project_owner_email,
                            metadata={
                                "type": "user_fact",
                                "extracted_from": "conversation",
                                "thread_id": thread.id
                            }
                        )
                    except Exception as e:
                        logger.error(f"❌ Failed to store fact: {e}")

            # Update thread
            thread.active_model = model
            if hasattr(thread, 'active_provider'):
                thread.active_provider = provider
            
            from datetime import datetime
            thread.last_message_at = datetime.utcnow()

            db.commit()
            logger.info("=== CHAT COMPLETE (Streaming V2) ===\n")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Streaming error: {e}", exc_info=True)
            
            error_detail = str(e)
            if "invalid_api_key" in error_detail.lower() or "incorrect api key" in error_detail.lower():
                error_msg = f"Invalid {provider} API key. Please update your API key in settings."
            elif "insufficient_quota" in error_detail.lower() or "quota" in error_detail.lower():
                error_msg = f"API key has insufficient quota. Please check your {provider} account."
            elif "rate_limit" in error_detail.lower():
                error_msg = f"Rate limit exceeded for {provider}. Please wait a moment and try again."
            elif "model" in error_detail.lower() and "not found" in error_detail.lower():
                error_msg = f"Model '{model}' not available."
            else:
                error_msg = f"Error: {error_detail}"
            
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/", response_model=ChatResponse)
def chat_endpoint(
    thread_id: int = Form(...),
    message: str = Form(...),
    model: str = Form("gpt-4o-mini"),
    temperature: float = Form(1.0),
    system_prompt: str = Form(""),
    file: Optional[UploadFile] = File(None),
    user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Chat endpoint (non-streaming) with V2 memory integration."""
    
    # Load thread first
    thread = db.query(Thread).filter_by(id=thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    _ensure_access(thread, user.id, db)
    
    # Get project owner email if this is a project thread
    project_owner_email = _get_project_owner_email(thread, db)
    
    # Get user's API keys
    user_api_keys = get_user_api_keys(user)
    
    # Determine provider
    provider = 'openai'
    model_lower = model.lower()
    
    if 'claude' in model_lower:
        provider = 'anthropic'
    elif 'gemini' in model_lower:
        provider = 'google'
    elif 'gpt' in model_lower or 'o1' in model_lower:
        provider = 'openai'
    
    user_api_key = user_api_keys.get(provider)
    
    if not user_api_key and provider != 'openai':
        user_api_key = user_api_keys.get('openai')
        provider = 'openai'
        model = 'gpt-4o-mini'
    
    if not user_api_key:
        raise HTTPException(
            status_code=400,
            detail=f"No {provider} API key configured."
        )

    uid = str(user.id)
    logger.info(f"=== CHAT REQUEST (Non-Streaming V2) ===")
    logger.info(f"Thread: {thread.id}, User: {uid}, Provider: {provider}")

    # Handle file upload
    uploaded_texts = []
    if file and file.filename:
        try:
            file_content = file.file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            try:
                parsed, _ = sniff_and_read(tmp_path, file.filename)
            except:
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    parsed = f.read()
            
            os.remove(tmp_path)
            uploaded_texts.append(f"[File: {file.filename}]\n{parsed}")
            
            # Store in semantic memory
            memmachine_client.add_semantic_memory(
                user_email=user.email,
                content=f"File: {file.filename}\n\n{parsed}",
                project_id=thread.project_id,
                project_owner_email=project_owner_email,
                metadata={"type": "file_upload", "filename": file.filename}
            )
            
        except Exception as e:
            logger.error(f"File processing error: {e}")

    # Get user location
    user_location = None
    try:
        if user.location_city:
            user_location = f"{user.location_city}, {user.location_state}" if user.location_state else user.location_city
    except:
        pass
    
    # Web search
    web_search_results = ""
    tavily_key = user_api_keys.get('tavily') or settings.tavily_api_key
    if tavily_key and should_search_web(message):
        try:
            search_query = message
            has_weather_keyword = any(kw in message.lower() for kw in 
                                    ["weather", "temperature", "forecast", "climate", "local", "here"])
            if user_location and has_weather_keyword:
                search_query = f"{message} in {user_location}"
            
            tavily = TavilyClient(tavily_key)
            search_result = tavily.search(search_query, max_results=5)
            web_search_results = tavily.format_results_for_llm(
                search_result,
                filter_location=user_location if has_weather_keyword else None
            )
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")

    # V2 Memory search
    try:
        memory_results = memmachine_client.search_memories(
            user_email=user.email,
            query=message or "context",
            project_id=thread.project_id,
            project_owner_email=project_owner_email,
            thread_id=None,  # Cross-thread
            limit=20,
            search_semantic=True,
            search_episodic=True
        )
        
        semantic_results = memory_results.get("semantic_results", [])
        episodic_results = memory_results.get("episodic_results", [])
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        semantic_results = []
        episodic_results = []

    # Categorize memories
    current_thread_memories = []
    other_thread_memories = []
    profile_facts = []
    
    for m in semantic_results:
        content = m.get("profile_content") or ""
        if content:
            profile_facts.append(content)
    
    for m in episodic_results:
        metadata = m.get("metadata", {})
        mem_thread_id = metadata.get("thread_id")
        content = m.get("episode_content") or ""
        
        if not content:
            continue
            
        if mem_thread_id == thread.id:
            current_thread_memories.append(content)
        else:
            other_thread_memories.append(content)

    # Build system prompt
    current_model_name = get_model_display_name(model)
    base_instructions = """You are an intelligent AI assistant with persistent memory."""
    
    from datetime import datetime
    current_datetime = datetime.now()
    
    system_prompt_parts = [
        system_prompt if system_prompt else base_instructions,
        f"\n\n⏰ Today is {current_datetime.strftime('%A, %B %d, %Y')}."
    ]

    if user_location:
        system_prompt_parts.append(f"\n\n📍 User location: {user_location}")

    if profile_facts:
        system_prompt_parts.append(
            f"\n\n👤 **USER PROFILE:**\n" + 
            "\n".join(f"✓ {f}" for f in profile_facts[:30])
        )

    if current_thread_memories:
        system_prompt_parts.append(
            f"\n\n📝 **CURRENT CONVERSATION:**\n" + 
            "\n".join(f"- {m}" for m in current_thread_memories[:20])
        )

    if other_thread_memories:
        system_prompt_parts.append(
            f"\n\n💭 **PREVIOUS CONVERSATIONS:**\n" + 
            "\n".join(f"- {m}" for m in other_thread_memories[:15])
        )

    if uploaded_texts:
        system_prompt_parts.append(f"\n\n📄 **UPLOADED:**\n" + "\n\n".join(uploaded_texts))

    if web_search_results:
        system_prompt_parts.append(f"\n\n🌐 **WEB SEARCH:**\n{web_search_results}")

    final_system_prompt = "".join(system_prompt_parts)

    # Load conversation history
    hist = db.query(Message).filter_by(thread_id=thread.id).order_by(Message.created_at.asc()).all()
    
    conversation_messages = []
    for h in hist:
        if h.sender == "user":
            conversation_messages.append({"role": "user", "content": h.content or "[file]"})
        else:
            conversation_messages.append({"role": "assistant", "content": h.content})
    
    conversation_messages.append({"role": "user", "content": message})

    # Save user message
    user_msg = Message(
        thread_id=thread.id, 
        sender="user", 
        content=message, 
        type="text", 
        model_used=model,
        provider_used=provider if hasattr(Message, 'provider_used') else None,
        user_email=user.email 
    )
    db.add(user_msg)
    db.flush()

    # Store in V2
    memmachine_client.add_memory(
        user_email=user.email,
        content=message,
        thread_id=thread.id,
        project_id=thread.project_id,
        project_owner_email=project_owner_email,
        role="user",
        metadata={"provider": provider}
    )

    try:
        # Call AI provider
        if provider == 'openai':
            client = OpenAI(api_key=user_api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": final_system_prompt}, *conversation_messages],
                temperature=temperature
            )
            reply = completion.choices[0].message.content
        
        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=user_api_key)
            
            claude_messages = [m for m in conversation_messages if m["role"] != "system"]
            
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                system=final_system_prompt,
                messages=claude_messages
            )
            reply = response.content[0].text
        
        elif provider == 'google':
            import google.generativeai as genai
            genai.configure(api_key=user_api_key)
            
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config={"temperature": temperature, "max_output_tokens": 4096}
            )
            
            prompt_parts = []
            if final_system_prompt:
                prompt_parts.append(f"System: {final_system_prompt}\n\n")
            
            for msg in conversation_messages:
                if msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}\n")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}\n")
            
            response = gemini_model.generate_content("".join(prompt_parts))
            reply = response.text

        # Save assistant message
        asst_msg = Message(
            thread_id=thread.id, 
            sender="assistant", 
            content=reply, 
            type="text", 
            model_used=model,
            provider_used=provider if hasattr(Message, 'provider_used') else None
        )
        db.add(asst_msg)

        # Store in V2
        memmachine_client.add_memory(
            user_email=user.email,
            content=reply,
            thread_id=thread.id,
            project_id=thread.project_id,
            project_owner_email=project_owner_email,
            role="assistant",
            metadata={"provider": provider}
        )

        # Extract profile facts
        profile_facts_extracted = extract_profile_facts(message, reply)
        for fact in profile_facts_extracted:
            memmachine_client.add_semantic_memory(
                user_email=user.email,
                content=fact,
                project_id=thread.project_id,
                project_owner_email=project_owner_email,
                metadata={"type": "user_fact", "thread_id": thread.id}
            )

        # Update thread
        thread.active_model = model
        if hasattr(thread, 'active_provider'):
            thread.active_provider = provider
        
        from datetime import datetime
        thread.last_message_at = datetime.utcnow()

        db.commit()
        
        return ChatResponse(reply=reply, used_context=[])

    except Exception as e:
        db.rollback()
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))