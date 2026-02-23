from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from uuid import uuid4
from typing import List

from ..auth import get_current_user, get_db
from ..models import Thread, Project, ProjectMember, Message
from ..schemas import ThreadCreate, ThreadOut, MessageOut

router = APIRouter(prefix="/threads", tags=["threads"])


def _ensure_project_access(project_id: int, user_id: int, db: Session):
    mem = (
        db.query(ProjectMember)
        .filter_by(project_id=project_id, user_id=user_id)
        .first()
    )
    if not mem:
        raise HTTPException(status_code=403, detail="No access to this project")


@router.get("", response_model=List[ThreadOut])
def list_threads(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
        List threads visible to the current user (personal + project membership).
    Sorted by last_message_at desc to show most recent conversations first.
    """
    # Get user's project IDs
    user_projects = (
        db.query(ProjectMember.project_id)
        .filter(ProjectMember.user_id == user.id)
        .all()
    )
    project_ids = [p[0] for p in user_projects]

    # Query: personal threads OR threads in user's projects
    q = db.query(Thread).filter(
        or_(
            Thread.owner_user_id == user.id,
            Thread.project_id.in_(project_ids) if project_ids else False
        )
    )

    # Order by last_message_at desc (most recent first)
    q = q.order_by(Thread.last_message_at.desc())
    items = q.all()
    return items


@router.post("", response_model=ThreadOut)
def create_thread(payload: ThreadCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
         Create a personal or project thread with proper scope initialization.
    """
    if payload.project_id:
        _ensure_project_access(payload.project_id, user.id, db)
        group_scope = f"user-{user.id}"  
    else:
        group_scope = f"user-{user.id}"

    thread = Thread(
        title=payload.title or "New chat",
        owner_user_id=user.id,
        project_id=payload.project_id,
        session_id=f"t-{uuid4().hex[:12]}",
        group_scope=group_scope,
        active_model="gpt-4o-mini",
    )
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


@router.get("/{thread_id}", response_model=ThreadOut)
def get_thread(thread_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
        Get a specific thread by ID.
    """
    thread = db.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # access check
    if thread.project_id:
        _ensure_project_access(thread.project_id, user.id, db)
    elif thread.owner_user_id != user.id:
        raise HTTPException(status_code=403, detail="No access to this chat")

    return thread


@router.get("/{thread_id}/messages", response_model=List[MessageOut])
def list_messages(thread_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
        Load all messages for a thread, properly sorted.
    """
    thread = db.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # access check
    if thread.project_id:
        _ensure_project_access(thread.project_id, user.id, db)
    elif thread.owner_user_id != user.id:
        raise HTTPException(status_code=403, detail="No access to this chat")

    msgs = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return msgs


@router.put("/{thread_id}", response_model=ThreadOut)
def rename_thread(thread_id: int, payload: ThreadCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Rename thread with proper access control.
    """
    thread = db.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Allow owner or project members to rename
    if thread.project_id:
        _ensure_project_access(thread.project_id, user.id, db)
    elif thread.owner_user_id != user.id:
        raise HTTPException(status_code=403, detail="No access to this chat")
    
    thread.title = payload.title or thread.title
    db.commit()
    db.refresh(thread)
    return thread


@router.delete("/{thread_id}")
def delete_thread(thread_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
     Delete thread with cascade to messages.
    """
    thread = db.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Only owner can delete
    if thread.owner_user_id != user.id:
        raise HTTPException(status_code=403, detail="Only the thread owner can delete it")

    # Messages will cascade delete due to foreign key constraint
    db.delete(thread)
    db.commit()
    return {"ok": True, "message": "Thread deleted successfully"}