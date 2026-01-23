from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..auth import get_current_user, get_db
from ..models import Project, ProjectMember, User
from ..schemas import ProjectCreate, ProjectOut, InviteMember

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("", response_model=ProjectOut)
def create_project(payload: ProjectCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    p = Project(name=payload.name, owner_id=user.id)
    db.add(p); db.commit(); db.refresh(p)
    db.add(ProjectMember(project_id=p.id, user_id=user.id, role="owner")); db.commit()
    return p

@router.get("", response_model=List[ProjectOut])
def list_projects(user=Depends(get_current_user), db: Session = Depends(get_db)):
    q = db.query(Project).join(ProjectMember, Project.id == ProjectMember.project_id)\
                         .filter(ProjectMember.user_id == user.id)
    return q.order_by(Project.created_at.desc()).all()

@router.post("/{project_id}/members")
def add_member(project_id: int, req: InviteMember, user=Depends(get_current_user), db: Session = Depends(get_db)):
    me = db.query(ProjectMember).filter_by(project_id=project_id, user_id=user.id).first()
    if not me:
        raise HTTPException(status_code=403, detail="Not a project member")
    member = db.query(User).filter(User.email == req.email).first()
    if not member:
        raise HTTPException(status_code=404, detail="User not found")
    exists = db.query(ProjectMember).filter_by(project_id=project_id, user_id=member.id).first()
    if exists:
        return {"status": "already_member"}
    db.add(ProjectMember(project_id=project_id, user_id=member.id, role=req.role or "member"))
    db.commit()
    return {"status": "added"}
