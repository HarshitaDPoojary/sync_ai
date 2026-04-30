import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, select

from app.models.db import User, UserIntegration, get_engine


def _session(engine):
    return Session(engine, expire_on_commit=False)


class UserRepository:
    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def get_by_clerk_id(self, clerk_user_id: str) -> Optional[User]:
        with _session(self._engine) as session:
            return session.exec(
                select(User).where(User.clerk_user_id == clerk_user_id)
            ).first()

    def get_by_id(self, user_id: str) -> Optional[User]:
        with _session(self._engine) as session:
            return session.get(User, user_id)

    def create(self, clerk_user_id: str, email: str, name: str = "") -> User:
        with _session(self._engine) as session:
            user = User(
                id=str(uuid.uuid4()),
                clerk_user_id=clerk_user_id,
                email=email,
                name=name,
            )
            session.add(user)
            session.commit()
            return user

    def get_integration(self, user_id: str, provider: str) -> Optional[UserIntegration]:
        with _session(self._engine) as session:
            return session.exec(
                select(UserIntegration)
                .where(UserIntegration.user_id == user_id)
                .where(UserIntegration.provider == provider)
            ).first()

    def upsert_integration(self, user_id: str, provider: str, **kwargs) -> UserIntegration:
        with _session(self._engine) as session:
            existing = session.exec(
                select(UserIntegration)
                .where(UserIntegration.user_id == user_id)
                .where(UserIntegration.provider == provider)
            ).first()
            if existing:
                for key, value in kwargs.items():
                    setattr(existing, key, value)
                existing.updated_at = datetime.now(timezone.utc)
                session.add(existing)
                session.commit()
                return existing
            integration = UserIntegration(
                id=str(uuid.uuid4()),
                user_id=user_id,
                provider=provider,
                **kwargs,
            )
            session.add(integration)
            session.commit()
            return integration

    def delete_integration(self, user_id: str, provider: str) -> None:
        with _session(self._engine) as session:
            existing = session.exec(
                select(UserIntegration)
                .where(UserIntegration.user_id == user_id)
                .where(UserIntegration.provider == provider)
            ).first()
            if existing:
                session.delete(existing)
                session.commit()
