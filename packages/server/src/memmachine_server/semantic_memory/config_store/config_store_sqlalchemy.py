"""SQLAlchemy-backed implementation of the semantic config storage."""

import logging

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    UniqueConstraint,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import Insert as PgInsert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import Insert as SQliteInsert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
)
from sqlalchemy.sql.sqltypes import Boolean, String

from memmachine_server.common.errors import ResourceNotFoundError
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.semantic_model import (
    CategoryIdT,
    SemanticCategory,
    SetIdT,
    SetTypeEntry,
    StructuredSemanticPrompt,
    TagIdT,
)

logger = logging.getLogger(__name__)


class BaseSemanticConfigStore(DeclarativeBase):
    """Declarative base class for Semantic Config Store."""


class SetIdResources(BaseSemanticConfigStore):
    """Resource-level configuration associated with a set identifier."""

    __tablename__ = "semantic_config_setidresources"

    set_id = mapped_column(String, primary_key=True, nullable=False)
    set_name = mapped_column(String, nullable=True)
    set_description = mapped_column(String, nullable=True)

    embedder_name = mapped_column(String, nullable=True)
    language_model_name = mapped_column(String, nullable=True)

    disabled_categories: Mapped[list["DisabledDefaultCategories"]] = relationship(
        "DisabledDefaultCategories",
        cascade="all, delete-orphan",
        single_parent=True,
    )


class SetIdSetType(BaseSemanticConfigStore):
    """One-way mapping assigning a set_id to a set type.

    This mapping is "first write wins": once a `set_id` has been associated
    with a `set_type_id`, subsequent registrations should not overwrite it.
    """

    __tablename__ = "semantic_config_setidresources_settype"

    set_id = mapped_column(String, primary_key=True, nullable=False)
    set_type_id = mapped_column(
        Integer,
        ForeignKey("set_type.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


class DisabledDefaultCategories(BaseSemanticConfigStore):
    """Default categories that are disabled for a given set."""

    __tablename__ = "semantic_config_setidresources_disabledcategories"

    set_id = mapped_column(
        String,
        ForeignKey(
            "semantic_config_setidresources.set_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    disabled_category = mapped_column(String, nullable=False, primary_key=True)


class Category(BaseSemanticConfigStore):
    """Semantic category definition with its prompt description."""

    __tablename__ = "semantic_config_category"

    id = mapped_column(Integer, primary_key=True, nullable=False)
    set_id = mapped_column(String, nullable=True, index=True)
    set_type_id = mapped_column(
        Integer,
        ForeignKey("set_type.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name = mapped_column(String, nullable=False)
    prompt = mapped_column(String, nullable=False)
    description = mapped_column(String, nullable=True)

    tags: Mapped[list["Tag"]] = relationship(
        "Tag",
        back_populates="category",
        cascade="all, delete-orphan",
        single_parent=True,
    )

    def to_typed_model(self) -> SemanticCategory:
        tags = {t.name: t.description for t in self.tags}

        if self.set_id is not None:
            origin_type = "set_id"
            origin_id = self.set_id
            inherited = False
        elif self.set_type_id is not None:
            origin_type = "set_type"
            origin_id = str(self.set_type_id)
            inherited = True
        else:
            origin_type = None
            origin_id = None
            inherited = None

        return SemanticCategory(
            id=CategoryIdT(self.id),
            origin_type=origin_type,
            origin_id=origin_id,
            inherited=inherited,
            name=self.name,
            prompt=StructuredSemanticPrompt(
                description=self.prompt,
                tags=tags,
            ),
        )

    __table_args__ = (
        UniqueConstraint("set_id", "name", name="_set_id_name_uc"),
        UniqueConstraint("set_type_id", "name", name="_set_type_id_name_uc"),
        CheckConstraint(
            "(set_type_id IS NOT NULL AND set_id IS NULL) OR (set_type_id IS NULL AND set_id IS NOT NULL)",
            name="_exactly_one_fk_settype_vs_setid",
        ),
    )


class Tag(BaseSemanticConfigStore):
    """Individual tag that belongs to a semantic category."""

    __tablename__ = "semantic_config_tag"

    id = mapped_column(Integer, primary_key=True, nullable=False)
    name = mapped_column(String, nullable=False)
    description = mapped_column(String, nullable=False)

    category_id = mapped_column(
        Integer,
        ForeignKey("semantic_config_category.id", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[Category] = relationship(
        "Category",
        back_populates="tags",
    )


class SetType(BaseSemanticConfigStore):
    """Mapping of org-level metadata tags to set ids."""

    __tablename__ = "set_type"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    org_level_set: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    metadata_tags_sig: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "org_id",
            "org_level_set",
            "metadata_tags_sig",
            name="uq_org_level_tagsig",
        ),
    )

    def to_typed_model(self) -> SetTypeEntry:
        tags = self.metadata_tags_sig.split(_TAG_SEP)

        return SetTypeEntry(
            id=str(self.id),
            tags=tags,
            is_org_level=self.org_level_set,
            name=self.name,
            description=self.description,
        )


_TAG_SEP = "\x1f"


class SemanticConfigStorageSqlAlchemy(SemanticConfigStorage):
    """Semantic configuration storage that persists data via SQLAlchemy."""

    def __init__(self, sqlalchemy_engine: AsyncEngine) -> None:
        """Initialize the storage with an async SQLAlchemy engine."""
        self._engine = sqlalchemy_engine
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseSemanticConfigStore.metadata.create_all)

    async def delete_all(self) -> None:
        async with self._create_session() as session:
            result = await session.execute(delete(SetIdResources))
            result.close()
            result = await session.execute(delete(SetIdSetType))
            result.close()
            result = await session.execute(delete(Category))
            result.close()
            result = await session.execute(delete(Tag))
            result.close()
            result = await session.execute(delete(SetType))
            result.close()
            result = await session.execute(delete(DisabledDefaultCategories))
            result.close()

            await session.commit()

    async def set_setid_config(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        dialect_name = self._engine.dialect.name

        ins: PgInsert | SQliteInsert
        if dialect_name == "postgresql":
            ins = pg_insert(SetIdResources)
        elif dialect_name == "sqlite":
            ins = sqlite_insert(SetIdResources)
        else:
            # other backends: no ON CONFLICT support
            raise NotImplementedError

        stmt = ins.values(
            set_id=set_id,
            embedder_name=embedder_name,
            language_model_name=llm_name,
        ).on_conflict_do_update(
            index_elements=["set_id"],
            set_={
                "embedder_name": embedder_name,
                "language_model_name": llm_name,
            },
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            result.close()
            await session.commit()

    async def get_setid_config(
        self,
        *,
        set_id: SetIdT,
    ) -> SemanticConfigStorage.Config:
        stmt = (
            select(SetIdResources)
            .where(SetIdResources.set_id == set_id)
            .options(selectinload(SetIdResources.disabled_categories))
        )

        local_category_stmt = (
            select(Category)
            .where(Category.set_id == set_id)
            .options(selectinload(Category.tags))
        )

        async with self._create_session() as session:
            res_resources = await session.execute(stmt)
            resources = res_resources.scalar_one_or_none()

            res_local_categories = await session.execute(local_category_stmt)
            local_categories_raw = res_local_categories.scalars().unique().all()
            local_categories = [c.to_typed_model() for c in local_categories_raw]

            res_set_type = await session.execute(
                select(SetIdSetType.set_type_id).where(SetIdSetType.set_id == set_id)
            )
            set_type_id = res_set_type.scalar_one_or_none()

            inherited_categories: list[SemanticCategory] = []
            if set_type_id is not None:
                inherited_stmt = (
                    select(Category)
                    .where(Category.set_type_id == set_type_id)
                    .options(selectinload(Category.tags))
                )
                res_inherited = await session.execute(inherited_stmt)
                inherited_raw = res_inherited.scalars().unique().all()
                inherited_categories = [c.to_typed_model() for c in inherited_raw]

        local_by_name = {c.name: c for c in local_categories}
        inherited_filtered = [
            c for c in inherited_categories if c.name not in local_by_name
        ]
        categories = local_categories + inherited_filtered

        if resources is not None:
            llm_name = resources.language_model_name
            embedder_name = resources.embedder_name
            disabled_categories = [
                d.disabled_category for d in resources.disabled_categories
            ]
        else:
            llm_name = None
            embedder_name = None
            disabled_categories = None

        return SemanticConfigStorage.Config(
            llm_name=llm_name,
            embedder_name=embedder_name,
            categories=categories,
            disabled_categories=disabled_categories,
        )

    async def register_set_id_set_type(
        self,
        *,
        set_id: SetIdT,
        set_type_id: str,
    ) -> None:
        set_type_id_int = int(set_type_id)

        dialect_name = self._engine.dialect.name
        ins: PgInsert | SQliteInsert
        if dialect_name == "postgresql":
            ins = pg_insert(SetIdSetType)
        elif dialect_name == "sqlite":
            ins = sqlite_insert(SetIdSetType)
        else:
            raise NotImplementedError

        stmt = ins.values(
            set_id=set_id,
            set_type_id=set_type_id_int,
        ).on_conflict_do_nothing(
            index_elements=["set_id"],
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            result.close()
            await session.commit()

    async def create_set_type_category(
        self,
        *,
        set_type_id: str,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        set_type_id_int = int(set_type_id)

        stmt = (
            insert(Category)
            .values(
                name=category_name,
                prompt=prompt,
                set_type_id=set_type_id_int,
                description=description,
            )
            .returning(Category.id)
        )

        async with self._create_session() as session:
            res = await session.execute(stmt)
            category_id = res.scalar_one()
            await session.commit()

        return CategoryIdT(category_id)

    async def get_set_type_categories(
        self,
        *,
        set_type_id: str,
    ) -> list[SemanticCategory]:
        set_type_id_int = int(set_type_id)

        stmt = (
            select(Category)
            .where(Category.set_type_id == set_type_id_int)
            .options(selectinload(Category.tags))
        )

        async with self._create_session() as session:
            res = await session.execute(stmt)
            categories_raw = res.scalars().unique().all()

        return [c.to_typed_model() for c in categories_raw]

    async def get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> SemanticConfigStorage.Category | None:
        try:
            category_id_int = int(category_id)
        except (TypeError, ValueError):
            logger.exception("Error parsing category ID")
            return None

        stmt = select(Category).where(Category.id == category_id_int)

        async with self._create_session() as session:
            res = await session.execute(stmt)
            category = res.scalar_one_or_none()

        if category is None:
            return None

        return SemanticConfigStorage.Category(
            id=CategoryIdT(category.id),
            name=category.name,
            prompt=category.prompt,
            description=category.description,
        )

    async def get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> list[SetIdT]:
        category_id_int = int(category_id)

        async with self._create_session() as session:
            category_stmt = select(Category).where(Category.id == category_id_int)
            category_res = await session.execute(category_stmt)
            category = category_res.scalar_one_or_none()

            if category is None:
                return []
            if category.set_id is not None:
                return [SetIdT(category.set_id)]
            if category.set_type_id is None:
                return []

            set_ids_stmt = select(SetIdSetType.set_id).where(
                SetIdSetType.set_type_id == category.set_type_id
            )
            set_ids_res = await session.execute(set_ids_stmt)
            all_set_ids = [SetIdT(sid) for sid in set_ids_res.scalars().all()]

            overriding_set_ids_stmt = select(Category.set_id).where(
                Category.name == category.name,
                Category.set_id.in_(all_set_ids),
            )
            overriding_res = await session.execute(overriding_set_ids_stmt)
            overriding_set_ids = set(overriding_res.scalars().all())

            # Return set_ids that don't override the category
            return [sid for sid in all_set_ids if sid not in overriding_set_ids]

    async def create_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        prompt: str,
        description: str | None = None,
    ) -> CategoryIdT:
        stmt = (
            insert(Category)
            .values(
                name=category_name,
                prompt=prompt,
                set_id=set_id,
                description=description,
            )
            .returning(Category.id)
        )

        async with self._create_session() as session:
            res = await session.execute(stmt)
            category_id = res.scalar_one()
            await session.commit()

        return CategoryIdT(category_id)

    async def clone_category(
        self,
        *,
        category_id: CategoryIdT,
        new_set_id: SetIdT,
        new_name: str,
    ) -> CategoryIdT:
        category_id_int = int(category_id)

        async with self._create_session() as session:
            res = await session.execute(
                select(Category)
                .where(Category.id == category_id_int)
                .options(selectinload(Category.tags))
            )
            category = res.scalar_one()

            cloned_category = Category(
                name=new_name,
                prompt=category.prompt,
                description=category.description,
                set_id=new_set_id,
            )
            session.add(cloned_category)
            await session.flush()

            for tag in category.tags:
                session.add(
                    Tag(
                        name=tag.name,
                        description=tag.description,
                        category_id=cloned_category.id,
                    )
                )

            await session.commit()

            return CategoryIdT(cloned_category.id)

    async def delete_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None:
        category_id_int = int(category_id)

        async with self._create_session() as session:
            result = await session.execute(
                delete(Tag).where(Tag.category_id == category_id_int)
            )
            result.close()
            result = await session.execute(
                delete(Category).where(Category.id == category_id_int)
            )
            result.close()

            await session.commit()

    async def add_disabled_category_to_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        dialect_name = self._engine.dialect.name

        ins: PgInsert | SQliteInsert
        if dialect_name == "postgresql":
            ins = pg_insert(DisabledDefaultCategories)
        elif dialect_name == "sqlite":
            ins = sqlite_insert(DisabledDefaultCategories)
        else:
            raise NotImplementedError

        stmt = ins.values(
            set_id=set_id,
            disabled_category=category_name,
        ).on_conflict_do_nothing(
            index_elements=["set_id", "disabled_category"],
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            result.close()
            await session.commit()

    async def remove_disabled_category_from_setid(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        stmt = delete(DisabledDefaultCategories).where(
            DisabledDefaultCategories.set_id == set_id,
            DisabledDefaultCategories.disabled_category == category_name,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            result.close()
            await session.commit()

    async def get_tag(
        self,
        *,
        tag_id: str,
    ) -> SemanticConfigStorage.Tag | None:
        try:
            tag_id_int = int(tag_id)
        except (TypeError, ValueError):
            logger.exception("Error parsing tag ID")
            return None

        stmt = select(Tag).where(Tag.id == tag_id_int)

        async with self._create_session() as session:
            res = await session.execute(stmt)
            tag = res.scalar_one_or_none()

        if tag is None:
            return None

        return SemanticConfigStorage.Tag(
            id=str(tag_id_int),
            name=tag.name,
            description=tag.description,
        )

    async def add_tag(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        description: str,
    ) -> TagIdT:
        try:
            category_id_int = int(category_id)
        except (TypeError, ValueError) as e:
            raise ResourceNotFoundError(f"Invalid feature ID: {category_id}") from e

        tag_stmt = (
            insert(Tag)
            .values(
                name=tag_name,
                description=description,
                category_id=category_id_int,
            )
            .returning(Tag.id)
        )

        async with self._create_session() as session:
            res = await session.execute(tag_stmt)
            tag_id = res.scalar_one()
            await session.commit()

        return TagIdT(tag_id)

    async def update_tag(
        self,
        *,
        tag_id: str,
        tag_name: str,
        tag_description: str,
    ) -> None:
        tag_id_int = int(tag_id)

        stmt = (
            update(Tag)
            .where(Tag.id == tag_id_int)
            .values(name=tag_name, description=tag_description)
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            result.close()
            await session.commit()

    async def delete_tag(
        self,
        *,
        tag_id: str,
    ) -> None:
        tag_id_int = int(tag_id)

        async with self._create_session() as session:
            result = await session.execute(delete(Tag).where(Tag.id == tag_id_int))
            result.close()
            await session.commit()

    async def add_set_type_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        cleaned_tags = sorted({t.strip() for t in metadata_tags if t and t.strip()})

        assert all(_TAG_SEP not in t for t in cleaned_tags)

        tag_str = _TAG_SEP.join(cleaned_tags)

        stmt = (
            insert(SetType)
            .values(
                org_id=org_id,
                org_level_set=org_level_set,
                metadata_tags_sig=tag_str,
                name=name,
                description=description,
            )
            .returning(SetType.id)
        )

        async with self._create_session() as session:
            try:
                res = await session.execute(stmt)
                set_type_id = res.scalar_one()
                await session.commit()
            except IntegrityError:
                await session.rollback()

                existing_stmt = (
                    select(SetType.id)
                    .where(SetType.org_id == org_id)
                    .where(SetType.org_level_set == org_level_set)
                    .where(SetType.metadata_tags_sig == tag_str)
                )

                existing_res = await session.execute(existing_stmt)
                existing_id = existing_res.scalar_one_or_none()

                if existing_id is None:
                    raise

                return str(existing_id)

        return str(set_type_id)

    async def list_set_type_ids(self, *, org_id: str) -> list[SetTypeEntry]:
        stmt = select(SetType).where(SetType.org_id == org_id)

        async with self._create_session() as session:
            res = await session.execute(stmt)

            results = res.scalars().all()
            models = [r.to_typed_model() for r in results]

        return models

    async def delete_set_type_id(self, *, set_type_id: str) -> None:
        stmt = delete(SetType).where(SetType.id == int(set_type_id))

        async with self._create_session() as session:
            result = await session.execute(stmt)
            result.close()
            await session.commit()
