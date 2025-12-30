"""SQLAlchemy-backed implementation of the semantic config storage."""

from sqlalchemy import (
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
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
)
from sqlalchemy.sql.sqltypes import Boolean, String

from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_model import (
    CategoryIdT,
    OrgSetIdEntry,
    SemanticCategory,
    SetIdT,
    StructuredSemanticPrompt,
    TagIdT,
)


class BaseSemanticConfigStore(DeclarativeBase):
    """Declarative base class for Semantic Config Store."""


class SetIdResources(BaseSemanticConfigStore):
    """Resource-level configuration associated with a set identifier."""

    __tablename__ = "semantic.config.setidresources"

    set_id = mapped_column(String, primary_key=True, nullable=False)
    embedder_name = mapped_column(String, nullable=True)
    language_model_name = mapped_column(String, nullable=True)

    disabled_categories: Mapped[list["DisabledDefaultCategories"]] = relationship(
        "DisabledDefaultCategories",
        cascade="all, delete-orphan",
        single_parent=True,
    )


class DisabledDefaultCategories(BaseSemanticConfigStore):
    """Default categories that are disabled for a given set."""

    __tablename__ = "semantic.config.setidresources.desabledcategories"

    set_id = mapped_column(
        String,
        ForeignKey(
            "semantic.config.setidresources.set_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    disabled_category = mapped_column(String, nullable=False, primary_key=True)


class Category(BaseSemanticConfigStore):
    """Semantic category definition with its prompt description."""

    __tablename__ = "semantic.config.category"

    id = mapped_column(Integer, primary_key=True, nullable=False)
    set_id = mapped_column(String, nullable=False, index=True)
    name = mapped_column(String, nullable=False)
    prompt_description = mapped_column(String, nullable=False)

    tags: Mapped[list["Tag"]] = relationship(
        "Tag",
        back_populates="category",
        cascade="all, delete-orphan",
        single_parent=True,
    )

    def to_typed_model(self) -> SemanticCategory:
        tags = {t.name: t.description for t in self.tags}

        return SemanticCategory(
            id=CategoryIdT(self.id),
            name=self.name,
            prompt=StructuredSemanticPrompt(
                description=self.prompt_description,
                tags=tags,
            ),
        )

    __table_args__ = (UniqueConstraint("set_id", "name", name="_set_id_name_uc"),)


class Tag(BaseSemanticConfigStore):
    """Individual tag that belongs to a semantic category."""

    __tablename__ = "semantic.config.tag"

    id = mapped_column(Integer, primary_key=True, nullable=False)
    name = mapped_column(String, nullable=False)
    description = mapped_column(String, nullable=False)

    category_id = mapped_column(
        Integer,
        ForeignKey("semantic.config.category.id", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[Category] = relationship(
        "Category",
        back_populates="tags",
    )


class OrgTagSet(BaseSemanticConfigStore):
    """Mapping of org-level metadata tags to set ids."""

    __tablename__ = "org_tag_set"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[str] = mapped_column(String, nullable=False)
    org_level_set: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    metadata_tags_sig: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "org_id", "org_level_set", "metadata_tags_sig", name="uq_org_level_tagsig"
        ),
    )

    def to_typed_model(self) -> OrgSetIdEntry:
        tags = self.metadata_tags_sig.split(_TAG_SEP)

        return OrgSetIdEntry(
            id=str(self.id),
            tags=tags,
            is_org_level=self.org_level_set,
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
            await session.execute(delete(SetIdResources))
            await session.execute(delete(Category))
            await session.execute(delete(Tag))
            await session.execute(delete(OrgTagSet))
            await session.execute(delete(DisabledDefaultCategories))

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
            await session.execute(stmt)
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

        category_stmt = (
            select(Category)
            .where(Category.set_id == set_id)
            .options(selectinload(Category.tags))
        )

        async with self._create_session() as session:
            res_resources = await session.execute(stmt)
            resources = res_resources.scalar_one_or_none()

            res_categories = await session.execute(category_stmt)
            categories_raw = res_categories.scalars().unique().all()

            categories = [c.to_typed_model() for c in categories_raw]

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

    async def create_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        description: str,
    ) -> CategoryIdT:
        stmt = (
            insert(Category)
            .values(
                name=category_name,
                prompt_description=description,
                set_id=set_id,
            )
            .returning(Category.id)
        )

        async with self._create_session() as session:
            res = await session.execute(stmt)
            await session.commit()
            category_id = res.scalar_one()

        return CategoryIdT(category_id)

    async def clone_category(
        self,
        *,
        category_id: CategoryIdT,
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
                prompt_description=category.prompt_description,
                set_id=category.set_id,
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
            await session.execute(delete(Tag).where(Tag.category_id == category_id_int))
            await session.execute(
                delete(Category).where(Category.id == category_id_int)
            )

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
            await session.execute(stmt)
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
            await session.execute(stmt)
            await session.commit()

    async def add_tag(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        description: str,
    ) -> TagIdT:
        category_id_int = int(category_id)

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

        return tag_id

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
            await session.execute(stmt)
            await session.commit()

    async def delete_tag(
        self,
        *,
        tag_id: str,
    ) -> None:
        tag_id_int = int(tag_id)

        async with self._create_session() as session:
            await session.execute(delete(Tag).where(Tag.id == tag_id_int))
            await session.commit()

    async def add_org_set_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
    ) -> str:
        assert len(metadata_tags) == len(set(metadata_tags)), (
            "metadata_tags must be unique"
        )

        cleaned_tags = sorted({t.strip() for t in metadata_tags if t and t.strip()})

        assert all(_TAG_SEP not in t for t in cleaned_tags)

        tag_str = _TAG_SEP.join(cleaned_tags)

        stmt = (
            insert(OrgTagSet)
            .values(
                org_id=org_id, org_level_set=org_level_set, metadata_tags_sig=tag_str
            )
            .returning(OrgTagSet.id)
        )

        async with self._create_session() as session:
            res = await session.execute(stmt)
            org_set_id = res.scalar_one()
            await session.commit()

        return str(org_set_id)

    async def list_org_set_ids(self, *, org_id: str) -> list[OrgSetIdEntry]:
        stmt = select(OrgTagSet).where(OrgTagSet.org_id == org_id)

        async with self._create_session() as session:
            res = await session.execute(stmt)

            results = res.scalars().all()
            models = [r.to_typed_model() for r in results]

        return models

    async def delete_org_set_id(self, *, org_set_id: str) -> None:
        stmt = delete(OrgTagSet).where(OrgTagSet.id == int(org_set_id))

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()
