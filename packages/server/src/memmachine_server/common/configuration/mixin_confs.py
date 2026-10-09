"""Metrics configuration mixins."""

import os
import re
from datetime import timedelta
from enum import Enum
from typing import ClassVar, Self, cast

import yaml
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from memmachine_server.common.errors import InvalidPasswordError
from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)


class UnknownMetricsFactoryError(ValueError):
    """Raised when the metrics factory name is invalid."""


class WithMetricsFactory:
    """Runtime mixin that provides access to a metrics factory."""

    _factories: ClassVar[dict[str, MetricsFactory]] = {}

    # These are *protocol attributes* — provided by subclasses
    metrics_factory_id: str | None

    def get_metrics_factory(self) -> MetricsFactory:
        """Return the configured metrics factory instance."""
        factory_id = self.metrics_factory_id or "prometheus"

        if factory_id not in self._factories:
            match factory_id:
                case "prometheus":
                    self._factories[factory_id] = PrometheusMetricsFactory()
                case _:
                    raise UnknownMetricsFactoryError(
                        f"Unknown MetricsFactory name: {factory_id}"
                    )

        return self._factories[factory_id]


class MetricsFactoryIdMixin(WithMetricsFactory, BaseModel):
    """Pydantic mixin for configs that include a metrics factory ID."""

    metrics_factory_id: str | None = Field(
        default=None,
        description="Metrics factory ID for monitoring and metrics collection.",
    )


class _EnvironmentValue(str):
    """Resolved text retaining its original environment expression for YAML."""

    reference: str

    def __new__(cls, value: str, reference: str | None = None) -> Self:
        instance = super().__new__(cls, value)
        instance.reference = value if reference is None else reference
        return instance


class _EnvironmentSecret(SecretStr):
    """A runtime secret whose YAML representation remains an env reference."""

    reference: str

    def __init__(self, value: str, reference: str) -> None:
        super().__init__(value)
        self.reference = reference

    def __eq__(self, other: object) -> bool:
        """Preserve SecretStr value equality, including ordinary SecretStrs."""
        return (
            isinstance(other, SecretStr)
            and self.get_secret_value() == other.get_secret_value()
        )

    def __hash__(self) -> int:
        """Keep hashing consistent with SecretStr value equality."""
        return hash(self.get_secret_value())


class WithValueFromEnv:
    """Mixin that adds support for resolving environment variable references."""

    # Matches $ENV or ${ENV}
    _ENV_RE: ClassVar[re.Pattern] = re.compile(r"\$(\w+)|\$\{(\w+)}")

    @classmethod
    def _resolve_env(cls, value: object) -> object:
        """Resolve environment variable references in the form $ENV or ${ENV}."""
        if isinstance(value, _EnvironmentValue):
            return value
        if isinstance(value, _EnvironmentSecret):
            return _EnvironmentValue(value.get_secret_value(), value.reference)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()

        if not isinstance(value, str):
            return value

        def _repl(match: re.Match) -> str:
            # One of the groups will be None
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))

        if cls._ENV_RE.search(value):
            return _EnvironmentValue(cls._ENV_RE.sub(_repl, value), value)
        return value

    @classmethod
    def _resolve_secret(cls, value: object) -> object:
        """Resolve secrets without losing the source expression during validation."""
        resolved = cls._resolve_env(value)
        if isinstance(resolved, _EnvironmentValue):
            return _EnvironmentSecret(str(resolved), resolved.reference)
        return SecretStr(resolved) if isinstance(resolved, str) else resolved

    @classmethod
    def _validate_env_string(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> str:
        """Validate text while retaining provenance stripped by string validation."""
        parsed = handler(value)
        if not isinstance(parsed, str):
            raise TypeError("Environment-backed configuration text must be a string")
        source = value if isinstance(value, _EnvironmentValue) else parsed
        resolved = cls._resolve_env(source)
        if not isinstance(resolved, str):
            raise TypeError("Environment-backed configuration text must be a string")
        return resolved


class PasswordMixin(BaseModel, WithValueFromEnv):
    """
    Mixin for configurations that include a password.

    It reads the password from environment variables if user
    specifies a pattern like $ENV_NAME in the value.
    """

    password: SecretStr = Field(
        ...,
        description="Password for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )

    @field_validator("password", mode="before")
    @classmethod
    def resolve_password(cls, v: object) -> SecretStr:
        """Resolve environment variable references in the password."""
        v = cls._resolve_secret(v)
        if not isinstance(v, SecretStr):
            raise InvalidPasswordError("password must be a string or SecretStr")
        return v


class AWSCredentialsMixin(BaseModel, WithValueFromEnv):
    """
    Mixin for configurations that include AWS credentials.

    It reads the credentials from environment variables if user
    specifies a pattern like $ENV_NAME in the value.
    """

    aws_access_key_id: SecretStr | None = Field(
        default=None,
        description="AWS Access Key ID. It default to environment variable "
        "AWS_ACCESS_KEY_ID if not provided. Can reference an "
        "environment variable using $ENV_NAME syntax.",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=None,
        description="AWS Secret Access Key. It defaults to environment variable "
        "AWS_SECRET_ACCESS_KEY if not provided. Can reference an "
        "environment variable using $ENV_NAME syntax.",
    )
    aws_session_token: SecretStr | None = Field(
        default=None,
        description="AWS session token for authentication. It defaults to environment variable "
        "AWS_SESSION_TOKEN if not provided. Can reference an "
        "environment variable using $ENV_NAME syntax.",
    )

    @field_validator("aws_access_key_id", mode="before")
    @classmethod
    def resolve_aws_access_key_id(cls, v: object) -> object:
        """Resolve environment variable references in the AWS Access Key ID."""
        return cls._resolve_secret(v)

    @field_validator("aws_secret_access_key", mode="before")
    @classmethod
    def resolve_aws_secret_access_key(cls, v: object) -> object:
        """Resolve environment variable references in the AWS Secret Access Key."""
        return cls._resolve_secret(v)

    @field_validator("aws_session_token", mode="before")
    @classmethod
    def resolve_aws_session_token(cls, v: object) -> object:
        """Resolve environment variable references in the AWS Session Token."""
        return cls._resolve_secret(v)

    @model_validator(mode="after")
    def resolve_aws_env_defaults(self) -> Self:
        """Fill in AWS credentials from environment variables if not provided."""
        if not self.aws_access_key_id:
            v = os.getenv("AWS_ACCESS_KEY_ID", None)
            if v:
                self.aws_access_key_id = _EnvironmentSecret(v, "$AWS_ACCESS_KEY_ID")

        if not self.aws_secret_access_key:
            v = os.getenv("AWS_SECRET_ACCESS_KEY", None)
            if v:
                self.aws_secret_access_key = _EnvironmentSecret(
                    v, "$AWS_SECRET_ACCESS_KEY"
                )

        if not self.aws_session_token:
            v = os.getenv("AWS_SESSION_TOKEN", None)
            if v:
                self.aws_session_token = _EnvironmentSecret(v, "$AWS_SESSION_TOKEN")

        return self


class ApiKeyMixin(BaseModel, WithValueFromEnv):
    """
    Mixin for configurations that include an API key.

    It reads the API key from environment variables if user
    specifies a pattern like $ENV_NAME in the value.
    """

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: object) -> object:
        """Resolve environment variable references in the API key."""
        return cls._resolve_secret(v)


class YamlSerializableMixin(BaseModel):
    """Mixin that adds YAML-safe serialization for Pydantic models."""

    def to_yaml_dict(self) -> dict:
        raw = cast(YamlInputType, self.model_dump())

        def unwrap(obj: YamlInputType) -> YamlObjType:
            """Recursively unwrap Pydantic models, SecretStr, Enums, and drop empty values."""
            if isinstance(obj, YamlSerializableMixin):
                obj = obj.to_yaml_dict()

            if isinstance(obj, (_EnvironmentSecret, _EnvironmentValue)):
                obj = obj.reference
            elif isinstance(obj, SecretStr):
                obj = obj.get_secret_value()

            # Unwrap enums like SimilarityMetric
            if isinstance(obj, Enum):
                obj = obj.value

            if isinstance(obj, timedelta):
                obj = obj.total_seconds()

            # Dict — recurse & drop empty
            if isinstance(obj, dict):
                obj_dict = cast(dict[str, YamlInputType], obj)
                cleaned: dict[str, YamlObjType] = {
                    k: unwrap(v) for k, v in obj_dict.items()
                }
                # drop keys whose values are None/empty
                cleaned = {
                    k: v for k, v in cleaned.items() if v not in (None, "", [], {})
                }
                return cast(YamlObjType, cleaned)

            # List — recurse & drop empty
            if isinstance(obj, list):
                obj_list = cast(list[YamlInputType], obj)
                cleaned: list[YamlObjType] = [unwrap(v) for v in obj_list]
                cleaned = [v for v in cleaned if v not in (None, "", [], {})]
                return cast(YamlObjType, cleaned)

            # Base condition
            return cast(YamlObjType, obj)

        ret = unwrap(raw)
        if not isinstance(ret, dict):
            raise TypeError(
                "to_yaml_dict can only be called on models that serialize to dicts"
            )
        return ret

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_yaml_dict(), sort_keys=False)


type YamlObjType = (
    dict[str, "YamlObjType"] | list["YamlObjType"] | str | int | float | bool | None
)

type YamlInputType = (
    YamlSerializableMixin
    | SecretStr
    | Enum
    | timedelta
    | dict[str, "YamlInputType"]
    | list["YamlInputType"]
    | str
    | int
    | float
    | bool
    | None
)
