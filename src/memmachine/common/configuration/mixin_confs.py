"""Metrics configuration mixins."""

import os
import re
from datetime import timedelta
from enum import Enum
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from memmachine.common.errors import InvalidPasswordError
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)


class UnknownMetricsFactoryError(ValueError):
    """Raised when the metrics factory name is invalid."""


class MetricsFactoryIdMixin(BaseModel):
    """Mixin for configurations that include a metrics factory ID."""

    metrics_factory_id: str | None = Field(
        default=None,
        description="Metrics factory ID for monitoring and metrics collection.",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined labels for metrics.",
    )

    _factories: ClassVar[dict[str, MetricsFactory]] = {}

    def get_metrics_factory(self) -> MetricsFactory | None:
        """Return the configured metrics factory instance, if any."""
        factory_id = self.metrics_factory_id
        if factory_id is None:
            factory_id = "prometheus"
        if factory_id not in self._factories:
            match factory_id:
                case "prometheus":
                    factory = PrometheusMetricsFactory()
                    self._factories[factory_id] = factory
                case _:
                    raise UnknownMetricsFactoryError(
                        f"Unknown MetricsFactory name: {factory_id}"
                    )
        return self._factories[factory_id]


class WithValueFromEnv:
    """Mixin that adds support for resolving environment variable references."""

    # Matches $ENV or ${ENV}
    _ENV_RE: ClassVar[re.Pattern] = re.compile(r"\$(\w+)|\$\{(\w+)}")

    @classmethod
    def _resolve_env(cls, value: SecretStr | str) -> str:
        """Resolve environment variable references in the form $ENV or ${ENV}."""
        if isinstance(value, SecretStr):
            value = value.get_secret_value()

        if not isinstance(value, str):
            return value

        def _repl(match: re.Match) -> str:
            # One of the groups will be None
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))

        return cls._ENV_RE.sub(_repl, value)


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
    def resolve_password(cls, v: str | SecretStr) -> SecretStr | str | None:
        """Resolve environment variable references in the password."""
        v = cls._resolve_env(v)
        if not isinstance(v, str):
            raise InvalidPasswordError("password must be a string or SecretStr")
        return SecretStr(v) if isinstance(v, str) else v


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
    def resolve_aws_access_key_id(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the AWS Access Key ID."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("aws_secret_access_key", mode="before")
    @classmethod
    def resolve_aws_secret_access_key(
        cls, v: SecretStr | str
    ) -> SecretStr | str | None:
        """Resolve environment variable references in the AWS Secret Access Key."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("aws_session_token", mode="before")
    @classmethod
    def resolve_aws_session_token(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the AWS Session Token."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def resolve_aws_env_defaults(self) -> Self:
        """Fill in AWS credentials from environment variables if not provided."""
        if not self.aws_access_key_id:
            v = os.getenv("AWS_ACCESS_KEY_ID", None)
            if v:
                self.aws_access_key_id = SecretStr(v)

        if not self.aws_secret_access_key:
            v = os.getenv("AWS_SECRET_ACCESS_KEY", None)
            if v:
                self.aws_secret_access_key = SecretStr(v)

        if not self.aws_session_token:
            v = os.getenv("AWS_SESSION_TOKEN", None)
            if v:
                self.aws_session_token = SecretStr(v)

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
    def resolve_api_key(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the API key."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v


class YamlSerializableMixin(BaseModel):
    """Mixin that adds YAML-safe serialization for Pydantic models."""

    def to_yaml_dict(self) -> dict:
        raw = self.model_dump()

        def unwrap(obj: YamlObjType) -> YamlObjType:
            """Recursively unwrap Pydantic models, SecretStr, Enums, and drop empty values."""
            if isinstance(obj, YamlSerializableMixin):
                obj = obj.to_yaml_dict()

            # Unwrap SecretStr
            if isinstance(obj, SecretStr):
                obj = obj.get_secret_value()

            # Unwrap enums like SimilarityMetric
            if isinstance(obj, Enum):
                obj = obj.value

            if isinstance(obj, timedelta):
                obj = obj.total_seconds()

            # Dict — recurse & drop empty
            if isinstance(obj, dict):
                cleaned = {k: unwrap(v) for k, v in obj.items()}
                # drop keys whose values are None/empty
                cleaned = {
                    k: v for k, v in cleaned.items() if v not in (None, "", [], {})
                }
                return cleaned

            # List — recurse & drop empty
            if isinstance(obj, list):
                cleaned = [unwrap(v) for v in obj]
                cleaned = [v for v in cleaned if v not in (None, "", [], {})]
                return cleaned

            # Base condition
            return obj

        ret = unwrap(raw)
        if not isinstance(ret, dict):
            raise TypeError(
                "to_yaml_dict can only be called on models that serialize to dicts"
            )
        return ret

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_yaml_dict(), sort_keys=False)


type YamlObjType = (
    YamlSerializableMixin
    | SecretStr
    | Enum
    | dict[str, "YamlObjType"]
    | list["YamlObjType"]
    | str
    | int
    | float
    | bool
    | None
)
