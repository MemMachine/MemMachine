"""Abstract interfaces for creating and managing metrics."""

from abc import ABC, abstractmethod
from collections.abc import Iterable


class MetricsFactory(ABC):
    """Abstract base class for a metrics factory."""

    class Counter(ABC):
        """Abstract base class for a counter metric."""

        @abstractmethod
        def increment(
            self,
            value: float = 1,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Increment the counter by the provided value."""
            raise NotImplementedError

    class Gauge(ABC):
        """Abstract base class for a gauge metric."""

        @abstractmethod
        def set(
            self,
            value: float,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Set the gauge to a specified value."""
            raise NotImplementedError

    class Histogram(ABC):
        """Abstract base class for a histogram metric."""

        @abstractmethod
        def observe(
            self,
            value: float,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Record a histogram observation."""
            raise NotImplementedError

    class Summary(ABC):
        """Abstract base class for a summary metric."""

        @abstractmethod
        def observe(
            self,
            value: float,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Record a summary observation."""
            raise NotImplementedError

    @abstractmethod
    def get_counter(
        self,
        name: str,
        description: str,
        label_names: Iterable[str] = (),
    ) -> Counter:
        """Return a counter metric by name, creating it if absent."""
        raise NotImplementedError

    @abstractmethod
    def get_gauge(
        self,
        name: str,
        description: str,
        label_names: Iterable[str] = (),
    ) -> Gauge:
        """Return a gauge metric by name, creating it if absent."""
        raise NotImplementedError

    @abstractmethod
    def get_histogram(
        self,
        name: str,
        description: str,
        label_names: Iterable[str] = (),
    ) -> Histogram:
        """Return a histogram metric by name, creating it if absent."""
        raise NotImplementedError

    @abstractmethod
    def get_summary(
        self,
        name: str,
        description: str,
        label_names: Iterable[str] = (),
    ) -> Summary:
        """Return a summary metric by name, creating it if absent."""
        raise NotImplementedError
