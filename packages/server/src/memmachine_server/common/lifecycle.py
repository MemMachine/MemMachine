"""Starting and stopping a component, which its owner does."""

from abc import abstractmethod
from typing import Protocol


class Lifecycle(Protocol):
    """
    A component its owner starts before using it and stops after.

    The owner is whatever constructed the component. It calls `start` once,
    before any other method except `prepare`, and `stop` once, after the last
    use. A component never starts or stops a component it was given; that
    component's owner does.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Start the component on the durable state its resources hold.

        Reads and checks durable state, and builds in-process state from it.
        Creates, changes, and deletes nothing durable, so it is safe while any
        number of components over the same resources, in this process or
        others, start or serve. A start that raises leaves nothing to stop:
        the owner neither uses nor stops the component.
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Release what the component holds in this process, writing nothing durable."""
        raise NotImplementedError


class Preparable(Protocol):
    """A component whose durable resources the setup step prepares."""

    @abstractmethod
    async def prepare(self) -> None:
        """
        Create the component's durable resources, starting from none.

        The setup step calls it in one process, while no other process uses
        the resources, before any component over them starts. Run again on
        what it created, it changes nothing. It promises nothing for any other
        state.
        """
        raise NotImplementedError
