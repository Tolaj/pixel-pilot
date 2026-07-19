from abc import ABC, abstractmethod
from PIL import Image


class PerceptionProvider(ABC):
    """Every perception backend implements this interface."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def parse(self, image: Image.Image) -> list[dict]:
        ...

    def warm_up(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<PerceptionProvider name={self.name!r}>"
