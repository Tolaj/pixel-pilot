# perception/base.py
from abc import ABC, abstractmethod
from PIL import Image


class PerceptionProvider(ABC):
    """
    Every perception backend implements this interface.
    parse() is the only required method.
    """

    name: str = "base"  # unique identifier, used for selection + logging
    description: str = ""  # one-line description shown in --list-providers

    @abstractmethod
    def parse(self, image: Image.Image) -> list[dict]:
        """
        Analyze a screenshot and return a list of UI elements.

        Each element must have at least:
            id      int       sequential index
            label   str       human-readable name / caption
            center  [x, y]    click target in screen coordinates
            bbox    [x1,y1,x2,y2]
            conf    float     detection confidence (1.0 = ground truth)
            source  str       provider name

        Extra keys (role, value, etc.) are allowed and passed through.
        """
        ...

    def warm_up(self) -> None:
        """
        Optional: pre-load models so first parse() call isn't slow.
        Called by the registry on startup if warm_up=True.
        """

    def teardown(self) -> None:
        """Optional: release GPU memory / close handles."""

    def __repr__(self) -> str:
        return f"<PerceptionProvider name={self.name!r}>"
