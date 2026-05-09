# perception/providers/__init__.py
# Import all providers so their @register decorators fire automatically.
# To add a new provider: create a file here and add it to this list.

from . import omniparser  # noqa: F401
from . import ax  # noqa: F401
from . import moondream  # noqa: F401
