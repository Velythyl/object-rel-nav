"""Stand-ins for the simulator modules upstream imports but never uses here.

``libs/common/utils.py`` imports ``habitat_sim`` (and
``habitat_sim.utils.common``), ``quaternion`` and ``curses`` at module level,
and every mapper, matcher and localizer module imports ``libs.common.utils``.
None of the functions this adapter reaches (``rle_to_mask``,
``mask_to_rle_numpy``, ``nodes2key``, ``change_edge_attr``,
``getSplitEdgeLists``, ``modify_graph``) touches them; they serve the Habitat
navigation loop only. So instead of shipping habitat-sim, each missing module
is registered as a placeholder whose every attribute raises, naming the module:
a code path that *does* need the simulator fails loudly instead of silently.

Real modules win: a stub is installed only when the import fails.
"""
from __future__ import annotations

import importlib
import sys
import types

#: module name -> attributes upstream imports by name (``from x import a, b``).
_STUBBED = {
    "habitat_sim": (),
    "habitat_sim.utils": (),
    "habitat_sim.utils.common": ("quat_to_magnum", "quat_from_magnum", "d3_40_colors_rgb"),
    "quaternion": (),
    "curses": (),
}

INSTALLED: list[str] = []


class _Missing:
    def __init__(self, module: str, name: str) -> None:
        self._where = f"{module}.{name}"

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._where} is a stub in the RAGMAP container (no simulator is installed); "
            "this code path needs habitat-sim."
        )

    def __getattr__(self, name: str):
        raise RuntimeError(f"{self._where}.{name} is a stub in the RAGMAP container.")


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Missing(self.__name__, name)


def install() -> list[str]:
    """Register a stub for every listed module that cannot be imported."""

    for name, exported in _STUBBED.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except ImportError:
            pass
        module = _StubModule(name)
        module.__path__ = []  # a package, so submodules resolve
        module.__dict__["__ragmap_stub__"] = True
        for attribute in exported:
            setattr(module, attribute, _Missing(name, attribute))
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, module)
        INSTALLED.append(name)
    return list(INSTALLED)
