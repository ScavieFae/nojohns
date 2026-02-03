"""
nojohns/registry.py - Fighter discovery and loading

Two sources:
  1. Built-ins (do-nothing, random) — registered at import time
  2. Local manifests (fighters/*/fighter.toml) — discovered via scan
"""

import importlib
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class FighterInfo:
    """Lightweight metadata from a fighter manifest or built-in registration.

    No melee import needed — characters are string names, not enums.
    """

    name: str
    version: str = "0.0.0"
    display_name: str = ""
    author: str = ""
    description: str = ""
    fighter_type: str = "rule-based"
    characters: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    avg_frame_delay: int = 0
    repo_url: str | None = None
    weights_required: bool = False

    # Loading
    entry_point: str | None = None  # "module.path:ClassName"
    source: str = "builtin"  # "builtin" | "manifest"
    manifest_path: Path | None = None
    fighter_dir: Path | None = None
    init_args: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    configurable: dict[str, Any] = field(default_factory=dict)


class FighterNotFoundError(KeyError):
    """Raised when a fighter name is not in the registry."""


class FighterLoadError(RuntimeError):
    """Raised when a fighter is found but can't be instantiated."""


# ============================================================================
# Registry State
# ============================================================================

_fighters: dict[str, FighterInfo] = {}
_builtin_classes: dict[str, type] = {}
_scanned: bool = False


# ============================================================================
# Public API
# ============================================================================


def register_builtin(name: str, fighter_class: type) -> None:
    """Register a built-in fighter class (no manifest needed).

    Instantiates the class once to extract metadata for FighterInfo,
    then discards the instance.
    """
    _builtin_classes[name] = fighter_class

    # Pull metadata from an instance so list-fighters/info work
    try:
        instance = fighter_class()
        meta = instance.metadata
        _fighters[name] = FighterInfo(
            name=meta.name,
            version=meta.version,
            display_name=meta.display_name,
            author=meta.author,
            description=meta.description,
            fighter_type=meta.fighter_type,
            characters=[c.name for c in meta.characters],
            stages=[s.name for s in meta.stages],
            avg_frame_delay=meta.avg_frame_delay,
            repo_url=meta.repo_url,
            weights_required=meta.weights_required,
            source="builtin",
            entry_point=f"{fighter_class.__module__}:{fighter_class.__qualname__}",
            hardware={
                "gpu_required": meta.gpu_required,
                "min_ram_gb": meta.min_ram_gb,
            },
        )
    except Exception:
        # Fallback if metadata extraction fails (shouldn't happen for built-ins)
        _fighters[name] = FighterInfo(
            name=name,
            source="builtin",
            entry_point=f"{fighter_class.__module__}:{fighter_class.__qualname__}",
        )


def list_fighters() -> list[FighterInfo]:
    """Return info for all registered fighters."""
    _ensure_scanned()
    return list(_fighters.values())


def get_fighter_info(name: str) -> FighterInfo | None:
    """Get info for a fighter without instantiating it. Returns None if not found."""
    _ensure_scanned()
    return _fighters.get(name)


def load_fighter(name: str) -> Any:
    """Instantiate a fighter by name.

    Returns a Fighter instance.

    Raises:
        FighterNotFoundError: name not in registry
        FighterLoadError: found but can't instantiate
    """
    _ensure_scanned()

    info = _fighters.get(name)
    if info is None:
        available = ", ".join(sorted(_fighters.keys()))
        raise FighterNotFoundError(f"Unknown fighter: {name!r}. Available: {available}")

    # Built-in: direct class instantiation
    if name in _builtin_classes:
        return _builtin_classes[name]()

    # Manifest-based: import via entry_point
    if not info.entry_point:
        raise FighterLoadError(f"Fighter {name!r} has no entry_point")

    cls = _import_entry_point(info.entry_point, name)
    kwargs = _resolve_init_args(info)

    try:
        return cls(**kwargs)
    except Exception as e:
        raise FighterLoadError(
            f"Failed to instantiate {name!r} ({info.entry_point}): {e}"
        ) from e


def scan_fighters(fighters_dir: Path | None = None) -> None:
    """Scan for fighter.toml manifests and register them.

    If fighters_dir is None, scans the default fighters/ directory
    relative to the project root.
    """
    global _scanned

    if fighters_dir is None:
        # Default: project_root/fighters/
        fighters_dir = Path(__file__).resolve().parent.parent / "fighters"

    if not fighters_dir.is_dir():
        logger.debug("Fighters directory not found: %s", fighters_dir)
        _scanned = True
        return

    for toml_path in sorted(fighters_dir.glob("*/fighter.toml")):
        try:
            info = _parse_manifest(toml_path)
        except Exception as e:
            logger.warning("Skipping bad manifest %s: %s", toml_path, e)
            continue

        if info.name in _fighters:
            existing = _fighters[info.name]
            if existing.source == "builtin":
                logger.warning(
                    "Fighter %r from %s shadows built-in — keeping built-in",
                    info.name,
                    toml_path,
                )
                continue

        _fighters[info.name] = info

    _scanned = True


def reset() -> None:
    """Clear all state. Intended for tests only."""
    global _scanned
    _fighters.clear()
    _builtin_classes.clear()
    _scanned = False
    _register_builtins()


# ============================================================================
# Internal
# ============================================================================


def _ensure_scanned() -> None:
    """Lazy-scan on first access."""
    if not _scanned:
        scan_fighters()


def _parse_manifest(toml_path: Path) -> FighterInfo:
    """Parse a fighter.toml into a FighterInfo."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    name = data.get("name")
    if not name:
        raise ValueError(f"Missing required field 'name' in {toml_path}")

    entry_point = data.get("entry_point")
    if not entry_point:
        raise ValueError(f"Missing required field 'entry_point' in {toml_path}")

    fighter_dir = toml_path.parent

    hardware = data.get("hardware", {})

    return FighterInfo(
        name=name,
        version=data.get("version", "0.0.0"),
        display_name=data.get("display_name", name),
        author=data.get("author", ""),
        description=data.get("description", ""),
        fighter_type=data.get("fighter_type", "rule-based"),
        characters=data.get("characters", []),
        stages=data.get("stages", []),
        avg_frame_delay=data.get("avg_frame_delay", 0),
        repo_url=data.get("repo_url"),
        weights_required=data.get("weights_required", False),
        entry_point=entry_point,
        source="manifest",
        manifest_path=toml_path,
        fighter_dir=fighter_dir,
        init_args=data.get("init_args", {}),
        hardware=hardware,
        configurable=data.get("configurable", {}),
    )


def _resolve_init_args(info: FighterInfo) -> dict[str, Any]:
    """Resolve {fighter_dir} placeholders in init_args."""
    if not info.init_args:
        return {}

    fighter_dir = str(info.fighter_dir) if info.fighter_dir else ""
    resolved = {}
    for key, value in info.init_args.items():
        if isinstance(value, str):
            resolved[key] = value.replace("{fighter_dir}", fighter_dir)
        else:
            resolved[key] = value
    return resolved


def _import_entry_point(entry_point: str, fighter_name: str) -> type:
    """Import a class from 'module.path:ClassName' string."""
    if ":" not in entry_point:
        raise FighterLoadError(
            f"Invalid entry_point for {fighter_name!r}: {entry_point!r} "
            f"(expected 'module.path:ClassName')"
        )

    module_path, class_name = entry_point.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise FighterLoadError(
            f"Cannot import module {module_path!r} for fighter {fighter_name!r}: {e}"
        ) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        raise FighterLoadError(
            f"Module {module_path!r} has no attribute {class_name!r} "
            f"(fighter {fighter_name!r})"
        )

    return cls


def _register_builtins() -> None:
    """Register built-in fighters. Called at module load."""
    from nojohns.fighter import DoNothingFighter, RandomFighter

    register_builtin("do-nothing", DoNothingFighter)
    register_builtin("random", RandomFighter)


# Register built-ins on import
_register_builtins()
