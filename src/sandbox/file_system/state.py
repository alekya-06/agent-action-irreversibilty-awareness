from dataclasses import dataclass, field
from copy import deepcopy
import json
import posixpath

def normalize_path(path: str) -> str:
    """Normalize a virtual, absolute POSIX path."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Paths must be absolute virtual paths")

    normalized = posixpath.normpath(path)

    if normalized == "/":
        return "/"

    if normalized.startswith("/../") or normalized == "/..":
        raise ValueError("Path escapes the virtual root")

    return normalized


@dataclass
class FileSystemState:
    # Directories include the root directory "/".
    directories: set[str] = field(default_factory=lambda: {"/"})

    # Canonical absolute path -> file contents.
    files: dict[str, str] = field(default_factory=dict)

    # Canonical absolute path -> metadata.
    metadata: dict[str, dict] = field(default_factory=dict)

    # Resource path -> paths referenced by that resource.
    dependencies: dict[str, list[str]] = field(default_factory=dict)

    def clone(self):
        return deepcopy(self)

    def canonical_key(self):
        """Stable representation for exact state comparison."""
        data = {
            "directories": sorted(self.directories),
            "files": dict(sorted(self.files.items())),
            "metadata": {
                path: self.metadata[path]
                for path in sorted(self.metadata)
            },
            "dependencies": {
                path: sorted(set(self.dependencies[path]))
                for path in sorted(self.dependencies)
            },
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def exactly_equals(self, other):
        return (isinstance(other, FileSystemState)
                and self.canonical_key() == other.canonical_key())

    def validate(self):
        """Check structural consistency of the virtual tree."""
        if "/" not in self.directories:
            raise ValueError("Root directory is missing")

        for path in self.directories:
            if normalize_path(path) != path:
                raise ValueError(f"Noncanonical directory: {path}")

        for path in self.files:
            if normalize_path(path) != path:
                raise ValueError(f"Noncanonical file: {path}")

            if path in self.directories:
                raise ValueError(f"Path is both file and directory: {path}")

        all_paths = self.directories | set(self.files)

        for path in all_paths:
            if path == "/":
                continue

            parent = posixpath.dirname(path)
            if parent not in self.directories:
                raise ValueError(f"Missing parent directory {parent} for {path}")

        for path in self.metadata:
            if path not in all_paths:
                raise ValueError(f"Metadata refers to missing path: {path}")

        for source, targets in self.dependencies.items():
            if source not in all_paths:
                raise ValueError(f"Dependency source does not exist: {source}")

            for target in targets:
                if normalize_path(target) != target:
                    raise ValueError( f"Noncanonical dependency target: {target}")

        return True


@dataclass
class EnvironmentState:
    filesystem: FileSystemState = field(
        default_factory=FileSystemState
    )

    # Snapshot ID -> saved filesystem state.
    snapshots: dict[str, FileSystemState] = field(default_factory=dict)

    # Action ID -> information required for a permitted undo.
    undo_records: dict[str, dict] = field(default_factory=dict)

