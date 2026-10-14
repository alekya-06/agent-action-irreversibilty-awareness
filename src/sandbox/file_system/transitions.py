from copy import deepcopy
import posixpath

from state import FileSystemState, normalize_path


def is_descendant(path: str, parent: str) -> bool:
    """Check whether path is strictly inside parent."""
    return path.startswith(parent.rstrip("/") + "/")


def path_exists(fs: FileSystemState, path: str) -> bool:
    """Check whether a file or directory exists."""
    return path in fs.files or path in fs.directories

def replace_path_prefix(path, old_prefix, new_prefix):
    if path == old_prefix:
        return new_prefix

    if is_descendant(path, old_prefix):
        suffix = path[len(old_prefix):]
        return new_prefix + suffix

    return path

def require_parent_directory(fs: FileSystemState, path: str) -> None:
    """Ensure the destination's parent directory exists."""
    parent = posixpath.dirname(path) or "/"

    if parent not in fs.directories:
        raise ValueError(f"Parent directory does not exist: {parent}")


def transition(fs: FileSystemState, action: dict):
    """
    Apply a filesystem action to a cloned state.

    Returns:
        (new_filesystem_state, transition_record)

    The original state is never mutated.
    """
    new_fs = fs.clone()
    before = fs.canonical_key()

    action_type = action["type"]

    # --------------------------------------------------
    # 1. Delete file
    # --------------------------------------------------
    if action_type == "delete_file":
        path = normalize_path(action["path"])

        if path not in new_fs.files:
            raise ValueError(f"Not a file: {path}")

        del new_fs.files[path]
        new_fs.metadata.pop(path, None)
        new_fs.dependencies.pop(path, None)

        # References from other resources remain unchanged.
        # They may now point to a nonexistent file.

    # --------------------------------------------------
    # 2. Move file
    # --------------------------------------------------
    elif action_type == "move_file":
        source = normalize_path(action["source"])
        destination = normalize_path(action["destination"])

        if source not in new_fs.files:
            raise ValueError(f"Not a file: {source}")

        if path_exists(new_fs, destination):
            raise ValueError(
                f"Destination already exists: {destination}"
            )

        require_parent_directory(new_fs, destination)

        new_fs.files[destination] = new_fs.files.pop(source)

        if source in new_fs.metadata:
            new_fs.metadata[destination] = (
                new_fs.metadata.pop(source)
            )

        if source in new_fs.dependencies:
            new_fs.dependencies[destination] = (
                new_fs.dependencies.pop(source)
            )

        # Update references pointing to the moved file.
        for targets in new_fs.dependencies.values():
            for i, target in enumerate(targets):
                if target == source:
                    targets[i] = destination

    # --------------------------------------------------
    # 3. Delete directory
    # --------------------------------------------------
    elif action_type == "delete_directory":
        path = normalize_path(action["path"])
        recursive = action.get("recursive", False)

        if path == "/":
            raise ValueError(
                "Cannot delete the virtual root"
            )

        if path not in new_fs.directories:
            raise ValueError(f"Not a directory: {path}")

        descendants = {
            p
            for p in new_fs.directories | set(new_fs.files)
            if is_descendant(p, path)
        }

        if descendants and not recursive:
            raise ValueError("Directory is not empty")

        removed_paths = descendants | {path}

        # Remove files, directories and their metadata.
        for p in descendants:
            new_fs.files.pop(p, None)
            new_fs.directories.discard(p)
            new_fs.metadata.pop(p, None)
            new_fs.dependencies.pop(p, None)

        new_fs.directories.discard(path)
        new_fs.metadata.pop(path, None)
        new_fs.dependencies.pop(path, None)

        # References from surviving resources remain intact.

    # --------------------------------------------------
    # 4. Rename file or directory
    # --------------------------------------------------
    elif action_type == "rename":
        source = normalize_path(action["source"])
        new_name = action["new_name"]

        if not isinstance(new_name, str):
            raise ValueError("new_name must be a string")

        if (
            not new_name
            or new_name in {".", ".."}
            or "/" in new_name
        ):
            raise ValueError(
                "new_name must be a single path component"
            )

        if source == "/":
            raise ValueError(
                "Cannot rename the virtual root"
            )

        if not path_exists(new_fs, source):
            raise ValueError(
                f"Source does not exist: {source}"
            )

        parent = posixpath.dirname(source) or "/"
        destination = posixpath.join(parent, new_name)

        if path_exists(new_fs, destination):
            raise ValueError(
                f"Destination already exists: {destination}"
            )

        # Collect the source and every descendant if this is
        # a directory rename.
        if source in new_fs.directories:
            affected = {
                p
                for p in (
                    new_fs.directories | set(new_fs.files)
                )
                if p == source or is_descendant(p, source)
            }
        else:
            affected = {source}

        # Build all path replacements before changing anything.
        replacements = {
            old: destination + old[len(source):]
            for old in affected
        }

        # Check for collisions with unaffected paths.
        existing_paths = (
            new_fs.directories | set(new_fs.files)
        )

        for old, new in replacements.items():
            if new in existing_paths and new not in affected:
                raise ValueError(
                    f"Destination collision: {new}"
                )

        # Rebuild files and directories using the new paths.
        new_fs.files = {
            replacements.get(path, path): content
            for path, content in new_fs.files.items()
        }

        new_fs.directories = {
            replacements.get(path, path)
            for path in new_fs.directories
        }

        # Remap metadata and dependency source paths.
        new_fs.metadata = {
            replacements.get(path, path): value
            for path, value in new_fs.metadata.items()
        }

        new_fs.dependencies = {
            replacements.get(path, path): targets
            for path, targets in new_fs.dependencies.items()
        }

        # Update dependency targets that reference renamed paths.
        for targets in new_fs.dependencies.values():
            for i, target in enumerate(targets):
                if target in replacements:
                    targets[i] = replacements[target]

    else:
        raise ValueError(
            f"Unknown action type: {action_type}"
        )

    # Reject any invalid resulting filesystem.
    new_fs.validate()

    record = {
        "action": deepcopy(action),
        "before": before,
        "after": new_fs.canonical_key(),
    }

    return new_fs, record
