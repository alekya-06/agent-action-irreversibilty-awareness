from state import FileSystemState
from transitions import transition,is_descendant,path_exists
import posixpath


def get_permissible_actions(fs: FileSystemState) -> list[dict]:
    """
    Generate valid filesystem task actions for the given state.

    Supported actions:
        - delete_file
        - move_file
        - delete_directory
        - rename

    Recovery actions are handled separately in recovery.py.
    """
    candidates = []

    # --------------------------------------------------
    # 1. Delete files
    # --------------------------------------------------
    for path in sorted(fs.files):
        candidates.append({
            "type": "delete_file",
            "path": path,
        })

    # --------------------------------------------------
    # 2. Move files into existing directories
    # --------------------------------------------------
    for source in sorted(fs.files):
        filename = posixpath.basename(source)

        for directory in sorted(fs.directories):
            destination = posixpath.join(directory, filename)

            candidates.append({
                "type": "move_file",
                "source": source,
                "destination": destination,
            })

    # --------------------------------------------------
    # 3. Delete directories
    # --------------------------------------------------
    for directory in sorted(fs.directories):
        if directory == "/":
            continue

        descendants = {
            path
            for path in fs.directories | set(fs.files)
            if is_descendant(path, directory)
        }

        if not descendants:
            candidates.append({
                "type": "delete_directory",
                "path": directory,
                "recursive": False,
            })

        candidates.append({
            "type": "delete_directory",
            "path": directory,
            "recursive": True,
        })

    # --------------------------------------------------
    # 4. Rename files and directories
    # --------------------------------------------------
    rename_suffixes = ["renamed", "backup", "old"]

    for source in sorted(fs.directories | set(fs.files)):
        if source == "/":
            continue

        basename = posixpath.basename(source)
        parent = posixpath.dirname(source) or "/"

        # Generate a finite set of alternative names.
        candidate_names = {
            f"{basename}_{suffix}"
            for suffix in rename_suffixes
        }

        candidate_names.update(rename_suffixes)

        for new_name in sorted(candidate_names):
            if new_name == basename:
                continue

            destination = posixpath.join(parent, new_name)

            candidates.append({
                "type": "rename",
                "source": source,
                "new_name": new_name,
            })

    # --------------------------------------------------
    # Validate candidates through the transition engine
    # --------------------------------------------------
    permissible = []
    seen = set()

    for action in candidates:
        # Remove duplicate action descriptions.
        action_key = tuple(sorted(action.items()))

        if action_key in seen:
            continue

        seen.add(action_key)

        try:
            transition(fs, action)
        except (ValueError, KeyError, TypeError):
            continue

        permissible.append(action)

    return permissible
