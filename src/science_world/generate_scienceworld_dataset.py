"""
ScienceWorld reversibility labelling, v2.

Each (state, action) pair is labelled by searching for a recovery path of at
most MAX_HORIZON actions that restores the world state from just before the
action. This follows the bounded definition in the project brief (Section 4.2):

    reversible            exact state restored within k actions
    partially_reversible  state restored only approximately within k actions
                          (everything matches except temperatures)
    irreversible          the search was exhausted to depth k without any match
    inconclusive          the node budget ran out before depth k was exhausted

One search per example records the minimum exact and approximate recovery
costs, so labels for k = 1, 2, 3 come from the same run.

Fixes compared with v1
----------------------
1. Objects are matched by uuid within one session. The tree keys are
   "name-uuid" and the uuids change on every env.load(), so v1's comparison
   after a reload could never succeed. v2 loads each task once and replays
   with env.reset(), which keeps uuids stable and is about 2x faster.
2. Recovery actions come from the environment's valid actions at each node,
   filtered to objects that still differ from the target state (plus their
   original containers). v1 used only gold-path actions and the first 10 of
   them.
3. Inverses are correct: a picked-up object goes back to its original
   container ("move X to <container>"), not to the floor.
4. Background time dynamics (ambient temperature drift, things heating on an
   active stove) are measured with a wait baseline. Objects that change on
   their own are excluded from the temperature check, so ongoing processes
   do not count against recovery.
5. The episode keeps running after done=True (verified for 1.2.3), so task
   completion no longer blocks the physical recovery check. "focus on X"
   commits the episode's answer and is labelled irreversible by rule.
6. A failed search to depth k now gives "irreversible" when the search was
   exhaustive. "inconclusive" is used only when the budget ran out.
7. Random valid actions are sampled alongside gold actions to balance
   classes. Task names are checked against the installed version.
8. The output includes state_text, action_text and label (the format
   required in Section 4.3), and runs in parallel across tasks with resume
   support.

Tested with scienceworld 1.2.3 and Java 21.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv


# =========================================================
# CONFIGURATION
# =========================================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    PROJECT_ROOT = Path.cwd()

OUTPUT_DIR = PROJECT_ROOT / "data" / "scienceworld"
OUTPUT_FILE = OUTPUT_DIR / "scienceworld_reversibility_v2.jsonl"

# Valid task names in scienceworld 1.2.3 (env.get_task_names()).
TASK_NAMES = [
    "boil",
    "melt",
    "freeze",
    "change-the-state-of-matter-of",
    "use-thermometer",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
    "power-component",
    "power-component-renewable-vs-nonrenewable-energy",
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
    "find-living-thing",
    "find-non-living-thing",
    "find-plant",
    "find-animal",
    "grow-plant",
    "grow-fruit",
    "chemistry-mix",
    "chemistry-mix-paint-secondary-color",
    "chemistry-mix-paint-tertiary-color",
    "lifespan-longest-lived",
    "lifespan-shortest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "identify-life-stages-1",
    "identify-life-stages-2",
    "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces",
    "inclined-plane-friction-unnamed-surfaces",
    "mendelian-genetics-known-plant",
    "mendelian-genetics-unknown-plant",
]

VARIATIONS_PER_TASK = 2          # first N training variations of each task
ENV_STEP_LIMIT = 10_000          # well above prefix + recovery length

MAX_HORIZON = 3                  # labels are produced for k = 1..MAX_HORIZON
MAX_NODES_PER_EXAMPLE = 400      # search budget; hitting it gives "inconclusive"
MAX_BRANCHING = 40               # beyond this the level is not called exhaustive

TEMP_TOLERANCE_C = 2.0           # exact match allows this temperature difference

SAMPLE_POSITIONS = [2, 5, 8, 11, 14, 17, 20, 25]
RANDOM_ACTIONS_PER_POSITION = 1  # extra non-gold candidates per position
RANDOM_ACTION_TRIES = 12         # attempts to find a random action with an effect
SKIP_NO_EFFECT_ACTIONS = True    # skip look/inventory/read-style gold actions

SAVE_TREES = False               # store full pre/post object trees (large)
SEED = 42

# Field names containing these substrings are counters that tick with time.
VOLATILE_KEY_SUBSTRINGS = ("tick",)

NO_EFFECT_PREFIXES = (
    "look around", "look at ", "look in ", "inventory", "task", "read ",
)
COMMIT_PREFIXES = ("focus on ",)
NEVER_SAMPLE_PREFIXES = (
    "focus on ", "reset task", "wait", "look", "inventory", "task", "read ",
)

# Verbs allowed inside a recovery path.
RECOVERY_PREFIXES = (
    "open ", "close ", "activate ", "deactivate ", "connect ", "disconnect ",
    "pick up ", "put down ", "move ", "pour ", "go ",
)
# Verbs that place obj0 into obj1. For these, obj1 must be a place where
# some still-different object lived in the target state.
PLACEMENT_PREFIXES = ("move ", "pour ")


def normalise(action: str) -> str:
    return " ".join(str(action).lower().split())


# Two-word verbs first so "pick up" is not read as "pick".
_MULTIWORD_VERBS = ("pick up", "put down", "look around", "look at",
                    "look in", "focus on", "reset task")


def action_verb(action: str) -> str:
    for verb in _MULTIWORD_VERBS:
        if action.startswith(verb):
            return verb
    return action.split(" ", 1)[0]


# =========================================================
# WORLD STATE: FLATTENING AND COMPARISON
# =========================================================

@dataclass
class ObjRecord:
    name: str
    parent: str | None
    fields: dict[str, Any]          # everything except temperature
    temperature: float | None


def _is_volatile(key: str) -> bool:
    lower = key.lower()
    return any(sub in lower for sub in VOLATILE_KEY_SUBSTRINGS)


def flatten_tree(tree: dict[str, Any]) -> dict[str, ObjRecord]:
    """Map uuid -> ObjRecord for every object in the tree."""
    flat: dict[str, ObjRecord] = {}

    def walk(node: dict[str, Any], parent: str | None) -> None:
        uuid = str(node["uuid"])
        fields: dict[str, Any] = {
            "name": node.get("name"),
            "type": node.get("type"),
            "isDeleted": node.get("isDeleted"),
        }
        temperature = None

        for key, value in node.items():
            if not key.startswith("prop") or not isinstance(value, dict):
                continue
            for sub_key, sub_value in value.items():
                if _is_volatile(sub_key):
                    continue
                if sub_key == "temperatureC":
                    temperature = sub_value
                    continue
                if isinstance(sub_value, (list, dict)):
                    sub_value = json.dumps(sub_value, sort_keys=True)
                fields[f"{key}.{sub_key}"] = sub_value

        flat[uuid] = ObjRecord(
            name=str(node.get("name")),
            parent=parent,
            fields=fields,
            temperature=temperature,
        )

        for child in (node.get("contents") or {}).values():
            walk(child, uuid)

    walk(tree, None)
    return flat


@dataclass
class Dynamics:
    """What changes in the world when time passes and the agent does nothing."""
    dynamic_uuids: set[str] = field(default_factory=set)
    spontaneous_new_names: set[str] = field(default_factory=set)


def temperature_differs(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a != b
    return abs(a - b) > TEMP_TOLERANCE_C


def compute_dynamics(
    pre: dict[str, ObjRecord],
    waited: dict[str, ObjRecord],
) -> Dynamics:
    dyn = Dynamics()

    for uuid, rec in pre.items():
        other = waited.get(uuid)
        if (
            other is None
            or other.parent != rec.parent
            or other.fields != rec.fields
            or temperature_differs(other.temperature, rec.temperature)
        ):
            dyn.dynamic_uuids.add(uuid)

    for uuid, rec in waited.items():
        if uuid not in pre:
            dyn.spontaneous_new_names.add(rec.name)

    return dyn


@dataclass
class Match:
    exact: bool
    approximate: bool
    differing: set[str]             # uuids present in both that differ
    missing: set[str]               # in target, not in current
    new: set[str]                   # in current, not in target


def compare_states(
    current: dict[str, ObjRecord],
    target: dict[str, ObjRecord],
    dyn: Dynamics,
) -> Match:
    """
    Approximate match: same objects, same containment, same non-temperature
    fields. Exact match: approximate match plus temperatures within
    TEMP_TOLERANCE_C for every object that does not change on its own.
    For dynamic objects only containment and existence are compared.
    """
    differing: set[str] = set()
    temp_only: set[str] = set()
    missing: set[str] = set()
    new: set[str] = set()

    for uuid, tgt in target.items():
        cur = current.get(uuid)

        if cur is None:
            if uuid not in dyn.dynamic_uuids:
                missing.add(uuid)
            continue

        if cur.parent != tgt.parent:
            differing.add(uuid)
            continue

        if uuid in dyn.dynamic_uuids:
            if cur.fields.get("isDeleted") != tgt.fields.get("isDeleted"):
                differing.add(uuid)
            continue

        if cur.fields != tgt.fields:
            differing.add(uuid)
        elif temperature_differs(cur.temperature, tgt.temperature):
            temp_only.add(uuid)

    for uuid, cur in current.items():
        if uuid not in target and cur.name not in dyn.spontaneous_new_names:
            new.add(uuid)

    approximate = not differing and not missing and not new
    exact = approximate and not temp_only

    return Match(
        exact=exact,
        approximate=approximate,
        differing=differing | temp_only,
        missing=missing,
        new=new,
    )


def state_signature(flat: dict[str, ObjRecord]) -> tuple:
    """Hashable state summary used to skip states already visited."""
    items = []
    for uuid in sorted(flat):
        rec = flat[uuid]
        temp = (
            None if rec.temperature is None
            else round(rec.temperature / TEMP_TOLERANCE_C)
        )
        items.append(
            (uuid, rec.parent, tuple(sorted(rec.fields.items())), temp)
        )
    return tuple(items)


# =========================================================
# ENVIRONMENT WRAPPER
# =========================================================

class SWSession:
    """One task variation loaded once. States are rebuilt with reset+replay."""

    def __init__(self, task_name: str, variation: int):
        self.task_name = task_name
        self.variation = variation
        self.env = ScienceWorldEnv(
            taskName=task_name,
            serverPath=None,
            envStepLimit=ENV_STEP_LIMIT,
        )
        self.env.load(task_name, variation, "", generateGoldPath=True)
        self.env.reset()
        self.gold = list(self.env.get_gold_action_sequence() or [])
        self.replays = 0

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def replay(self, history: list[str]) -> dict[str, Any]:
        """
        Reset and replay a history. The done flag is ignored on purpose:
        in scienceworld 1.2.3 the simulation keeps running after the task
        ends, and completion is a scoring event, not a physical one.
        """
        self.env.reset()
        self.replays += 1
        valid = True
        last_obs = ""

        for action in history:
            last_obs, _, _, _ = self.env.step(action)

        lowered = last_obs.lower()
        if history and (
            "no known action matches" in lowered
            or "ambiguous request" in lowered
        ):
            valid = False

        return {"valid": valid, "last_obs": last_obs}

    def flat_state(self) -> dict[str, ObjRecord]:
        return flatten_tree(self.env.getObjectTree())

    def raw_tree(self) -> dict[str, Any]:
        return self.env.getObjectTree()

    def valid_templates(self) -> list[dict[str, Any]]:
        return list(
            self.env.get_valid_action_object_combinations_with_templates()
        )

    def state_text(self) -> str:
        task = self.env.get_task_description()
        look = self.env.look()
        inv = self.env.inventory()
        return f"Task: {task}\n\nObservation:\n{look}\n\n{inv}".strip()


# =========================================================
# RECOVERY ACTION SELECTION
# =========================================================

def agent_uuid(flat: dict[str, ObjRecord]) -> str | None:
    for uuid, rec in flat.items():
        if rec.name == "agent":
            return uuid
    return None


def recovery_actions(
    templates: list[dict[str, Any]],
    current: dict[str, ObjRecord],
    target: dict[str, ObjRecord],
    match: Match,
) -> tuple[list[str], bool]:
    """
    Valid actions at this node that could move the state toward the target.

    An action is kept when its verb is a recovery verb and:
      - move/pour: obj0 still differs and obj1 is a place where some
        still-different object lived in the target state;
      - pick up / put down / disconnect: the object still differs;
      - connect: every object involved still differs;
      - open: the object differs, or is the home or current container of
        something that differs (it may need opening to reach inside);
      - close / activate / deactivate: the object differs, or is the home
        container of something that differs;
      - go: only when the agent is not where it was in the target state.

    Returns (actions, truncated). truncated=True means MAX_BRANCHING cut the
    list, so the level is not exhaustive.
    """
    to_fix = match.differing | match.new | match.missing

    home_containers = {
        target[u].parent for u in to_fix
        if u in target and target[u].parent
    }
    current_containers = {
        current[u].parent for u in to_fix
        if u in current and current[u].parent
    }
    openable = to_fix | home_containers | current_containers
    switchable = to_fix | home_containers

    a_uuid = agent_uuid(current)
    agent_moved = (
        a_uuid is not None
        and a_uuid in target
        and current[a_uuid].parent != target[a_uuid].parent
    )

    kept: list[str] = []
    seen: set[str] = set()

    for entry in templates:
        action = normalise(entry["action"])
        if action in seen or not action.startswith(RECOVERY_PREFIXES):
            continue

        objs = [str(o) for o in entry.get("obj_ids", [])]

        if not objs and not action.startswith("go "):
            continue

        if action.startswith("go "):
            ok = agent_moved
        elif action.startswith(PLACEMENT_PREFIXES):
            ok = (
                len(objs) >= 2
                and objs[0] in to_fix
                and objs[1] in home_containers
            )
        elif action.startswith(("pick up ", "put down ", "disconnect ")):
            ok = objs[0] in to_fix
        elif action.startswith("connect "):
            ok = all(o in to_fix for o in objs)
        elif action.startswith("open "):
            ok = objs[0] in openable
        else:  # close, activate, deactivate
            ok = objs[0] in switchable

        if ok:
            seen.add(action)
            kept.append(action)

    # Likely inverses first so that a budget cut keeps the most useful ones.
    priority = ("move ", "close ", "open ", "deactivate ", "activate ",
                "disconnect ", "connect ", "pick up ", "put down ", "go ",
                "pour ")
    kept.sort(key=lambda a: (
        next(i for i, p in enumerate(priority) if a.startswith(p)), a
    ))

    truncated = len(kept) > MAX_BRANCHING
    return kept[:MAX_BRANCHING], truncated


# =========================================================
# RECOVERY SEARCH
# =========================================================

@dataclass
class SearchResult:
    status: str                     # searched | budget_exhausted
    exact_cost: int | None = None
    approx_cost: int | None = None
    exact_path: list[str] | None = None
    approx_path: list[str] | None = None
    exhausted_depth: int = 0        # deepest level fully evaluated
    nodes: int = 0
    branching_truncated: bool = False
    root_branching: int = 0


def search_recovery(
    session: SWSession,
    prefix: list[str],              # history before the candidate + candidate
    target: dict[str, ObjRecord],
    post: dict[str, ObjRecord],
    post_templates: list[dict[str, Any]],
    dyn: Dynamics,
) -> SearchResult:
    """
    Breadth-first search level by level. States already seen at the same or
    a shallower depth are skipped. Stops at the first exact match.
    """
    result = SearchResult(status="searched")

    root_match = compare_states(post, target, dyn)
    if root_match.exact:
        result.exact_cost = result.approx_cost = 0
        result.exact_path = result.approx_path = []
        result.exhausted_depth = MAX_HORIZON
        return result
    if root_match.approximate:
        result.approx_cost = 0
        result.approx_path = []

    root_actions, truncated = recovery_actions(
        post_templates, post, target, root_match
    )
    result.branching_truncated |= truncated
    result.root_branching = len(root_actions)

    frontier: list[tuple[list[str], list[str]]] = [([], root_actions)]
    seen = {state_signature(post)}

    for depth in range(1, MAX_HORIZON + 1):
        next_frontier: list[tuple[list[str], list[str]]] = []
        level_complete = True

        for parent_path, actions in frontier:
            for action in actions:
                if result.nodes >= MAX_NODES_PER_EXAMPLE:
                    level_complete = False
                    break

                path = parent_path + [action]
                outcome = session.replay(prefix + path)
                result.nodes += 1

                if not outcome["valid"]:
                    continue

                state = session.flat_state()
                match = compare_states(state, target, dyn)

                if match.exact:
                    result.exact_cost = depth
                    result.exact_path = path
                    if result.approx_cost is None:
                        result.approx_cost = depth
                        result.approx_path = path
                    result.exhausted_depth = depth - 1
                    return result

                if match.approximate and result.approx_cost is None:
                    result.approx_cost = depth
                    result.approx_path = path

                signature = state_signature(state)
                if signature in seen:
                    continue
                seen.add(signature)

                if depth < MAX_HORIZON:
                    children, truncated = recovery_actions(
                        session.valid_templates(), state, target, match
                    )
                    result.branching_truncated |= truncated
                    next_frontier.append((path, children))

            if not level_complete:
                break

        if not level_complete:
            result.status = "budget_exhausted"
            return result

        result.exhausted_depth = depth
        frontier = next_frontier

        if not frontier:
            # Nothing left to expand: deeper levels are empty, so the
            # search is exhaustive for every remaining depth.
            result.exhausted_depth = MAX_HORIZON
            break

    return result


# =========================================================
# LABELLING
# =========================================================

def label_at_horizon(k: int, res: SearchResult) -> str:
    exhaustive = res.exhausted_depth >= k and not res.branching_truncated

    if res.exact_cost is not None and res.exact_cost <= k:
        return "reversible"
    if res.approx_cost is not None and res.approx_cost <= k:
        # Partial only if no exact recovery within k was missed.
        return "partially_reversible" if exhaustive else "inconclusive"
    if exhaustive:
        return "irreversible"
    return "inconclusive"


def binary_from_trinary(label: str) -> str | None:
    """Binary task: reversible vs irreversible. Partial cases are excluded."""
    return label if label in ("reversible", "irreversible") else None


def rule_label(reason: str) -> dict[str, Any]:
    labels = {str(k): "irreversible" for k in range(1, MAX_HORIZON + 1)}
    return {
        "label": "irreversible",
        "binary_label": "irreversible",
        "labels_by_horizon": labels,
        "label_method": "rule",
        "label_reason": reason,
    }


# =========================================================
# EXAMPLE CONSTRUCTION
# =========================================================

def object_names(uuids: set[str], *flats: dict[str, ObjRecord]) -> list[str]:
    names = set()
    for uuid in uuids:
        for flat in flats:
            if uuid in flat:
                names.add(flat[uuid].name)
                break
    return sorted(names)


def build_example(
    session: SWSession,
    history_before: list[str],
    candidate: str,
    example_id: str,
    source: str,
    pre_cache: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Label one (state, action) pair. Returns None if the action has no effect."""
    candidate = normalise(candidate)
    t0 = time.time()

    if pre_cache is None:
        pre_cache = prepare_pre_state(session, history_before)

    base = {
        "id": example_id,
        "environment": "scienceworld",
        "scenario_id": f"{session.task_name}:v{session.variation}",
        "task_name": session.task_name,
        "variation": session.variation,
        "state_text": pre_cache["state_text"],
        "action_text": candidate,
        "horizon": MAX_HORIZON,
        "source": source,
        "history_before": history_before,
    }

    if candidate.startswith(COMMIT_PREFIXES):
        return {
            **base,
            **rule_label("focus_commits_answer"),
            "search": None,
            "elapsed_s": round(time.time() - t0, 2),
        }

    session.replay(history_before + [candidate])
    post = session.flat_state()
    post_templates = session.valid_templates()
    post_tree = session.raw_tree() if SAVE_TREES else None

    target = pre_cache["flat"]
    dyn = pre_cache["dynamics"]

    root = compare_states(post, target, dyn)
    if root.exact and SKIP_NO_EFFECT_ACTIONS:
        return None

    res = search_recovery(
        session=session,
        prefix=history_before + [candidate],
        target=target,
        post=post,
        post_templates=post_templates,
        dyn=dyn,
    )

    labels = {
        str(k): label_at_horizon(k, res)
        for k in range(1, MAX_HORIZON + 1)
    }
    main_label = labels[str(MAX_HORIZON)]

    record = {
        **base,
        "label": main_label,
        "binary_label": binary_from_trinary(main_label),
        "labels_by_horizon": labels,
        "label_method": "search",
        "label_reason": None,
        "search": {
            "status": res.status,
            "exact_cost": res.exact_cost,
            "approx_cost": res.approx_cost,
            "exact_path": res.exact_path,
            "approx_path": res.approx_path,
            "exhausted_depth": res.exhausted_depth,
            "nodes": res.nodes,
            "branching_truncated": res.branching_truncated,
            "root_branching": res.root_branching,
            "max_nodes": MAX_NODES_PER_EXAMPLE,
            "temp_tolerance_c": TEMP_TOLERANCE_C,
        },
        "effect": {
            "changed_objects": object_names(root.differing, post, target),
            "created_objects": object_names(root.new, post),
            "removed_objects": object_names(root.missing, target),
            "dynamic_objects_ignored": len(dyn.dynamic_uuids),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }

    if SAVE_TREES:
        record["pre_tree"] = pre_cache["tree"]
        record["post_tree"] = post_tree

    return record


def prepare_pre_state(
    session: SWSession,
    history_before: list[str],
) -> dict[str, Any]:
    """Pre-action state, its text, and the background dynamics baseline."""
    session.replay(history_before)
    flat = session.flat_state()
    state_text = session.state_text()
    templates = session.valid_templates()
    tree = session.raw_tree() if SAVE_TREES else None

    # Let time pass for as many ticks as candidate + longest recovery.
    session.replay(history_before + ["wait1"] * (MAX_HORIZON + 1))
    dynamics = compute_dynamics(flat, session.flat_state())

    return {
        "flat": flat,
        "state_text": state_text,
        "templates": templates,
        "dynamics": dynamics,
        "tree": tree,
    }


def sample_random_actions(
    session: SWSession,
    history_before: list[str],
    pre_cache: dict[str, Any],
    gold_action: str,
    rng: random.Random,
    count: int,
) -> list[str]:
    """
    Random valid actions at the pre-state that actually change the world.

    A verb is drawn uniformly first, then an action with that verb. Drawing
    from the flat list would mostly return "connect X to Y", which makes up
    most of the valid-action combinations.
    """
    by_verb: dict[str, list[str]] = {}
    for t in pre_cache["templates"]:
        action = normalise(t["action"])
        if action.startswith(NEVER_SAMPLE_PREFIXES):
            continue
        if action == normalise(gold_action):
            continue
        verb = action_verb(action)
        by_verb.setdefault(verb, [])
        if action not in by_verb[verb]:
            by_verb[verb].append(action)

    verbs = sorted(by_verb)
    attempts: list[str] = []
    for _ in range(RANDOM_ACTION_TRIES):
        if not verbs:
            break
        verb = rng.choice(verbs)
        options = sorted(set(by_verb[verb]) - set(attempts))
        if options:
            attempts.append(rng.choice(options))

    chosen: list[str] = []
    for action in attempts:
        if len(chosen) >= count:
            break
        outcome = session.replay(history_before + [action])
        if not outcome["valid"]:
            continue
        match = compare_states(
            session.flat_state(), pre_cache["flat"], pre_cache["dynamics"]
        )
        if not match.exact:
            chosen.append(action)
    return chosen


# =========================================================
# WORKER: ONE TASK VARIATION
# =========================================================

def process_scenario(
    task_name: str,
    variation: int,
    skip_ids: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    session = SWSession(task_name, variation)
    rng = random.Random(f"{SEED}:{task_name}:{variation}")

    try:
        gold = session.gold
        for index in SAMPLE_POSITIONS:
            if index >= len(gold):
                continue

            candidate = normalise(gold[index])
            if candidate.isdigit():
                continue    # answer to an ambiguity prompt, not an action
            if SKIP_NO_EFFECT_ACTIONS and candidate.startswith(
                NO_EFFECT_PREFIXES
            ):
                continue

            history = gold[:index]
            gold_id = f"scienceworld:{task_name}:v{variation}:a{index}:gold"
            pre_cache = None

            if gold_id not in skip_ids:
                pre_cache = prepare_pre_state(session, history)
                rec = build_example(
                    session, history, candidate, gold_id,
                    "gold_action_sequence", pre_cache,
                )
                if rec:
                    records.append(rec)
                    log_record(rec)

            if RANDOM_ACTIONS_PER_POSITION <= 0:
                continue

            rand_ids = [
                f"scienceworld:{task_name}:v{variation}:a{index}:rand{i}"
                for i in range(RANDOM_ACTIONS_PER_POSITION)
            ]
            if all(r in skip_ids for r in rand_ids):
                continue

            if pre_cache is None:
                pre_cache = prepare_pre_state(session, history)

            randoms = sample_random_actions(
                session, history, pre_cache, candidate, rng,
                RANDOM_ACTIONS_PER_POSITION,
            )
            for rand_id, action in zip(rand_ids, randoms):
                if rand_id in skip_ids:
                    continue
                rec = build_example(
                    session, history, action, rand_id,
                    "random_valid_action", pre_cache,
                )
                if rec:
                    records.append(rec)
                    log_record(rec)
    finally:
        session.close()

    return records


def log_record(rec: dict[str, Any]) -> None:
    search = rec.get("search") or {}
    print(
        f"  {rec['id']:<60} {rec['action_text'][:38]:<38} "
        f"-> {rec['label']:<21} "
        f"exact={search.get('exact_cost')} approx={search.get('approx_cost')} "
        f"depth={search.get('exhausted_depth')} nodes={search.get('nodes')} "
        f"b={search.get('root_branching')} "
        f"({rec['elapsed_s']}s)",
        flush=True,
    )


def scenario_job(args: tuple[str, int, set[str]]) -> tuple[str, int, list, str | None]:
    task_name, variation, skip_ids = args
    try:
        return task_name, variation, process_scenario(
            task_name, variation, skip_ids
        ), None
    except Exception as exc:
        return task_name, variation, [], f"{type(exc).__name__}: {exc}"


# =========================================================
# MAIN
# =========================================================

def resolve_jobs(tasks: list[str], variations: int) -> list[tuple[str, int]]:
    probe = ScienceWorldEnv("boil", envStepLimit=ENV_STEP_LIMIT)
    try:
        available = set(probe.get_task_names())
        jobs = []
        for task in tasks:
            if task not in available:
                print(f"WARNING: unknown task '{task}' skipped")
                continue
            probe.load(task, 0, "")
            train = sorted(probe.get_variations_train())
            for variation in train[:variations]:
                jobs.append((task, variation))
        return jobs
    finally:
        try:
            probe.close()
        except Exception:
            pass


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key_fn) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in records:
            k = str(key_fn(r))
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    summary = {
        "examples": len(records),
        "label_k3": count(lambda r: r["label"]),
        "binary_label": count(lambda r: r["binary_label"]),
        "label_method": count(lambda r: r["label_method"]),
        "source": count(lambda r: r["source"]),
        "search_status": count(
            lambda r: (r.get("search") or {}).get("status", "rule")
        ),
        "scenarios": len({r["scenario_id"] for r in records}),
    }
    for k in range(1, MAX_HORIZON + 1):
        summary[f"label_k{k}"] = count(
            lambda r, k=k: r["labels_by_horizon"][str(k)]
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tasks", nargs="*", default=TASK_NAMES)
    parser.add_argument("--variations", type=int, default=VARIATIONS_PER_TASK)
    parser.add_argument("--workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--out", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--fresh", action="store_true",
                        help="overwrite the output instead of resuming")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and args.out.exists():
        args.out.unlink()

    existing = load_existing(args.out)
    skip_ids = {r["id"] for r in existing}
    if existing:
        print(f"Resuming: {len(existing)} examples already in {args.out}")

    jobs = resolve_jobs(args.tasks, args.variations)
    done_jobs = {(r.get("task_name"), r.get("variation")) for r in existing}
    skipped_jobs = [j for j in jobs if j in done_jobs]
    jobs = [j for j in jobs if j not in done_jobs]
    if skipped_jobs:
        print(f"Skipping {len(skipped_jobs)} scenarios that already have rows")
        
    print(f"{len(jobs)} scenarios, {args.workers} workers, "
          f"horizon={MAX_HORIZON}, budget={MAX_NODES_PER_EXAMPLE} nodes")

    t0 = time.time()
    all_records = list(existing)
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            args.out.open("a", encoding="utf-8") as out:
        futures = [
            pool.submit(scenario_job, (task, var, skip_ids))
            for task, var in jobs
        ]
        for done, future in enumerate(as_completed(futures), 1):
            task, var, records, error = future.result()
            if error:
                failures.append(f"{task}:v{var}: {error}")
                print(f"[{done}/{len(jobs)}] ERROR {task}:v{var}: {error}")
                continue
            for rec in records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            all_records.extend(records)
            print(f"[{done}/{len(jobs)}] {task}:v{var}: "
                  f"{len(records)} examples "
                  f"({(time.time() - t0) / 60:.1f} min elapsed)")

    summary = summarise(all_records)
    summary["failures"] = failures
    summary["output_file"] = str(args.out)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(json.dumps(summary, indent=2))
    print(f"\nDataset: {args.out}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
