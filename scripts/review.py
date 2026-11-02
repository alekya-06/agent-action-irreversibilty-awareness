"""
Review ToolEmu drafts one case at a time.

Each reviewer's decisions go to their own file, data/toolemu/pairs/reviews/<name>.jsonl,
so several people can review in parallel without editing the same file.
--apply copies the decisions into drafts_v2.jsonl (sets "reviewed").

Usage (from the project root):
  python scripts/review.py --reviewer ravi                         # everything you haven't decided
  python scripts/review.py --reviewer ravi --cases official_5 official_27
  python scripts/review.py --reviewer ravi --start 0 --end 72       # a slice, to split the work
  python scripts/review.py --apply                                  # merge all reviewers' decisions

A case counts as reviewed only if someone approved its CURRENT version and
nobody flagged that version. Editing a draft (e.g. apply_fix.py) makes old
decisions stale, so the case returns to unreviewed until it is reviewed again.
"""
import argparse
import hashlib
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

DRAFTS = Path("data/toolemu/pairs/drafts_v2.jsonl")
REVIEWS = Path("data/toolemu/pairs/reviews")
CASES = Path("external/ToolEmu/assets/all_cases.json")

CHECKLIST = """Checklist:
  1. Is the risky call the hasty action described in the risks?
  2. Is each safe call really safer, and does it still make progress?
  3. Is the check call read-only, and would it resolve the uncertainty?
  4. Are the observations read-only, with no made-up ids hidden in free text?
  5. Are made-up values in the risky call the intended risk, not a mistake?"""


def draft_hash(r: dict) -> str:
    content = {"observations": r.get("observations", ""), "calls": r.get("calls", [])}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def show(r: dict, case: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{r['case']}   author: {r.get('author')}   hand_edited: {r.get('hand_edited', False)}")
    print("=" * 78)
    print("USER REQUEST:\n" + textwrap.fill(case["User Instruction"], 78))
    print("\nRISKS:")
    for risk in case.get("Potential Risky Actions", []):
        print(textwrap.fill(risk, 78, initial_indent="- ", subsequent_indent="  "))
    print("\nOBSERVATIONS:")
    for line in r.get("observations", "").splitlines():
        print("  " + line)
    print("\nCALLS:")
    for c in r.get("calls", []):
        print(f"  {c.get('role', '?'):>5} -> {c.get('tool')} "
              f"{json.dumps(c.get('args'), ensure_ascii=False)}")
    if r.get("risky_ungrounded"):
        print(f"\nMADE-UP VALUES IN RISKY CALL: {r['risky_ungrounded']}")
    if r.get("draft_problems"):
        print(f"\nVALIDATOR PROBLEMS: {r['draft_problems']}")
    print("\n" + CHECKLIST)


def ask(prompt: str, allowed: set[str]) -> str:
    while True:
        choice = input(prompt).strip().lower()
        if choice in allowed:
            return choice


def review(args) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.reviewer):
        sys.exit("--reviewer must be a simple name (letters, digits, _ or -)")
    cases = {c["name"]: c for c in json.load(open(CASES, encoding="utf-8"))}
    drafts = load_jsonl(DRAFTS)
    if args.cases:
        known = {r["case"] for r in drafts}
        missing = sorted(set(args.cases) - known)
        if missing:
            sys.exit(f"Not in {DRAFTS}: {', '.join(missing)}")
        drafts = [r for r in drafts if r["case"] in set(args.cases)]
    drafts = drafts[args.start:args.end]

    REVIEWS.mkdir(parents=True, exist_ok=True)
    out = REVIEWS / f"{args.reviewer}.jsonl"
    decided = {(d["case"], d["draft_hash"]) for d in load_jsonl(out)}
    todo = [r for r in drafts if (r["case"], draft_hash(r)) not in decided]
    print(f"{len(todo)} to review ({len(drafts) - len(todo)} already decided by {args.reviewer}).")

    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            show(r, cases[r["case"]])
            choice = ask(f"\n[{i}/{len(todo)}] (a)pprove  (f)lag  (s)kip  (q)uit: ",
                         {"a", "f", "s", "q"})
            if choice == "q":
                break
            if choice == "s":
                continue
            if choice == "f":
                note = ""
                while not note:
                    note = input("What is wrong / what should be fixed? ").strip()
            else:
                note = input("Note (optional, Enter to skip): ").strip()
            fh.write(json.dumps({
                "case": r["case"],
                "decision": "approve" if choice == "a" else "flag",
                "note": note,
                "reviewer": args.reviewer,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "draft_hash": draft_hash(r),
            }, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"Decisions saved to {out}. Run  python scripts/review.py --apply  to merge them.")


def apply() -> None:
    decisions = [d for p in sorted(REVIEWS.glob("*.jsonl")) for d in load_jsonl(p)]
    drafts = load_jsonl(DRAFTS)
    text = DRAFTS.read_text(encoding="utf-8")
    DRAFTS.with_name(DRAFTS.stem + ".before_review.jsonl").write_text(text, encoding="utf-8")

    approved, flagged, stale, conflicts = [], [], 0, []
    for r in drafts:
        current = draft_hash(r)
        mine = [d for d in decisions if d["case"] == r["case"]]
        valid = [d for d in mine if d["draft_hash"] == current]
        stale += len(mine) - len(valid)
        approvals = [d for d in valid if d["decision"] == "approve"]
        flags = [d for d in valid if d["decision"] == "flag"]
        r["reviewed"] = bool(approvals) and not flags
        r["reviewed_by"] = sorted({d["reviewer"] for d in approvals})
        r["review_flags"] = [f"{d['reviewer']}: {d['note']}" for d in flags]
        r["review_notes"] = [f"{d['reviewer']}: {d['note']}" for d in approvals if d["note"]]
        if approvals and flags:
            conflicts.append(r["case"])
        if flags:
            flagged.append(r)
        elif r["reviewed"]:
            approved.append(r)

    DRAFTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in drafts),
                      encoding="utf-8")
    print(f"Approved: {len(approved)}   Flagged: {len(flagged)}   "
          f"Unreviewed: {len(drafts) - len(approved) - len(flagged)}   "
          f"Stale decisions ignored: {stale}")
    if conflicts:
        print(f"Reviewers disagree (flag wins until resolved): {', '.join(conflicts)}")
    for r in flagged:
        print(f"  FLAGGED {r['case']}: {r['review_flags']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review ToolEmu drafts.")
    parser.add_argument("--reviewer", help="your name; decisions go to reviews/<name>.jsonl")
    parser.add_argument("--cases", nargs="+", help="review only these cases")
    parser.add_argument("--start", type=int, default=0, help="first draft index (for splitting)")
    parser.add_argument("--end", type=int, default=None, help="last draft index, exclusive")
    parser.add_argument("--apply", action="store_true", help="merge decisions into the drafts")
    args = parser.parse_args()
    if args.apply:
        apply()
    elif args.reviewer:
        review(args)
    else:
        parser.error("give --reviewer NAME to review, or --apply to merge decisions")


if __name__ == "__main__":
    main()