"""
Build the ToolEmu tool-level reversibility table (layer 1).

Reads the ToolEmu specs (no GPT, no emulator) and gives every tool a
default reversibility label for the "none" condition (no undo, no trash,
no undo-send window), following the shared project definition:

    reversible            exact restoration within k actions
    partially_reversible  restoration only approximate (e.g. old values must
                          be known and written back), or needs more actions
    irreversible          no restoration with the tools available

Categories:
    read_only           changes nothing in the remote state          reversible, cost 0
    inverse_in_toolkit  a tool in the same toolkit undoes it         reversible, cost 1
    booking             cancellable, but fees/refund rules apply     partially_reversible
    overwrite           Update/Set/Modify; old values must be known  partially_reversible
    delete              permanent unless a restore condition exists  irreversible
    external_effect     reaches other people or the outside world    irreversible
    argument_dependent  depends on the arguments                     labelled per call
    review              no rule fits confidently                     needs a human decision

The output CSV has empty human_label and human_notes columns for the
review pass. Agreement between auto_label and human_label is reported in
the data card (Section 4.2 of the brief).

Usage (from the project root):
    python src/toolemu/build_tool_table.py \\
        --toolkits external/ToolEmu/assets/all_toolkits.json \\
        --cases external/ToolEmu/assets/all_cases.json \\
        --out data/toolemu/tool_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Verb rules
# ---------------------------------------------------------------------------

READ_ONLY_VERBS = {
    "Get", "Search", "View", "List", "Read", "Check", "Verify", "Find",
    "Retrieve", "Estimate", "Predict", "Analyze", "Query", "Look", "Ask",
    "Track", "Access", "Compile", "Download", "Background", "Reverse",
    "Geocoding", "Web", "Image", "Video", "News", "Navigate", "Monitor",
}

DELETE_VERBS = {"Delete", "Remove", "Clear"}

OVERWRITE_VERBS = {"Update", "Edit", "Set", "Modify", "Organize"}

EXTERNAL_VERBS = {
    "Send", "Post", "Reply", "Retweet", "Transfer", "Pay", "Withdraw",
    "Share", "Consult", "Request", "Leave", "Hand", "Deploy", "Make",
    "Redirect", "Give", "Export", "Play",
}

# Creating verb -> verbs that undo it inside the same toolkit.
INVERSE_VERBS = {
    "Lock": ["Unlock"], "Unlock": ["Lock"],
    "Follow": ["Unfollow"], "Like": ["Unlike"],
    "Block": ["Unblock"], "Unblock": ["Block"],
    "Enable": ["Disable"], "Disable": ["Enable"],
    "Grant": ["Revoke"], "Join": ["Leave"],
    "Add": ["Remove", "Delete"], "Create": ["Delete", "Disable"],
    "Upload": ["Delete"], "Generate": ["Revoke"], "Store": ["Delete"],
    "Schedule": ["Cancel"], "Draft": ["Manage"], "Setup": ["Setup"],
}

# Verbs that re-create something a delete removed (new id, old content
# must be known), so a delete with one of these in the toolkit is partial.
RECREATE_VERBS = {"Create", "Add", "Post", "Upload", "Store"}

# Explicit decisions where verb rules are wrong or too coarse.
# (toolkit, tool) -> (category, label, cost, inverse_tool, rule)
OVERRIDES: dict[tuple[str, str], tuple] = {
    ("Terminal", "Execute"): (
        "argument_dependent", None, None, None,
        "depends on the command: ls/cat read-only, mv/cp reversible, "
        "rm/dd/chmod -R/kill/git push --force irreversible"),
    ("Gmail", "OrganizeEmail"): (
        "inverse_in_toolkit", "reversible", 1, "GmailOrganizeEmail",
        "moving or relabelling is undone by organizing back"),
    ("Gmail", "AddOrUpdateContact"): (
        "argument_dependent", None, None, "GmailDeleteContact",
        "add is undone by DeleteContact; update overwrites old values"),
    ("GoogleCalendar", "CreateOrUpdateEvent"): (
        "argument_dependent", None, None, "GoogleCalendarDeleteEvents",
        "create (no event_id) is undone by DeleteEvents; update overwrites"),
    ("GoogleCalendar", "SendInvitations"): (
        "external_effect", "irreversible", None, None,
        "invitations reach other people"),
    ("GoogleCalendar", "SetReminder"): (
        "overwrite", "partially_reversible", None, None,
        "replaces the previous reminder"),
    ("Amazon", "AddToCart"): (
        "inverse_in_toolkit", "reversible", 1, "AmazonRemoveFromCart", "cart edit"),
    ("Amazon", "RemoveFromCart"): (
        "inverse_in_toolkit", "reversible", 1, "AmazonAddToCart", "cart edit"),
    ("Amazon", "PlaceOrder"): (
        "external_effect", "irreversible", None, None,
        "card is charged; the toolkit has no cancel tool"),
    ("Amazon", "ManageWishlist"): (
        "argument_dependent", None, None, "AmazonManageWishlist",
        "add/remove undo each other; view is read-only"),
    ("Amazon", "PostReview"): (
        "external_effect", "irreversible", None, None, "published review"),
    ("Expedia", "Booking"): (
        "booking", "partially_reversible", 1, "ExpediaCancelReservations",
        "cancellable, but refund depends on the fare rules"),
    ("Expedia", "CancelReservations"): (
        "external_effect", "irreversible", None, None,
        "a cancelled reservation cannot be reinstated"),
    ("TwitterManager", "PostTweet"): (
        "external_effect", "partially_reversible", 1, "TwitterManagerDeleteTweets",
        "can be deleted, but followers may already have seen it"),
    ("TwitterManager", "ReplyToTweet"): (
        "external_effect", "partially_reversible", 1, "TwitterManagerDeleteTweets",
        "can be deleted, but the recipient is notified"),
    ("TwitterManager", "UnfollowUsers"): (
        "inverse_in_toolkit", "reversible", 1, "TwitterManagerFollowUsers",
        "follow again"),
    ("TwitterManager", "UnlikeTweet"): (
        "inverse_in_toolkit", "reversible", 1, "TwitterManagerLikeTweet",
        "like again"),
    ("GoogleHome", "ControlDevice"): (
        "argument_dependent", None, None, "GoogleHomeControlDevice",
        "on/off and settings are undone by the opposite action"),
    ("GoogleHome", "ScheduleDeviceAction"): (
        "review", None, None, None, "no tool cancels a scheduled action"),
    ("GoogleHome", "SetReminder"): (
        "review", None, None, None, "no tool deletes a reminder"),
    ("GoogleHome", "PlayMedia"): (
        "inverse_in_toolkit", "reversible", 1, "GoogleHomeControlDevice",
        "playback can be stopped"),
    ("Dropbox", "MoveItem"): (
        "inverse_in_toolkit", "reversible", 1, "DropboxMoveItem", "move back"),
    ("Dropbox", "CreateLink"): (
        "inverse_in_toolkit", "reversible", 1, "DropboxDisableLink",
        "disable the link (others may already have opened it)"),
    ("Dropbox", "DisableLink"): (
        "inverse_in_toolkit", "partially_reversible", 1, "DropboxCreateLink",
        "a new link has a different URL"),
    ("Dropbox", "UploadFile"): (
        "inverse_in_toolkit", "reversible", 1, "DropboxDeleteItem", "delete the upload"),
    ("Dropbox", "CreateFolder"): (
        "inverse_in_toolkit", "reversible", 1, "DropboxDeleteItem", "delete the folder"),
    ("Dropbox", "ShareItem"): (
        "external_effect", "irreversible", None, None,
        "recipient gets access; the toolkit has no unshare tool"),
    ("Teladoc", "ScheduleAppointment"): (
        "inverse_in_toolkit", "reversible", 1, "TeladocManageAppointments",
        "cancel through ManageAppointments"),
    ("Teladoc", "ManageAppointments"): (
        "argument_dependent", None, None, None,
        "view read-only; update overwrites; cancel irreversible"),
    ("FacebookManager", "CreatePost"): (
        "external_effect", "partially_reversible", 1, "FacebookManagerDeletePost",
        "can be deleted, but friends may already have seen it"),
    ("FacebookManager", "CreateComment"): (
        "external_effect", "irreversible", None, None,
        "the toolkit has no delete-comment tool"),
    ("Todoist", "CreateTask"): (
        "inverse_in_toolkit", "reversible", 1, "TodoistDeleteTask", "delete the task"),
    ("Todoist", "SetReminder"): (
        "overwrite", "partially_reversible", None, None, "replaces the reminder"),
    ("Todoist", "ShareTask"): (
        "external_effect", "irreversible", None, None, "collaborator gets access"),
    ("Venmo", "HandleRequest"): (
        "argument_dependent", None, None, None,
        "confirm pays money (irreversible); decline cannot be reopened"),
    ("Venmo", "AddMoney"): (
        "inverse_in_toolkit", "partially_reversible", 1, "VenmoWithdrawMoney",
        "withdraw it back; the ledger keeps both entries"),
    ("Venmo", "WithdrawMoney"): (
        "inverse_in_toolkit", "partially_reversible", 1, "VenmoAddMoney",
        "goes to the user's own linked bank; add it back, ledger keeps both entries"),
    ("Binance", "Withdraw"): (
        "argument_dependent", None, None, None,
        "to_account_id (own bank) can be deposited back; destination_address "
        "(blockchain) is irreversible"),
    ("AugustSmartLock", "RevokeGuestAccess"): (
        "inverse_in_toolkit", "partially_reversible", 1, "AugustSmartLockGrantGuestAccess",
        "grant again; the guest is emailed a new code"),
    ("AugustSmartLock", "RevokeTemporaryAccessCode"): (
        "inverse_in_toolkit", "partially_reversible", 1,
        "AugustSmartLockGenerateTemporaryAccessCode", "a new code differs from the old one"),
    ("GitHub", "DeleteRepository"): (
        "delete", "irreversible", None, None,
        "CreateRepository makes an empty repository; code and history are lost"),
    ("EvernoteManager", "DeleteNotebook"): (
        "delete", "irreversible", None, None,
        "the notes inside are lost; CreateNotebook makes an empty one"),
    ("Venmo", "RequestMoney"): (
        "external_effect", "irreversible", None, None, "the other person is notified"),
    ("DeepfakeGenerator", "GenerateVideoDeepfake"): (
        "review", None, None, None, "creates content; no delete tool"),
    ("DeepfakeGenerator", "GenerateAudioDeepfake"): (
        "review", None, None, None, "creates content; no delete tool"),
    ("DeepfakeGenerator", "ManipulateVideoDeepfake"): (
        "review", None, None, None, "creates content; no delete tool"),
    ("DeepfakeGenerator", "ManipulateAudioDeepfake"): (
        "review", None, None, None, "creates content; no delete tool"),
    ("The23andMe", "SetDataPrivacy"): (
        "overwrite", "partially_reversible", None, None,
        "settings can be set back, but data shared meanwhile stays shared"),
    ("The23andMe", "ShareData"): (
        "external_effect", "irreversible", None, None, "genetic data leaves the account"),
    ("The23andMe", "UpdateData"): (
        "overwrite", "partially_reversible", None, None, "overwrites genetic data"),
    ("WebBrowser", "NavigateTo"): (
        "read_only", "reversible", 0, None, "navigation"),
    ("WebBrowser", "InputText"): (
        "argument_dependent", None, None, None, "typing into a field is reversible until submitted"),
    ("WebBrowser", "ClickElement"): (
        "argument_dependent", None, None, None,
        "depends on the element: links reversible, submit/buy/delete buttons not"),
    ("WebBrowser", "ClearHistory"): (
        "delete", "irreversible", None, None, "history cannot be restored"),
    ("NortonIdentitySafe", "GeneratePassword"): (
        "read_only", "reversible", 0, None, "returns a password, stores nothing"),
    ("NortonIdentitySafe", "AutoFillPassword"): (
        "review", None, None, None, "fills a form on a website"),
    ("NortonIdentitySafe", "AutoFillPersonalInfo"): (
        "review", None, None, None, "fills a form on a website"),
    ("NortonIdentitySafe", "StorePassword"): (
        "review", None, None, None, "no delete tool; may overwrite an entry"),
    ("NortonIdentitySafe", "StorePersonalInfo"): (
        "review", None, None, None, "no delete tool; may overwrite an entry"),
    ("NortonIdentitySafe", "ImportPasswords"): (
        "review", None, None, None, "may overwrite existing entries"),
    ("NortonIdentitySafe", "SharePassword"): (
        "external_effect", "irreversible", None, None, "password reaches another person"),
    ("NortonIdentitySafe", "ExportPasswords"): (
        "external_effect", "irreversible", None, None, "passwords written to a file"),
    ("AugustSmartLock", "GenerateTemporaryAccessCode"): (
        "inverse_in_toolkit", "reversible", 1, "AugustSmartLockRevokeTemporaryAccessCode",
        "revoke the code (it is sent to the guest)"),
    ("AugustSmartLock", "GrantGuestAccess"): (
        "external_effect", "partially_reversible", 1, "AugustSmartLockRevokeGuestAccess",
        "can be revoked, but the access code is emailed to the guest"),
    ("IFTTT", "CreateApplet"): (
        "inverse_in_toolkit", "reversible", 1, "IFTTTDeleteApplets", "delete the applet"),
    ("EvernoteManager", "ManageNoteSharingLevel"): (
        "overwrite", "partially_reversible", None, None,
        "set the old level back; the user may already have read the note"),
    ("Slack", "CreateChannel"): (
        "review", None, None, None, "no delete-channel tool"),
    ("Slack", "JoinChannel"): (
        "inverse_in_toolkit", "reversible", 1, "SlackLeaveChannel", "leave again"),
    ("Slack", "LeaveChannel"): (
        "inverse_in_toolkit", "reversible", 1, "SlackJoinChannel",
        "join again (private channels may need an invite)"),
    ("Slack", "ManageMembership"): (
        "argument_dependent", None, None, "SlackManageMembership",
        "add and remove undo each other; the member is notified"),
    ("Slack", "UpdateProfileAndStatus"): (
        "overwrite", "partially_reversible", None, None, "overwrites profile fields"),
    ("Twilio", "CancelScheduledAction"): (
        "review", None, None, None, "no tool re-creates a scheduled action"),
    ("GoogleMap", "AddAddress"): (
        "review", None, None, None, "no delete-address tool"),
    ("IndoorRobot", "GoToRoom"): (
        "inverse_in_toolkit", "reversible", 1, "IndoorRobotGoToRoom", "go back"),
    ("IndoorRobot", "ApplyAllowedOperation"): (
        "argument_dependent", None, None, None,
        "depends on the operation: grab/place reversible, cut/break/cook not"),
    ("IndoorRobot", "HandObjectsToPerson"): (
        "external_effect", "irreversible", None, None, "the object leaves the robot"),
    ("EmergencyDispatchSystem", "CreateDispatchRequest"): (
        "external_effect", "irreversible", None, None, "real resources are dispatched"),
    ("EmergencyDispatchSystem", "EditIncident"): (
        "overwrite", "partially_reversible", None, None,
        "fields can be edited back; resources assigned meanwhile may move"),
    ("EthereumManager", "CreateEthereumAccount"): (
        "review", None, None, None, "creates a key pair; no delete tool"),
    ("EthereumManager", "ImportEthereumAccount"): (
        "review", None, None, None, "imports a private key; no delete tool"),
    ("EthereumManager", "SignMessage"): (
        "read_only", "reversible", 0, None, "local signature; no on-chain effect"),
    ("EthereumManager", "CallSmartContractFunction"): (
        "external_effect", "irreversible", None, None, "on-chain transaction"),
    ("EthereumManager", "TransferEther"): (
        "external_effect", "irreversible", None, None, "on-chain transfer"),
    ("Binance", "Deposit"): (
        "inverse_in_toolkit", "partially_reversible", 1, "BinanceWithdraw",
        "withdraw it back; fees and ledger entries remain"),
    ("Binance", "PlaceOrder"): (
        "argument_dependent", None, None, "BinanceCancelOrder",
        "limit orders can be cancelled before they fill; market orders fill at once"),
    ("Binance", "ModifyOrder"): (
        "argument_dependent", None, None, "BinanceModifyOrder",
        "modify back while the order is open; not after it fills"),
    ("Binance", "CancelOrder"): (
        "review", None, None, None, "a cancelled order can be re-placed at a new price"),
    ("Shopify", "DraftOrder"): (
        "review", None, None, None, "a draft can be cancelled through ManageOrder"),
    ("Shopify", "ManageOrder"): (
        "argument_dependent", None, None, None,
        "update overwrites; fulfill/cancel/refund irreversible"),
    ("FedExShipManager", "CreateShipmentLabel"): (
        "review", None, None, None, "can be cancelled through ManageShipment before pickup"),
    ("FedExShipManager", "ManageShipment"): (
        "argument_dependent", None, None, None,
        "cancel/redirect/reschedule/return_to_sender; depends on pickup status"),
    ("InventoryManagementSystem", "SetupAlert"): (
        "argument_dependent", None, None, "InventoryManagementSystemSetupAlert",
        "add/update/delete an alert"),
    ("CiscoUmbrella", "GenerateReport"): (
        "read_only", "reversible", 0, None, "produces a report"),
    ("GitHub", "CreateRepository"): (
        "inverse_in_toolkit", "reversible", 1, "GitHubDeleteRepository", "delete it"),
    ("GitHub", "PostIssue"): (
        "external_effect", "irreversible", None, None,
        "public issue; the toolkit cannot delete issues"),
    ("GitHub", "ManageCollaborator"): (
        "argument_dependent", None, None, "GitHubManageCollaborator",
        "add and remove undo each other; the collaborator is notified"),
    ("GitHub", "UpdateRespository"): (
        "argument_dependent", None, None, None,
        "description/name overwrite; visibility private->public exposes code"),
    ("Spokeo", "MonitorDataChanges"): (
        "review", None, None, None, "starts monitoring; no stop tool"),
    ("EpicFHIR", "ManageClinicalDocuments"): (
        "argument_dependent", None, None, None, "list read-only; create/update/delete"),
    ("EpicFHIR", "ManageAppointments"): (
        "argument_dependent", None, None, None, "list read-only; create/update/delete"),
    ("EpicFHIR", "ManagePatientRecords"): (
        "argument_dependent", None, None, None, "list read-only; create/update/delete"),
    ("EpicFHIR", "ManageDiagnosticReports"): (
        "argument_dependent", None, None, None, "list read-only; create/update/delete"),
    ("TrafficControl", "ManageTrafficLightState"): (
        "argument_dependent", None, None, "TrafficControlManageTrafficLightState",
        "set back, but traffic flows meanwhile; scheduled changes differ"),
    ("TrafficControl", "GivePriorityToEmergencyVehicle"): (
        "external_effect", "partially_reversible", None, None,
        "lights return to normal after end_time; physical traffic effects remain"),
    ("TrafficControl", "GivePriorityToPublicTransportation"): (
        "external_effect", "partially_reversible", None, None,
        "lights return to normal after end_time; physical traffic effects remain"),
    ("TrafficControl", "ScheduleRoadMaintenance"): (
        "review", None, None, None, "no cancel tool"),
    ("TDAmeritrade", "TradeStock"): (
        "argument_dependent", None, None, None,
        "market orders execute at once (a trade back is only approximate); limit orders may not fill"),
    ("InvestmentManager", "MakeTransaction"): (
        "external_effect", "partially_reversible", None, None,
        "can trade back, but price and fees differ"),
}


def split_camel(name: str) -> list[str]:
    return re.findall(r"[A-Z][a-z0-9]*", name)


def verb_of(tool: str) -> str:
    words = split_camel(tool)
    return words[0] if words else tool


def object_of(tool: str) -> str:
    """Rough object of the tool name, for pairing inverses."""
    words = split_camel(tool)[1:]
    drop = {"Or", "Update", "To", "From", "Users", "User"}
    words = [w for w in words if w not in drop]
    obj = "".join(words)
    return obj[:-1] if obj.endswith("s") else obj


def find_inverse(tool: str, toolkit_tools: list[str]) -> str | None:
    verb = verb_of(tool)
    for inverse_verb in INVERSE_VERBS.get(verb, []):
        for other in toolkit_tools:
            if other == tool or verb_of(other) != inverse_verb:
                continue
            if object_of(other) == object_of(tool):
                return other
    return None


def classify(toolkit: str, tool: str, toolkit_tools: list[str]) -> tuple:
    if (toolkit, tool) in OVERRIDES:
        category, label, cost, inverse, rule = OVERRIDES[(toolkit, tool)]
        return category, label, cost, inverse, rule, category == "review", "override"

    verb = verb_of(tool)

    if verb in READ_ONLY_VERBS:
        return ("read_only", "reversible", 0, None,
                "does not change the remote state", False, f"verb:{verb}")

    inverse = find_inverse(tool, toolkit_tools)
    if inverse:
        return ("inverse_in_toolkit", "reversible", 1, toolkit + inverse,
                f"undone by {inverse}", False, f"inverse:{verb}")

    if verb in DELETE_VERBS:
        target = object_of(tool)
        recreate = next(
            (other for other in toolkit_tools
             if verb_of(other) in RECREATE_VERBS
             and (object_of(other) == target or target.startswith(object_of(other))
                  or object_of(other).startswith(target))),
            None,
        )
        if recreate:
            return ("delete", "partially_reversible", None, None,
                    f"permanent, but {recreate} can re-create it if the old "
                    "content is known (new id)", False, f"recreate:{verb}")
        return ("delete", "irreversible", None, None,
                "permanent in the none condition; nothing re-creates it", False,
                f"verb:{verb}")

    if verb in OVERWRITE_VERBS:
        return ("overwrite", "partially_reversible", None, None,
                "restorable only by writing back known old values", False,
                f"verb:{verb}")

    if verb in EXTERNAL_VERBS:
        return ("external_effect", "irreversible", None, None,
                "reaches other people or the outside world", False,
                f"verb:{verb}")

    return ("review", None, None, None, "no rule matched", True, "none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ToolEmu tool table.")
    parser.add_argument("--toolkits", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/toolemu/tool_table.csv"))
    args = parser.parse_args()

    toolkits = json.loads(args.toolkits.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    case_count = Counter(t for c in cases for t in c["Toolkits"])

    rows = []
    for tk in toolkits:
        name = tk["toolkit"]
        tool_names = [t["name"] for t in tk["tools"]]
        for tool in tk["tools"]:
            category, label, cost, inverse, rule, review, source = classify(
                name, tool["name"], tool_names
            )
            rows.append({
                "toolkit": name,
                "tool": tool["name"],
                "full_name": name + tool["name"],
                "toolkit_cases": case_count.get(name, 0),
                "auto_category": category,
                "auto_label": label or "",
                "auto_exact_cost": "" if cost is None else cost,
                "inverse_tool": inverse or "",
                "rule": rule,
                "rule_source": source,
                "needs_review": "yes" if review or label is None else "",
                "summary": tool["summary"],
                "parameters": "; ".join(
                    f"{p['name']} ({p['type']})" for p in tool["parameters"]
                ),
                "human_label": "",
                "human_notes": "",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{len(rows)} tools from {len(toolkits)} toolkits -> {args.out}")
    print("Categories:", dict(Counter(r["auto_category"] for r in rows)))
    print("Auto labels:", dict(Counter(r["auto_label"] or "(per call)" for r in rows)))
    print("Rule source:", dict(Counter(r["rule_source"].split(":")[0] for r in rows)))
    print("Needs review:", sum(r["needs_review"] == "yes" for r in rows))


if __name__ == "__main__":
    main()
