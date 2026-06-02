import datetime
import os
import re
import subprocess
import uuid
from typing import Dict, Optional

from app.database import db


DELETE_COMMAND_RE = re.compile(
    r"(^|[;&|]\s*)("
    r"del|erase|rd|rmdir|rm|remove-item|unlink"
    r")(\s|$)",
    re.IGNORECASE,
)


def get_workspace_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def is_destructive_command(command_line: str) -> bool:
    if not command_line:
        return False
    normalized = command_line.strip().lower()
    if DELETE_COMMAND_RE.search(normalized):
        return True
    return bool(re.search(r"\b(remove-item|rm)\b.*\s-(r|recurse|force)\b", normalized, re.I))


def describe_delete_target(command_line: str) -> str:
    if not command_line:
        return "unknown"
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", command_line)
    if quoted:
        return quoted[-1]

    tokens = command_line.strip().split()
    filtered = [
        t for t in tokens[1:]
        if not t.startswith("-")
        and t.lower() not in {"-r", "-rf", "/s", "/q", "/f", "recurse", "force"}
    ]
    return filtered[-1] if filtered else command_line.strip()


def create_pending_approval(
    sender: str,
    action_type: str,
    command_line: str,
    target_path: Optional[str] = None,
    ttl_minutes: int = 15,
) -> Dict[str, str]:
    approval_id = f"apr_{uuid.uuid4().hex[:8]}"
    expires_at = (datetime.datetime.utcnow() + datetime.timedelta(minutes=ttl_minutes)).isoformat()
    target = target_path or describe_delete_target(command_line)
    ok = db.create_pending_approval(
        approval_id=approval_id,
        sender=sender or "unknown_sender",
        action_type=action_type,
        command_line=command_line,
        target_path=target,
        expires_at=expires_at,
    )
    if not ok:
        return {
            "status": "error",
            "message": "I could not create the approval request, so I did not run the destructive action."
        }
    return {
        "status": "pending",
        "id": approval_id,
        "sender": sender or "unknown_sender",
        "action_type": action_type,
        "command_line": command_line,
        "target_path": target,
        "expires_at": expires_at,
        "message": (
            "Confirmation needed before I delete anything.\n\n"
            f"Approval ID: {approval_id}\n"
            f"Action: {action_type}\n"
            f"Target: {target}\n"
            f"Command: {command_line}\n\n"
            f"Reply `approve {approval_id}` to run it, or `cancel {approval_id}` to stop it."
        )
    }


def run_shell_command(command_line: str, timeout: int = 30) -> str:
    base_dir = get_workspace_dir()
    try:
        process = subprocess.Popen(
            ["powershell", "-Command", command_line],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        stdout, stderr = process.communicate(timeout=timeout)

        output_str = ""
        if stdout:
            output_str += f"STDOUT:\n{stdout}\n"
        if stderr:
            output_str += f"STDERR:\n{stderr}\n"
        if not output_str:
            output_str = "Command completed successfully with no terminal output."
        return f"Command exited with status code {process.returncode}.\n{output_str}"
    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Failed to execute terminal command '{command_line}': {str(e)}"


def resolve_approval_command(sender: str, message: str) -> Optional[Dict[str, str]]:
    match = re.match(r"^\s*(approve|cancel)\s+(apr_[a-f0-9]{8})\s*$", message or "", re.I)
    if not match:
        return None

    action = match.group(1).lower()
    approval_id = match.group(2).lower()
    approval = db.get_pending_approval(approval_id)
    if not approval:
        return {"status": "not_found", "message": f"I could not find approval `{approval_id}`."}

    if approval["sender"] != sender:
        return {
            "status": "wrong_sender",
            "message": "I can only accept that approval from the same WhatsApp chat that requested it."
        }

    if approval["status"] != "pending":
        return {
            "status": approval["status"],
            "message": f"Approval `{approval_id}` is already `{approval['status']}`."
        }

    now = datetime.datetime.utcnow()
    try:
        expires_at = datetime.datetime.fromisoformat(approval["expires_at"])
    except Exception:
        expires_at = now
    if now > expires_at:
        db.update_pending_approval(approval_id, "expired", "Approval expired before execution.")
        return {"status": "expired", "message": f"Approval `{approval_id}` expired. Please ask me again if you still want this."}

    if action == "cancel":
        db.update_pending_approval(approval_id, "cancelled", "Cancelled by user.")
        return {"status": "cancelled", "message": f"Cancelled `{approval_id}`. I did not run the destructive action."}

    if approval["action_type"] == "delete_agent":
        ok = db.delete_custom_agent(approval["target_path"])
        result = f"Deleted custom agent '{approval['target_path']}'." if ok else f"Could not delete custom agent '{approval['target_path']}'."
    else:
        result = run_shell_command(approval["command_line"])

    db.update_pending_approval(approval_id, "executed", result)
    return {
        "status": "executed",
        "message": (
            f"Approved and executed `{approval_id}`.\n\n"
            f"Action: {approval['action_type']}\n"
            f"Target: {approval['target_path']}\n\n"
            f"{result}"
        )
    }
