from typing import Dict, Any, List
from app.brain.workspace import workspace_sandbox

class WorkspaceService:
    """
    Decoupled service orchestrating code evaluation and file management
    inside the secure './jarvis_workspace' sandbox.
    """
    @staticmethod
    def execute_python(code: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Runs Python scripts in an isolated subprocess with strict CPU/memory limits."""
        return workspace_sandbox.run_python(code, timeout_seconds=timeout)

    @staticmethod
    def read_file(filename: str) -> str:
        """Reads files from the sandboxed environment with path traversal guards."""
        return workspace_sandbox.read_file(filename)

    @staticmethod
    def write_file(filename: str, content: str) -> str:
        """Writes files into the sandboxed environment with path traversal guards."""
        return workspace_sandbox.write_file(filename, content)

    @staticmethod
    def list_files() -> List[str]:
        """Lists files inside the sandboxed environment recursively."""
        return workspace_sandbox.list_files()

workspace_service = WorkspaceService()
