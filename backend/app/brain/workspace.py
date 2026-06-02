import os
import subprocess
import sys
from typing import Dict, Any, List
from pathlib import Path

class WorkspaceSandbox:
    """
    Isolated execution workspace for running Python scripts and managing files
    within a sandboxed directory `./jarvis_workspace`.
    """
    def __init__(self, workspace_name: str = "jarvis_workspace"):
        # Root directory of workspace inside backend
        self.workspace_dir = Path(__file__).resolve().parent.parent.parent / workspace_name
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, filename: str) -> Path:
        """Helper to resolve filename within sandbox, preventing path traversal attacks."""
        resolved = (self.workspace_dir / filename).resolve()
        # Verify the resolved path is inside the sandbox directory
        if not str(resolved).startswith(str(self.workspace_dir.resolve())):
            raise ValueError(f"Path traversal detected! File path '{filename}' is outside sandbox boundaries.")
        return resolved

    def write_file(self, filename: str, content: str) -> str:
        """Writes a file to the sandboxed workspace."""
        try:
            target_path = self._resolve_path(filename)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to '{filename}' inside the workspace."
        except Exception as e:
            return f"Error writing file '{filename}': {str(e)}"

    def read_file(self, filename: str) -> str:
        """Reads a file from the sandboxed workspace."""
        try:
            target_path = self._resolve_path(filename)
            if not target_path.exists():
                return f"Error: File '{filename}' does not exist inside the workspace."
            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file '{filename}': {str(e)}"

    def list_files(self) -> List[str]:
        """Lists all files recursively in the workspace sandbox."""
        files = []
        try:
            for root, _, filenames in os.walk(self.workspace_dir):
                for f in filenames:
                    abs_path = Path(root) / f
                    rel_path = abs_path.relative_to(self.workspace_dir)
                    files.append(str(rel_path))
            return files
        except Exception as e:
            print(f"Error listing files in workspace: {e}")
            return []

    def run_python(self, code: str, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        """
        Executes a block of Python code inside the sandbox workspace,
        returning stdout, stderr, and the return code. Protected by a timeout.
        """
        temp_script = "sandbox_exec.py"
        script_path = self._resolve_path(temp_script)
        
        # Write the code block to the sandbox script file
        self.write_file(temp_script, code)
        
        try:
            # Execute python script in a subprocess using the current environment interpreter
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(self.workspace_dir)
            )
            
            # Clean up the script file immediately
            if script_path.exists():
                script_path.unlink()
                
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired as te:
            if script_path.exists():
                script_path.unlink()
            return {
                "success": False,
                "stdout": te.stdout or "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds.",
                "exit_code": -1
            }
        except Exception as e:
            if script_path.exists():
                script_path.unlink()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error running script: {str(e)}",
                "exit_code": -1
            }

workspace_sandbox = WorkspaceSandbox()
