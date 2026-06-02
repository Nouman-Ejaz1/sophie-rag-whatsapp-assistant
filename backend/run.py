import uvicorn
import os
import sys

# Ensure the parent directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import HOST, PORT

if __name__ == "__main__":
    print("Launching SentinelAI API Server...")
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_excludes=[
            "test_sandbox_workspace/*",
            "jarvis_workspace/*",
            "test_jarvis.py",
            "test_brain.py",
            "*.db",
            "local_data/*",
            "node_modules/*",
        ],
    )
