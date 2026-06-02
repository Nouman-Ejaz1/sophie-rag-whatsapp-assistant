"""
Global Trust Registry — loads all trusted site definitions.
Combines: global_core + regional (auto-detected from user location) + user_custom.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "trusted_sites"

class TrustRegistry:
    def __init__(self):
        self._scores: Dict[str, Tuple[int, str]] = {}  # domain -> (score, category)
        self._user_sites: Dict[str, dict] = {}
        self._loaded = False
    
    def load(self, regions: list = None):
        """
        Load all trust data. Call once at startup.
        regions: list of regional files to load, e.g. ["pakistan", "uk"]
        """
        self._scores = {}
        self._user_sites = {}
        
        # 1. Load global core
        self._load_file(DATA_DIR / "global_core.json")
        
        # 2. Load requested regional files
        for region in (regions or []):
            region_file = DATA_DIR / "regional" / f"{region.lower()}.json"
            if region_file.exists():
                self._load_file(region_file)
        
        # 3. Load user custom (highest priority — overrides everything)
        user_file = DATA_DIR / "user_custom.json"
        if user_file.exists():
            try:
                with open(user_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.get("sites", []):
                    domain = entry.get("domain", "").lower().strip()
                    if domain:
                        self._user_sites[domain] = entry
                        self._scores[domain] = (
                            entry.get("trust_score", 70),
                            "user_trusted"
                        )
            except Exception as e:
                print(f"[TrustRegistry] Could not load user_custom.json: {e}")
        
        self._loaded = True
        print(f"[TrustRegistry] Loaded {len(self._scores)} trusted domains.")
    
    def _load_file(self, path: Path):
        """Load a single trust file — nested category -> domain -> score."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            domains = data.get("domains", {})
            for category, domain_map in domains.items():
                for domain, score in domain_map.items():
                    domain = domain.lower().strip()
                    # Only override if new score is higher
                    existing_score, _ = self._scores.get(domain, (0, ""))
                    if score > existing_score:
                        self._scores[domain] = (score, category)
        except Exception as e:
            print(f"[TrustRegistry] Could not load {path}: {e}")
    
    def get_score(self, domain: str) -> Tuple[int, str]:
        """Returns (trust_score, category) for a domain. (0, 'unknown') if not found."""
        if not self._loaded:
            self.load()
        domain = domain.lower().strip()
        # Handle exact domain match
        if domain in self._scores:
            return self._scores[domain]
        
        # Fallback for subdomains: check if parent domain is trusted
        parts = domain.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            if parent in self._scores:
                return self._scores[parent]
                
        return 0, "unknown"
    
    def add_user_site(self, domain: str, trust_score: int = 70,
                      topics: list = None, note: str = "") -> bool:
        """Add or update a user-trusted site. Saves to user_custom.json."""
        user_file = DATA_DIR / "user_custom.json"
        try:
            domain = domain.lower().strip()
            if user_file.exists():
                with open(user_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"version": 1, "sites": []}
            
            # Remove existing entry for this domain if present
            data["sites"] = [s for s in data.get("sites", []) if s.get("domain") != domain]
            
            # Add new entry
            from datetime import datetime
            data["sites"].append({
                "domain": domain,
                "trust_score": trust_score,
                "topics": topics or [],
                "note": note,
                "added_at": datetime.utcnow().isoformat()
            })
            data["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
            
            user_file.parent.mkdir(parents=True, exist_ok=True)
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Reload registry
            self._user_sites[domain] = data["sites"][-1]
            self._scores[domain] = (trust_score, "user_trusted")
            return True
        except Exception as e:
            print(f"[TrustRegistry] add_user_site failed: {e}")
            return False
    
    def remove_user_site(self, domain: str) -> bool:
        """Remove a user-trusted site."""
        user_file = DATA_DIR / "user_custom.json"
        try:
            domain = domain.lower().strip()
            if not user_file.exists():
                return False
            with open(user_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            original_count = len(data.get("sites", []))
            data["sites"] = [s for s in data.get("sites", []) if s.get("domain") != domain]
            if len(data["sites"]) == original_count:
                return False  # Not found
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._scores.pop(domain, None)
            self._user_sites.pop(domain, None)
            return True
        except Exception as e:
            print(f"[TrustRegistry] remove_user_site failed: {e}")
            return False
    
    def list_user_sites(self) -> list:
        """Returns the user-trusted sites list."""
        if not self._loaded:
            self.load()
        return list(self._user_sites.values())
    
    def is_user_trusted(self, domain: str) -> bool:
        if not self._loaded:
            self.load()
        return domain.lower().strip() in self._user_sites

# Global instance — initialized at startup
trust_registry = TrustRegistry()
