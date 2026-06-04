"""Dataset registry for managing available datasets."""

from pathlib import Path
from typing import Optional
import json


class DatasetRegistry:
    """Registry of available VRPTW datasets."""
    
    KNOWN_DATASETS = {
        "solomon": {
            "name": "Solomon-100",
            "description": "Solomon benchmark instances with 100 customers",
            "families": ["C1", "C2", "R1", "R2", "RC1", "RC2"],
            "source_dir": "data/VRPTW/Solomon",
        },
        "homberger": {
            "name": "Homberger-800",
            "description": "Homberger benchmark instances with 800 customers",
            "families": ["C1", "C2", "R1", "R2", "RC1", "RC2"],
            "source_dir": "data/VRPTW/GH800",
        },
        "ortec": {
            "name": "ORTEC-Competition",
            "description": "EURO Meets NeurIPS 2022 ORTEC competition instances (200-1000 customers)",
            "families": ["ORTEC_S", "ORTEC_M", "ORTEC_L", "ORTEC_XL"],
            "source_dir": "data/VRPTW/ORTEC/static",
        },
        "ortec_synthetic": {
            "name": "ORTEC-Synthetic",
            "description": "Synthetically generated ORTEC-style instances",
            "families": ["ORTEC_S", "ORTEC_M", "ORTEC_L", "ORTEC_XL"],
            "source_dir": "data/VRPTW/ORTEC/static",
        },
        "toy": {
            "name": "Toy",
            "description": "Small test instances",
            "families": ["debug"],
            "source_dir": "data/toy",
        },
    }
    
    def __init__(self):
        """Initialize the registry with loaded instances."""
        self._instances: dict[str, dict] = {}
        self._by_source: dict[str, list[str]] = {}
        self._by_family: dict[str, list[str]] = {}
        self._by_size: dict[int, list[str]] = {}
    
    def __len__(self) -> int:
        """Return number of loaded instances."""
        return len(self._instances)
    
    def load_from_json(self, file_path: str | Path) -> bool:
        """Load a single instance from a JSON file.
        
        Args:
            file_path: Path to the JSON instance file.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            name = data.get("name", path.stem)
            self._instances[name] = data
            
            source = data.get("source", "unknown")
            if source not in self._by_source:
                self._by_source[source] = []
            self._by_source[source].append(name)
            
            family = data.get("family")
            if family:
                if family not in self._by_family:
                    self._by_family[family] = []
                self._by_family[family].append(name)
            
            size = data.get("size", 0)
            if size not in self._by_size:
                self._by_size[size] = []
            self._by_size[size].append(name)
            
            return True
        except Exception:
            return False
    
    def load_from_directory(self, dir_path: str | Path, pattern: str = "*.json") -> int:
        """Load all instances from a directory.
        
        Args:
            dir_path: Directory containing instance files.
            pattern: Glob pattern for matching files.
            
        Returns:
            Number of instances loaded.
        """
        path = Path(dir_path)
        if not path.exists():
            return 0
        
        count = 0
        for f in path.glob(pattern):
            if f.is_file() and self.load_from_json(f):
                count += 1
        return count
    
    def get_instance(self, name: str) -> Optional[dict]:
        """Get a loaded instance by name.
        
        Args:
            name: Instance name.
            
        Returns:
            Instance data dict or None if not found.
        """
        return self._instances.get(name)
    
    def filter_by_source(self, source: str) -> list[dict]:
        """Filter instances by source.
        
        Args:
            source: Data source (e.g., 'solomon', 'toy').
            
        Returns:
            List of instance data dicts.
        """
        names = self._by_source.get(source, [])
        return [self._instances[n] for n in names if n in self._instances]
    
    def filter_by_family(self, family: str) -> list[dict]:
        """Filter instances by family.
        
        Args:
            family: Instance family (e.g., 'C1', 'R2').
            
        Returns:
            List of instance data dicts.
        """
        names = self._by_family.get(family, [])
        return [self._instances[n] for n in names if n in self._instances]
    
    def filter_by_size(self, size: int) -> list[dict]:
        """Filter instances by number of customers.
        
        Args:
            size: Exact number of customers.
            
        Returns:
            List of instance data dicts.
        """
        names = self._by_size.get(size, [])
        return [self._instances[n] for n in names if n in self._instances]
    
    def list_names(self) -> list[str]:
        """List all loaded instance names."""
        return list(self._instances.keys())
    
    def get_dataset_info(self, name: str) -> Optional[dict]:
        """Get information about a known dataset."""
        return self.KNOWN_DATASETS.get(name)
    
    def list_datasets(self) -> list[str]:
        """List all known dataset names."""
        return list(self.KNOWN_DATASETS.keys())
    
    def get_source_dir(self, name: str) -> Path:
        """Get the source directory for a dataset."""
        info = self.KNOWN_DATASETS.get(name)
        if info:
            return Path(info["source_dir"])
        return Path("data")
