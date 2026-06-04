"""Generate ORTEC synthetic instances directly (bypasses CLI imports)."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.data.dynamic_ortec_generator import generate_ortec_benchmark_suite

output_dir = Path("data/VRPTW/ORTEC/static")
output_dir.mkdir(parents=True, exist_ok=True)

instances = generate_ortec_benchmark_suite(
    sizes=[300, 500, 800],
    instances_per_size=3,
    output_dir=output_dir,
    start_seed=42,
)
print(f"Generated {len(instances)} instances")
