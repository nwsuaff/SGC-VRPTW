# Data

Large raw benchmark files are not included in this repository. The public package includes a toy JSON instance, compact processed result records, figure-source tables, and summary tables used by the manuscript.

## Included Data

```text
data/toy/vrptw_tiny.json
data/toy/expected_tiny_solution.json
data/manifests/
results/official/
evidence/figure_sources/
evidence/tables/
```

The included records are sufficient to inspect the reported aggregations and regenerate the manuscript-level evidence figures after the plotting environment is configured.

## Raw Benchmark Layout

Place downloaded or locally generated raw instances under:

```text
data/raw/VRPTW/
```

The original project used VRPTW benchmark families from Solomon-style, Gehring-Homberger/Homberger-style, and ORTEC-style instances. Keep the raw directory structure consistent with the loader or experiment script you run. Typical layouts are:

```text
data/raw/VRPTW/Solomon/
data/raw/VRPTW/GH200/
data/raw/VRPTW/GH400/
data/raw/VRPTW/GH600/
data/raw/VRPTW/GH800/
data/raw/VRPTW/GH1000/
data/raw/VRPTW/ORTEC/static/
data/raw/VRPTW/ORTEC/dynamic/
```

Some scripts also accept processed or project-specific paths through command-line arguments. Prefer passing explicit paths rather than editing source code.

## Data Policy

Raw benchmark archives are excluded to keep the repository lightweight and to avoid redistributing datasets whose hosting terms may differ from this code repository. The included processed result records and figure-source tables are the auditable inputs for the paper figures and tables.
