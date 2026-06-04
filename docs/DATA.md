# Data

This repository provides a toy JSON instance, compact processed result records, figure-source tables, and summary tables used by the manuscript. Benchmark loaders and experiment scripts also support standard VRPTW datasets arranged under the layout below.

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

## Benchmark Layout

Place benchmark instances under:

```text
data/raw/VRPTW/
```

The experiments use VRPTW benchmark families from Solomon-style, Gehring-Homberger/Homberger-style, and ORTEC-style instances. Keep the directory structure consistent with the loader or experiment script you run. Typical layouts are:

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

## Paper-Level Evidence

The processed result records and figure-source tables are the auditable inputs for the paper figures and tables. They can be inspected directly or used with the plotting scripts listed in `docs/USAGE.md`.
