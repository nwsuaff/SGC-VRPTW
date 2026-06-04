"""Batch generate dynamic ORTEC scenarios from static instances."""
import json, math, random, sys
sys.path.insert(0, ".")
from pathlib import Path
from src.utils.io import read_instance

static_dir = Path("data/VRPTW/ORTEC/static")
output_dir = Path("data/VRPTW/ORTEC/dynamic")
output_dir.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 5
txt_files = sorted(static_dir.glob("ORTEC-SYNTH-*.txt"))
skip_stems = {p.stem for p in output_dir.glob("*_dyn*.json")}
to_generate = [p for p in txt_files if p.stem not in skip_stems]

print(f"Static: {len(txt_files)} | Existing: {len(skip_stems)} | To gen: {len(to_generate)}")

for i, txt_path in enumerate(to_generate, 1):
    out_path = output_dir / f"{txt_path.stem}_dyn{NUM_EPOCHS}e.json"
    inst = read_instance(str(txt_path))

    dm = inst.duration_matrix or inst.distance_matrix
    if not dm:
        n = inst.size + 1
        dm = [[0.0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                dx = inst.x_coords.get(r, 0) - inst.x_coords.get(c, 0)
                dy = inst.y_coords.get(r, 0) - inst.y_coords.get(c, 0)
                dm[r][c] = math.sqrt(dx * dx + dy * dy)

    rng = random.Random(hash(inst.name) & 0xFFFFFFFF)
    EPOCH_DURATION, MARGIN_DISPATCH = 3600, 3600
    depot_due = inst.due_time.get(0, 999999)

    orders = [{
        "request_id": 0, "customer_idx": 0,
        "x": float(inst.x_coords.get(0, 0.0)),
        "y": float(inst.y_coords.get(0, 0.0)),
        "demand": 0,
        "ready_time": inst.ready_time.get(0, 0),
        "due_time": depot_due,
        "service_time": inst.service_time.get(0, 0),
        "release_epoch": 0, "is_dispatched": False, "must_dispatch": False,
    }]

    all_cust = [
        {"customer_idx": c,
         "demand": inst.demand.get(c, 0),
         "service_time": inst.service_time.get(c, 0),
         "ready_time": inst.ready_time.get(c, 0),
         "due_time": inst.due_time.get(c, 0),
         "x": float(inst.x_coords.get(c, 0.0)),
         "y": float(inst.y_coords.get(c, 0.0))}
        for c in inst.customer_ids
    ]
    rng.shuffle(all_cust)

    by_epoch = [[] for _ in range(NUM_EPOCHS)]
    for idx, o in enumerate(all_cust):
        by_epoch[idx % NUM_EPOCHS].append(o)

    req_id = 1
    for epoch in range(NUM_EPOCHS):
        for o in by_epoch[epoch]:
            must = False
            if epoch < NUM_EPOCHS - 1:
                t_next = (epoch + 1) * EPOCH_DURATION + MARGIN_DISPATCH
                t_arrival = max(t_next + dm[0][o["customer_idx"]], o["ready_time"])
                must = t_arrival > o["due_time"] or (
                    t_arrival + o["service_time"] + dm[o["customer_idx"]][0] > depot_due
                )
            orders.append({
                "request_id": req_id,
                "customer_idx": o["customer_idx"],
                "x": o["x"], "y": o["y"],
                "demand": o["demand"],
                "ready_time": o["ready_time"],
                "due_time": o["due_time"],
                "service_time": o["service_time"],
                "release_epoch": epoch,
                "is_dispatched": False,
                "must_dispatch": must,
            })
            req_id += 1

    scenario = {
        "name": f"{inst.name}_dyn{NUM_EPOCHS}e",
        "base_instance_name": inst.name,
        "source": "ortec_synthetic",
        "num_epochs": NUM_EPOCHS,
        "epoch_duration": EPOCH_DURATION,
        "margin_dispatch": MARGIN_DISPATCH,
        "num_orders_total": len(orders),
        "num_orders_per_epoch": len(all_cust) // NUM_EPOCHS,
        "orders": orders,
        "duration_matrix": dm,
        "vehicle_capacity": inst.vehicle_capacity,
        "vehicle_count": inst.vehicle_count,
        "metadata": {"generated_from": inst.name, "seed": rng.randint(0, 2**31 - 1)},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, default=str)

    if i % 10 == 0 or i == 1:
        print(f"[{i}/{len(to_generate)}] {out_path.name}")

print(f"Done! Total dynamic scenarios: {len(list(output_dir.glob('*_dyn*.json')))}")
