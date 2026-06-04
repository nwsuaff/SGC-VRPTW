import csv
from collections import defaultdict

with open('e:/vrp/results/comparison_all_ortools_01/comparison_results.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

def parse_instance(name):
    parts = name.split('_')
    return parts[0], int(parts[1]), parts[2]

def is_ortools_feasible(r):
    return r["baseline_feasible"] == "True" and float(r["baseline_vehicles"]) > 0

def is_droc_feasible(r):
    return r["droc_success"] == "True" and float(r["droc_vehicles"]) > 0

def lexicographic_winner(r):
    dv = float(r["droc_vehicles"])
    dd = float(r["droc_distance"])
    bv = float(r["baseline_vehicles"])
    bd = float(r["baseline_distance"])
    if dv < bv:
        return "droc"
    elif dv > bv:
        return "baseline"
    elif dd < bd:
        return "droc"
    elif dd > bd:
        return "baseline"
    else:
        return "tie"

groups = defaultdict(list)
for r in rows:
    fam, sc, idx = parse_instance(r["instance"])
    groups[fam + "_" + str(sc)].append(r)

families = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
sizes = [2, 4, 6, 8]
size_labels = {2: "200", 4: "400", 6: "600", 8: "800"}

results = {}
for f in families:
    for sc in sizes:
        key = f + "_" + str(sc)
        grp = groups[key]
        n = len(grp)
        ort_feas = sum(1 for r in grp if is_ortools_feasible(r))
        droc_feas = sum(1 for r in grp if is_droc_feasible(r))
        both_feas = sum(1 for r in grp if is_ortools_feasible(r) and is_droc_feasible(r))
        droc_v_win = sum(1 for r in grp if is_ortools_feasible(r) and is_droc_feasible(r) and float(r["droc_vehicles"]) < float(r["baseline_vehicles"]))
        droc_d_win = sum(1 for r in grp if is_ortools_feasible(r) and is_droc_feasible(r) and float(r["droc_distance"]) < float(r["baseline_distance"]))
        lexi_w = {"droc": 0, "baseline": 0, "tie": 0}
        for r in grp:
            if is_ortools_feasible(r) and is_droc_feasible(r):
                lexi_w[lexicographic_winner(r)] += 1
        v_imps = [float(r["vehicle_improvement"]) for r in grp
                  if is_ortools_feasible(r) and is_droc_feasible(r)
                  and float(r["baseline_vehicles"]) > 0 and float(r["droc_vehicles"]) > 0
                  and r["vehicle_improvement"] != ""]
        avg_v_imp = sum(v_imps)/len(v_imps) if v_imps else None
        d_imps = [float(r["distance_improvement"]) for r in grp
                  if is_ortools_feasible(r) and is_droc_feasible(r)
                  and float(r["baseline_distance"]) > 0 and float(r["droc_distance"]) > 0
                  and r["distance_improvement"] != ""]
        avg_d_imp = sum(d_imps)/len(d_imps) if d_imps else None
        results[key] = {
            "n": n, "ort_feas": ort_feas, "droc_feas": droc_feas,
            "both_feas": both_feas, "droc_v_win": droc_v_win, "droc_d_win": droc_d_win,
            "lexi_droc": lexi_w["droc"], "lexi_baseline": lexi_w["baseline"], "lexi_tie": lexi_w["tie"],
            "avg_v_imp": avg_v_imp, "avg_d_imp": avg_d_imp
        }

# ---- Print tables ----
hdr = "| {group:12s} | {n:>3s} | {of:>8s} | {df:>8s} | {bf:>9s} | {vw:>9s} | {dw:>9s} | {ld:>9s} | {lb:>8s} | {lt:>8s} | {vi:>10s} | {di:>10s} |"
sep = "|" + "|".join(["-"*12] + ["-"*8]*3 + ["-"*9]*2 + ["-"*9]*2 + ["-"*8]*2 + ["-"*10]*2) + "|"

print("=" * 120)
print("PER FAMILY-SCALE DETAILED RESULTS")
print("=" * 120)
print(hdr.format(group="Group", n="N", of="ORT Feas", df="DRoC Feas", bf="Both Feas", vw="DRoC V-Win", dw="DRoC D-Win", ld="Lexi DRoC", lb="Lexi Base", lt="Lexi Tie", vi="Avg V-Imp%", di="Avg D-Imp%"))
print(sep)
for f in families:
    for sc in sizes:
        key = f + "_" + str(sc)
        r = results[key]
        avg_v = "{:.2f}".format(r["avg_v_imp"]) if r["avg_v_imp"] is not None else "N/A"
        avg_d = "{:.2f}".format(r["avg_d_imp"]) if r["avg_d_imp"] is not None else "N/A"
        print(hdr.format(group=key, n=str(r["n"]), of=str(r["ort_feas"]), df=str(r["droc_feas"]), bf=str(r["both_feas"]), vw=str(r["droc_v_win"]), dw=str(r["droc_d_win"]), ld=str(r["lexi_droc"]), lb=str(r["lexi_baseline"]), lt=str(r["lexi_tie"]), vi=avg_v, di=avg_d))

print()
print("=" * 120)
print("TIGHT vs LOOSE SUMMARY (by family group)")
print("=" * 120)
for grp_name, grp_families in [("tight (seen)", ["C1","R1","RC1"]), ("loose (unseen)", ["C2","R2","RC2"])]:
    t_n=0; t_of=0; t_df=0; t_bf=0; t_vw=0; t_dw=0; t_ld=0; t_lb=0; t_lt=0
    v_all=[]; d_all=[]
    for f in grp_families:
        for sc in sizes:
            key = f + "_" + str(sc)
            res = results[key]
            t_n+=res["n"]; t_of+=res["ort_feas"]; t_df+=res["droc_feas"]
            t_bf+=res["both_feas"]; t_vw+=res["droc_v_win"]; t_dw+=res["droc_d_win"]
            t_ld+=res["lexi_droc"]; t_lb+=res["lexi_baseline"]; t_lt+=res["lexi_tie"]
            for row in groups[key]:
                if is_ortools_feasible(row) and is_droc_feasible(row):
                    if row["vehicle_improvement"] != "" and float(row["baseline_vehicles"]) > 0:
                        v_all.append(float(row["vehicle_improvement"]))
                    if row["distance_improvement"] != "" and float(row["baseline_distance"]) > 0:
                        d_all.append(float(row["distance_improvement"]))
    av = sum(v_all)/len(v_all) if v_all else None
    ad = sum(d_all)/len(d_all) if d_all else None
    print("Group: {}  (families: {})".format(grp_name, grp_families))
    print("  N={}, ORT_Feas={}, DRoC_Feas={}, Both_Feas={}".format(t_n, t_of, t_df, t_bf))
    print("  DRoC V-Win={}, DRoC D-Win={}".format(t_vw, t_dw))
    print("  Lexi: DRoC={}, Baseline={}, Tie={}".format(t_ld, t_lb, t_lt))
    print("  Avg V-Imp={}, Avg D-Imp={}".format("{:.2f}%".format(av) if av else "N/A", "{:.2f}%".format(ad) if ad else "N/A"))
    print()

print("=" * 120)
print("SIZE SUMMARY (all families merged per size)")
print("=" * 120)
for sc in sizes:
    t_n=0; t_of=0; t_df=0; t_bf=0; t_vw=0; t_dw=0; t_ld=0; t_lb=0; t_lt=0
    v_all=[]; d_all=[]
    for f in families:
        key = f + "_" + str(sc)
        res = results[key]
        t_n+=res["n"]; t_of+=res["ort_feas"]; t_df+=res["droc_feas"]
        t_bf+=res["both_feas"]; t_vw+=res["droc_v_win"]; t_dw+=res["droc_d_win"]
        t_ld+=res["lexi_droc"]; t_lb+=res["lexi_baseline"]; t_lt+=res["lexi_tie"]
        for row in groups[key]:
            if is_ortools_feasible(row) and is_droc_feasible(row):
                if row["vehicle_improvement"] != "" and float(row["baseline_vehicles"]) > 0:
                    v_all.append(float(row["vehicle_improvement"]))
                if row["distance_improvement"] != "" and float(row["baseline_distance"]) > 0:
                    d_all.append(float(row["distance_improvement"]))
    av = sum(v_all)/len(v_all) if v_all else None
    ad = sum(d_all)/len(d_all) if d_all else None
    print("Size {} nodes (scale={}):".format(size_labels[sc], sc))
    print("  N={}, ORT_Feas={}, DRoC_Feas={}, Both_Feas={}".format(t_n, t_of, t_df, t_bf))
    print("  DRoC V-Win={}, DRoC D-Win={}".format(t_vw, t_dw))
    print("  Lexi: DRoC={}, Baseline={}, Tie={}".format(t_ld, t_lb, t_lt))
    print("  Avg V-Imp={}, Avg D-Imp={}".format("{:.2f}%".format(av) if av else "N/A", "{:.2f}%".format(ad) if ad else "N/A"))
    print()

print("=" * 120)
print("GRAND TOTAL")
print("=" * 120)
t_n=sum(results[k]["n"] for k in results)
t_of=sum(results[k]["ort_feas"] for k in results)
t_df=sum(results[k]["droc_feas"] for k in results)
t_bf=sum(results[k]["both_feas"] for k in results)
t_vw=sum(results[k]["droc_v_win"] for k in results)
t_dw=sum(results[k]["droc_d_win"] for k in results)
t_ld=sum(results[k]["lexi_droc"] for k in results)
t_lb=sum(results[k]["lexi_baseline"] for k in results)
t_lt=sum(results[k]["lexi_tie"] for k in results)
v_all=[]; d_all=[]
for k in results:
    for row in groups[k]:
        if is_ortools_feasible(row) and is_droc_feasible(row):
            if row["vehicle_improvement"] != "" and float(row["baseline_vehicles"]) > 0:
                v_all.append(float(row["vehicle_improvement"]))
            if row["distance_improvement"] != "" and float(row["baseline_distance"]) > 0:
                d_all.append(float(row["distance_improvement"]))
av = sum(v_all)/len(v_all) if v_all else None
ad = sum(d_all)/len(d_all) if d_all else None
print("N={}, ORT_Feas={}, DRoC_Feas={}, Both_Feas={}".format(t_n, t_of, t_df, t_bf))
print("DRoC V-Win={}, DRoC D-Win={}".format(t_vw, t_dw))
print("Lexi: DRoC={}, Baseline={}, Tie={}".format(t_ld, t_lb, t_lt))
print("Avg V-Imp={}, Avg D-Imp={}".format("{:.2f}%".format(av) if av else "N/A", "{:.2f}%".format(ad) if ad else "N/A"))
