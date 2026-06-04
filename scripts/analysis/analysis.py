import csv
import re
from collections import defaultdict

rows = []
with open(r'e:\vrp\results\comparison_all_ortools_01\comparison_results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def get_scale(instance):
    parts = instance.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except:
            return None
    return None

def is_baseline_feasible(row):
    return row['baseline_feasible'] == 'True' and float(row['baseline_vehicles']) > 0

def is_droc_feasible(row):
    return row['droc_success'] == 'True' and float(row['droc_vehicles']) > 0

scale_data = defaultdict(list)
for row in rows:
    scale = get_scale(row['instance'])
    if scale:
        scale_data[scale].append(row)

scale_labels = {2: '200', 4: '400', 6: '600', 8: '800'}
results_table = []

for scale in sorted(scale_data.keys()):
    data = scale_data[scale]
    n = len(data)
    ort_feas = [r for r in data if is_baseline_feasible(r)]
    droc_feas = [r for r in data if is_droc_feasible(r)]
    both_feas = [r for r in data if is_baseline_feasible(r) and is_droc_feasible(r)]
    droc_vehicle_wins = [r for r in both_feas if float(r['droc_vehicles']) < float(r['baseline_vehicles'])]
    droc_distance_wins = [r for r in both_feas if float(r['droc_distance']) < float(r['baseline_distance'])]
    
    avg_droc_v = sum(float(r['droc_vehicles']) for r in both_feas) / len(both_feas) if both_feas else 0
    avg_ort_v = sum(float(r['baseline_vehicles']) for r in both_feas) / len(both_feas) if both_feas else 0
    avg_droc_d = sum(float(r['droc_distance']) for r in both_feas) / len(both_feas) if both_feas else 0
    avg_ort_d = sum(float(r['baseline_distance']) for r in both_feas) / len(both_feas) if both_feas else 0
    veh_improve_pct = (avg_ort_v - avg_droc_v) / avg_ort_v * 100 if avg_ort_v > 0 else 0
    dist_improve_pct = (avg_ort_d - avg_droc_d) / avg_ort_d * 100 if avg_ort_d > 0 else 0
    
    label = scale_labels[scale]
    results_table.append({
        'scale': scale, 'label': label, 'n': n,
        'ort_feas': len(ort_feas), 'droc_feas': len(droc_feas), 'both_feas': len(both_feas),
        'droc_v_wins': len(droc_vehicle_wins), 'droc_d_wins': len(droc_distance_wins),
        'avg_droc_v': avg_droc_v, 'avg_ort_v': avg_ort_v,
        'avg_droc_d': avg_droc_d, 'avg_ort_d': avg_ort_d,
        'veh_improve_pct': veh_improve_pct, 'dist_improve_pct': dist_improve_pct,
    })

print('=' * 80)
print('Homberger 240-Instance Comparison: DRoC vs OR-Tools (comparison_all_ortools_01)')
print('=' * 80)

for r in results_table:
    print()
    print('--- Scale ' + r['label'] + ' nodes (scale=' + str(r['scale']) + ') ---')
    print('  Total instances:           ' + str(r['n']))
    print('  OR-Tools feasible:         ' + str(r['ort_feas']) + '/' + str(r['n']) + '  (' + str(round(r['ort_feas']/r['n']*100,1)) + '%)')
    print('  DRoC feasible:             ' + str(r['droc_feas']) + '/' + str(r['n']) + '  (' + str(round(r['droc_feas']/r['n']*100,1)) + '%)')
    print('  Both feasible:             ' + str(r['both_feas']) + '/' + str(r['n']) + '  (' + str(round(r['both_feas']/r['n']*100,1)) + '%)')
    print('  DRoC vehicle wins:         ' + str(r['droc_v_wins']) + '/' + str(r['both_feas']) + '  (' + str(round(r['droc_v_wins']/r['both_feas']*100,1)) + '%)')
    print('  DRoC distance wins:        ' + str(r['droc_d_wins']) + '/' + str(r['both_feas']) + '  (' + str(round(r['droc_d_wins']/r['both_feas']*100,1)) + '%)')
    print('  Avg DRoC vehicles:          ' + str(round(r['avg_droc_v'],4)))
    print('  Avg OR-Tools vehicles:      ' + str(round(r['avg_ort_v'],4)))
    print('  Avg DRoC distance:          ' + str(round(r['avg_droc_d'],4)))
    print('  Avg OR-Tools distance:      ' + str(round(r['avg_ort_d'],4)))
    print('  Vehicle improvement:         ' + str(round(r['veh_improve_pct'],4)) + '%')
    print('  Distance improvement:        ' + str(round(r['dist_improve_pct'],4)) + '%')

all_data = rows
all_ort_feas = [r for r in all_data if is_baseline_feasible(r)]
all_droc_feas = [r for r in all_data if is_droc_feasible(r)]
all_both_feas = [r for r in all_data if is_baseline_feasible(r) and is_droc_feasible(r)]
all_v_wins = [r for r in all_both_feas if float(r['droc_vehicles']) < float(r['baseline_vehicles'])]
all_d_wins = [r for r in all_both_feas if float(r['droc_distance']) < float(r['baseline_distance'])]
all_avg_droc_v = sum(float(r['droc_vehicles']) for r in all_both_feas) / len(all_both_feas)
all_avg_ort_v = sum(float(r['baseline_vehicles']) for r in all_both_feas) / len(all_both_feas)
all_avg_droc_d = sum(float(r['droc_distance']) for r in all_both_feas) / len(all_both_feas)
all_avg_ort_d = sum(float(r['baseline_distance']) for r in all_both_feas) / len(all_both_feas)
all_veh_imp = (all_avg_ort_v - all_avg_droc_v) / all_avg_ort_v * 100
all_dist_imp = (all_avg_ort_d - all_avg_droc_d) / all_avg_ort_d * 100

print()
print('=' * 80)
print('OVERALL (all 240 instances)')
print('=' * 80)
print('  Total instances:           ' + str(len(all_data)))
print('  OR-Tools feasible:         ' + str(len(all_ort_feas)) + '/' + str(len(all_data)) + '  (' + str(round(len(all_ort_feas)/len(all_data)*100,1)) + '%)')
print('  DRoC feasible:             ' + str(len(all_droc_feas)) + '/' + str(len(all_data)) + '  (' + str(round(len(all_droc_feas)/len(all_data)*100,1)) + '%)')
print('  Both feasible:             ' + str(len(all_both_feas)) + '/' + str(len(all_data)) + '  (' + str(round(len(all_both_feas)/len(all_data)*100,1)) + '%)')
print('  DRoC vehicle wins:          ' + str(len(all_v_wins)) + '/' + str(len(all_both_feas)) + '  (' + str(round(len(all_v_wins)/len(all_both_feas)*100,1)) + '%)')
print('  DRoC distance wins:         ' + str(len(all_d_wins)) + '/' + str(len(all_both_feas)) + '  (' + str(round(len(all_d_wins)/len(all_both_feas)*100,1)) + '%)')
print('  Avg DRoC vehicles:           ' + str(round(all_avg_droc_v,4)))
print('  Avg OR-Tools vehicles:      ' + str(round(all_avg_ort_v,4)))
print('  Avg DRoC distance:           ' + str(round(all_avg_droc_d,4)))
print('  Avg OR-Tools distance:       ' + str(round(all_avg_ort_d,4)))
print('  Vehicle improvement:         ' + str(round(all_veh_imp,4)) + '%')
print('  Distance improvement:        ' + str(round(all_dist_imp,4)) + '%')

print()
print('=' * 80)
print('BREAKDOWN BY PROBLEM TYPE')
print('=' * 80)
types = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
for t in types:
    t_rows = [r for r in rows if r['instance'].startswith(t + '_')]
    t_both = [r for r in t_rows if is_baseline_feasible(r) and is_droc_feasible(r)]
    t_vwins = [r for r in t_both if float(r['droc_vehicles']) < float(r['baseline_vehicles'])]
    t_dwins = [r for r in t_both if float(r['droc_distance']) < float(r['baseline_distance'])]
    if t_both:
        at_dv = sum(float(r['droc_vehicles']) for r in t_both) / len(t_both)
        at_ov = sum(float(r['baseline_vehicles']) for r in t_both) / len(t_both)
        at_dd = sum(float(r['droc_distance']) for r in t_both) / len(t_both)
        at_od = sum(float(r['baseline_distance']) for r in t_both) / len(t_both)
        v_imp = (at_ov - at_dv) / at_ov * 100
        d_imp = (at_od - at_dd) / at_od * 100
    else:
        at_dv = at_ov = at_dd = at_od = v_imp = d_imp = 0.0
    print('  ' + t + ': both_feas=' + str(len(t_both)) + '/40, DRoC_v_wins=' + str(len(t_vwins)) + ', DRoC_d_wins=' + str(len(t_dwins)) + ', AvgV_imp=' + str(round(v_imp,2)) + '%, AvgD_imp=' + str(round(d_imp,2)) + '%')

print()
print('=' * 80)
print('BREAKDOWN BY PROBLEM TYPE AND SCALE')
print('=' * 80)
for scale in sorted(scale_data.keys()):
    label = scale_labels[scale]
    data = scale_data[scale]
    print()
    print('Scale ' + label + ':')
    for t in types:
        t_rows = [r for r in data if r['instance'].startswith(t + '_')]
        t_both = [r for r in t_rows if is_baseline_feasible(r) and is_droc_feasible(r)]
        t_vwins = [r for r in t_both if float(r['droc_vehicles']) < float(r['baseline_vehicles'])]
        t_dwins = [r for r in t_both if float(r['droc_distance']) < float(r['baseline_distance'])]
        if t_both:
            at_dv = sum(float(r['droc_vehicles']) for r in t_both) / len(t_both)
            at_ov = sum(float(r['baseline_vehicles']) for r in t_both) / len(t_both)
            at_dd = sum(float(r['droc_distance']) for r in t_both) / len(t_both)
            at_od = sum(float(r['baseline_distance']) for r in t_both) / len(t_both)
            v_imp = (at_ov - at_dv) / at_ov * 100
            d_imp = (at_od - at_dd) / at_od * 100
        else:
            at_dv = at_ov = at_dd = at_od = v_imp = d_imp = 0.0
        print('  ' + t + ': both=' + str(len(t_both)) + '/10, v_wins=' + str(len(t_vwins)) + ', d_wins=' + str(len(t_dwins)) + ', v_imp=' + str(round(v_imp,2)) + '%, d_imp=' + str(round(d_imp,2)) + '%')