"""
Helper to generate results.csv in all reproducibility evaluation directories.
"""

import os
import json
import csv

summary_path = "yolov8/results/stage6b_audit_summary.json"
with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dirs_map = {
    "yolov8/results/repro_eval_modelA": data["production_eval_conf025"]["model_a"]["normal_test"]["overall"],
    "yolov8/results/repro_eval_modelB_normal": data["production_eval_conf025"]["model_b"]["normal_test"]["overall"],
    "yolov8/results/repro_eval_modelA_camo": data["production_eval_conf025"]["model_a"]["camo_test"]["overall"],
    "yolov8/results/repro_eval_modelB_camo": data["production_eval_conf025"]["model_b"]["camo_test"]["overall"],
}

for dpath, ov in dirs_map.items():
    os.makedirs(dpath, exist_ok=True)
    csv_file = os.path.join(dpath, "results.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
        writer.writerow(["test", ov["precision"], ov["recall"], ov["map50"], ov["map50_95"]])
    print(f"Generated: {csv_file}")
