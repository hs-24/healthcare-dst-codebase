"""
build_dashboard.py

Injects the consolidated dashboard_data.json and simulation config
values into the dashboard HTML template, producing a single
self-contained file for the deliverable.
"""

import os
import json

ROOT = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
DATA_PATH = os.path.join(ROOT, "outputs", "dashboard_data.json")
OUT_PATH = os.path.join(ROOT, "outputs", "healthcare_dst_dashboard.html")


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()

    cfg = data["results"]["simulation_config"]

    html = html.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__SITE_NAME__", cfg["site"])
    html = html.replace("__DURATION_DAYS__", str(cfg["duration_days"]))
    html = html.replace("__NUM_SKUS__", str(cfg["num_skus"]))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard written to {OUT_PATH}")
    print(f"File size: {os.path.getsize(OUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
