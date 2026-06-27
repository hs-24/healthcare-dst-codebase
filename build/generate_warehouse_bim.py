"""
generate_warehouse_bim.py

Produces the structured JSON coordinate map (bounding boxes) for the
GDP-compliant Singapore ramp-up Healthcare Logistics Digital Twin
building envelope -- walls, structural columns, and the four Level-1
functional zones (SAS, VNA, VAS, Cold Chain), plus the SKU colour
taxonomy and compliance buffers.

Coordinate convention in this file (matches the brief): X = building
width (m), Y = building depth (m), Z = height above floor (m). The
three.js consumer (web/warehouse_models.js) maps Z -> three.js Y (up)
and Y -> three.js Z, since three.js is Y-up.

Run:
    python build/generate_warehouse_bim.py

Writes:
    outputs/warehouse_bim_layout.json   (BIM deliverable of record)
    web/warehouse_bim_layout.json       (served to the live 3D twin)
"""

import json
import os

OUTPUT_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "outputs", "warehouse_bim_layout.json"),
    os.path.join(os.path.dirname(__file__), "..", "web", "warehouse_bim_layout.json"),
]

ENVELOPE = {"x": 120.0, "y": 200.0}
FLOOR_HEIGHTS = {1: 15.0, 2: 12.0, 3: 12.0, 4: 12.0}
COLUMN_GRID = {"spacing_x": 12.0, "spacing_y": 12.0, "radius": 0.4}
FIRE_BUFFER_M = 0.5      # no rack/pallet within this distance of the ceiling
EGRESS_WALKWAY_M = 1.5   # clear perimeter walkway / cleanroom buffer

PALLET = {"length_mm": 1200, "width_mm": 1000, "height_mm": 1400}
BAY_WIDTH_MM = 2700

SKU_COLOR_TAXONOMY = {
    "PHARMA": {"hex": "#2ecc71", "name": "Emerald Green", "use": "Pharmaceuticals & Prescription Medications (batch-controlled, high security)"},
    "MEDICAL_DEVICE": {"hex": "#6fb7ff", "name": "Light Blue", "use": "Medical Devices & Electronics (fragile, humidity-sensitive)"},
    "BULK_PPE": {"hex": "#9aa5b1", "name": "Slate Grey", "use": "Bulk PPE & Consumables (high volume, low stack-sensitivity)"},
    "ULTRA_COLD_VACCINE": {"hex": "#5b2a86", "name": "Deep Violet", "use": "mRNA Vaccines, -70C to -80C"},
    "STANDARD_VACCINE": {"hex": "#c0152f", "name": "Ruby Red", "use": "Standard Vaccines / Biologics, 2C to 8C"},
}


def generate_columns():
    """11 x 17 structural grid -- 200m / 12m doesn't divide evenly, so the
    last row sits at y=192m (8m short of the rear wall); flagged below."""
    xs = [round(i * COLUMN_GRID["spacing_x"], 1) for i in range(int(ENVELOPE["x"] // COLUMN_GRID["spacing_x"]) + 1)]
    ys = [round(i * COLUMN_GRID["spacing_y"], 1) for i in range(int(ENVELOPE["y"] // COLUMN_GRID["spacing_y"]) + 1)]
    columns = []
    for x in xs:
        for y in ys:
            columns.append({
                "center": [x, y],
                "radius": COLUMN_GRID["radius"],
                "spans_floors": [1, 2, 3, 4],
            })
    return columns, {"x_lines": xs, "y_lines": ys, "note": "y=200 rear wall is 8m beyond the last column line (192m) -- 200/12 does not divide evenly; standard tolerance for a ramp-up shell."}


def zone_box(min_xy, max_xy, height):
    return {
        "min_point": [min_xy[0], min_xy[1], 0.0],
        "max_point": [max_xy[0], max_xy[1], height],
    }


def generate_zones():
    return {
        "SAS_AUTOMATED_STORAGE": {
            **zone_box([0, 100], [60, 200], 14.0),
            "description": "Automated Storage & Retrieval (SAS) -- high-velocity pharmaceuticals",
            "rack_tiers": 10,
            "rack_height_m": 14.0,
            "crane_aisle_width_m": 1.1,
            "fire_buffer_clearance_m": FLOOR_HEIGHTS[1] - 14.0,
            "sku_taxonomy": ["PHARMA"],
        },
        "VNA_VERY_NARROW_AISLE": {
            **zone_box([60, 100], [120, 200], 12.5),
            "description": "Very Narrow Aisle (VNA) racking -- bulk medical products & devices",
            "rack_tiers": 8,
            "rack_height_m": 12.5,
            "aisle_width_m": 1.65,
            "fire_buffer_clearance_m": FLOOR_HEIGHTS[1] - 12.5,
            "sku_taxonomy": ["MEDICAL_DEVICE", "BULK_PPE"],
        },
        "VAS_VALUE_ADDED_SERVICES": {
            **zone_box([0, 0], [60, 100], 3.0),
            "description": "Value-Added Services -- healthcare kitting, HSA labeling, cross-dock sorting",
            "workbench_count": 20,
            "workbench_dims_m": {"width": 2.0, "depth": 1.2, "height": 0.9},
            "sku_taxonomy": [],
        },
        "COLD_CHAIN_ENCLOSURE": {
            **zone_box([80, 40], [120, 100], 6.0),
            "description": "Insulated cold-chain room -- vaccines & biologics deep-freeze",
            "wall_construction": "insulated sandwich panel (off-white / silver metallic)",
            "airlock_doors": 2,
            "subzones": {
                "ULTRA_COLD_FREEZERS": {"temp_c": "-70 to -80", "sku_taxonomy": ["ULTRA_COLD_VACCINE"]},
                "CHILLED_RACKING": {"temp_c": "2 to 8", "sku_taxonomy": ["STANDARD_VACCINE"]},
            },
        },
    }


def main():
    columns, column_grid_meta = generate_columns()
    layout = {
        "building_envelope": {
            "footprint_m": ENVELOPE,
            "floor_count": 4,
            "floor_heights_m": FLOOR_HEIGHTS,
            "vertical_circulation": "external circular vehicle ramps",
        },
        "structural_columns": {
            "grid": column_grid_meta,
            "radius_m": COLUMN_GRID["radius"],
            "count": len(columns),
            "positions": columns,
        },
        "pallet_spec": PALLET,
        "rack_bay_width_mm": BAY_WIDTH_MM,
        "sku_color_taxonomy": SKU_COLOR_TAXONOMY,
        "level_1_zones": generate_zones(),
        "compliance": {
            "fire_buffer_top_clearance_m": FIRE_BUFFER_M,
            "egress_walkway_width_m": EGRESS_WALKWAY_M,
            "egress_walkway_note": "Continuous clear perimeter walkway around the interior building shell and around the cold-chain enclosure walls.",
        },
    }

    for path in OUTPUT_PATHS:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
