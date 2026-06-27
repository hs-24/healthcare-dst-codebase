/**
 * warehouse_models.js
 *
 * Procedural builders for the two additional warehouse 3D models
 * requested alongside the existing live operational twin ("Model A",
 * built inline in live_dashboard.html):
 *
 *   Model B - GDP Singapore Ramp-Up BIM
 *     Full structural model from warehouse_bim_layout.json: 120x200m
 *     envelope, 4-storey ramp-up shell, 12m structural column grid,
 *     SAS / VNA / VAS / Cold-Chain zoning, GDP colour-coded SKU
 *     taxonomy, fire buffer + egress walkway compliance bands.
 *
 *   Model C - Compliance Bounding-Box Schematic
 *     Flat, top-down extruded floor plan of the same zone/column
 *     bounding boxes -- cheap to render, useful for a GDP audit
 *     walkthrough or a quick "does the zoning make sense" review.
 *
 * Both builders take the live per-SKU state from /api/state and map
 * each SKU's `category` to the GDP colour taxonomy, so pallet fill in
 * SAS/VNA/Cold-Chain still reflects the same agentic simulation data
 * driving Model A -- the structural layout changes, the data doesn't.
 *
 * three.js is Y-up; the BIM JSON is X (width) / Y (depth) / Z (height).
 * Mapping used throughout this file: bimX -> three.x, bimY -> three.z,
 * bimZ (height) -> three.y.
 */

import * as THREE from "three";

export const TAXONOMY_COLOR = {
  PHARMA: 0x2ecc71,
  MEDICAL_DEVICE: 0x6fb7ff,
  BULK_PPE: 0x9aa5b1,
  ULTRA_COLD_VACCINE: 0x5b2a86,
  STANDARD_VACCINE: 0xc0152f,
};

/** Maps a SKU's report category (config/warehouse_config.yaml) onto the
 *  GDP colour taxonomy + target zone required by the brief. */
export function classifySku(category, storageMode) {
  const cat = (category || "").toLowerCase();
  if (storageMode === "COLD_VAULT") {
    return { zone: "COLD_CHAIN_ENCLOSURE", taxonomy: "STANDARD_VACCINE" };
  }
  if (cat.includes("pharma")) {
    return { zone: "SAS_AUTOMATED_STORAGE", taxonomy: "PHARMA" };
  }
  if (cat.includes("diagnostic") || cat.includes("equipment")) {
    return { zone: "VNA_VERY_NARROW_AISLE", taxonomy: "MEDICAL_DEVICE" };
  }
  return { zone: "VNA_VERY_NARROW_AISLE", taxonomy: "BULK_PPE" };
}

function disposeObject(obj) {
  obj.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      (Array.isArray(child.material) ? child.material : [child.material]).forEach((m) => m.dispose());
    }
  });
}

export function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    disposeObject(child);
  }
}

function bimToThree(x, y, z = 0) {
  return new THREE.Vector3(x - 60, z, y - 100); // centre the 120x200 footprint on the origin
}

function labelSprite(text, scale = 1, color = "#e6edf3") {
  const canvas = document.createElement("canvas");
  canvas.width = 512; canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 46px Segoe UI";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(text, 256, 76);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
  sprite.scale.set(8 * scale, 2 * scale, 1);
  return sprite;
}

// ---------------------------------------------------------------------
// Model B - GDP Singapore Ramp-Up BIM
// ---------------------------------------------------------------------

function buildEnvelopeShell(group, bim) {
  const { x: bx, y: by } = bim.building_envelope.footprint_m;
  const groundFloorHeight = bim.building_envelope.floor_heights_m["1"];

  const wallMat = new THREE.MeshStandardMaterial({ color: 0x1b2740, transparent: true, opacity: 0.18, side: THREE.DoubleSide });
  const floorMat = new THREE.MeshStandardMaterial({ color: 0x10192c, roughness: 1 });

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(bx, by), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 0);
  group.add(floor);

  // perimeter shell walls (level 1, semi-transparent so interior zoning is visible)
  const wallGeo = new THREE.BoxGeometry(bx, groundFloorHeight, 0.3);
  const front = new THREE.Mesh(wallGeo, wallMat);
  front.position.set(0, groundFloorHeight / 2, -by / 2);
  group.add(front);
  const back = front.clone(); back.position.z = by / 2; group.add(back);
  const sideGeo = new THREE.BoxGeometry(0.3, groundFloorHeight, by);
  const left = new THREE.Mesh(sideGeo, wallMat);
  left.position.set(-bx / 2, groundFloorHeight / 2, 0);
  group.add(left);
  const right = left.clone(); right.position.x = bx / 2; group.add(right);

  // stacked upper floors (2-4) as simplified flat decks + perimeter rail,
  // matching the brief's storey count/height without re-detailing zoning
  // that the spec only defines for Level 1
  let cumHeight = groundFloorHeight;
  const floorHeights = bim.building_envelope.floor_heights_m;
  [2, 3, 4].forEach((lvl) => {
    const h = floorHeights[String(lvl)];
    const deck = new THREE.Mesh(
      new THREE.BoxGeometry(bx, 0.25, by),
      new THREE.MeshStandardMaterial({ color: 0x223252, transparent: true, opacity: 0.5 })
    );
    deck.position.set(0, cumHeight, 0);
    group.add(deck);
    const railMat = new THREE.LineBasicMaterial({ color: 0x3a4d72 });
    const pts = [
      [-bx / 2, -by / 2], [bx / 2, -by / 2], [bx / 2, by / 2], [-bx / 2, by / 2], [-bx / 2, -by / 2],
    ].map(([dx, dz]) => new THREE.Vector3(dx, cumHeight + 0.13, dz));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), railMat));
    const sign = labelSprite(`LEVEL ${lvl} (+${cumHeight.toFixed(0)}m, ${h}m clear)`, 0.7, "#7d8aa3");
    sign.position.set(-bx / 2 + 14, cumHeight + 2, -by / 2 + 4);
    group.add(sign);
    cumHeight += h;
  });

  // two external circular vehicle ramps (corner helices), schematic
  [[-bx / 2 + 6, -by / 2 + 6], [bx / 2 - 6, -by / 2 + 6]].forEach(([rx, rz]) => {
    const turns = 4, ramp = new THREE.Group();
    const segCount = 48;
    const pts = [];
    for (let i = 0; i <= segCount; i++) {
      const t = i / segCount;
      const ang = t * Math.PI * 2 * turns;
      pts.push(new THREE.Vector3(rx + Math.cos(ang) * 5, t * cumHeight, rz + Math.sin(ang) * 5));
    }
    ramp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x5fa8ff })));
    group.add(ramp);
  });

  const buildingLabel = labelSprite("SINGAPORE RAMP-UP FACILITY · 120m × 200m × 4 LEVELS", 1.1, "#9fb3d8");
  buildingLabel.position.set(0, cumHeight + 3, 0);
  group.add(buildingLabel);
}

function buildColumns(group, bim) {
  const radius = bim.structural_columns.radius_m;
  const positions = bim.structural_columns.positions;
  const groundHeight = bim.building_envelope.floor_heights_m["1"];
  const geo = new THREE.CylinderGeometry(radius, radius, groundHeight, 12);
  const mat = new THREE.MeshStandardMaterial({ color: 0x4a5872, roughness: 0.8 });
  const inst = new THREE.InstancedMesh(geo, mat, positions.length);
  const dummy = new THREE.Object3D();
  positions.forEach((col, i) => {
    const p = bimToThree(col.center[0], col.center[1]);
    dummy.position.set(p.x, groundHeight / 2, p.z);
    dummy.updateMatrix();
    inst.setMatrixAt(i, dummy.matrix);
  });
  inst.instanceMatrix.needsUpdate = true;
  group.add(inst);
}

function zoneFootprintMesh(zone, color, opacity = 0.16) {
  const [minX, minY] = zone.min_point;
  const [maxX, maxY] = zone.max_point;
  const w = maxX - minX, d = maxY - minY;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, d),
    new THREE.MeshStandardMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide })
  );
  mesh.rotation.x = -Math.PI / 2;
  const center = bimToThree(minX + w / 2, minY + d / 2);
  mesh.position.set(center.x, 0.03, center.z);
  return mesh;
}

/** Builds a representative (not exhaustive) rack-and-pallet field inside
 *  a rectangular zone, respecting the egress walkway inset, bay width,
 *  aisle width, and tier count from the spec. */
function buildRackField(group, zone, opts) {
  const { aisleWidth, tiers, rackHeight, baySpanAxis, taxonomyFillRatio, defaultColor } = opts;
  const [minX, minY] = zone.min_point;
  const [maxX, maxY] = zone.max_point;
  const inset = 1.5; // egress walkway / cleanroom buffer
  const usableMinX = minX + inset, usableMaxX = maxX - inset;
  const usableMinY = minY + inset, usableMaxY = maxY - inset;
  const bayWidth = 2.7; // m, double-pallet bay
  const rowDepth = 1.2; // m, single pallet depth
  const levelHeight = rackHeight / tiers;

  const rowPitch = rowDepth * 2 + aisleWidth; // facing pair of rack rows + aisle between them
  const usableDepth = usableMaxY - usableMinY;
  const usableWidth = usableMaxX - usableMinX;
  const rowPairs = Math.max(1, Math.floor(usableDepth / rowPitch));
  const baysAlongWidth = Math.max(1, Math.floor(usableWidth / bayWidth));

  const palletGeo = new THREE.BoxGeometry(1.0, levelHeight * 0.82, 1.0);
  const totalSlots = rowPairs * 2 * baysAlongWidth * tiers;
  const inst = new THREE.InstancedMesh(palletGeo, new THREE.MeshStandardMaterial({ color: defaultColor }), totalSlots);
  const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(totalSlots * 3), 3);
  const dummy = new THREE.Object3D();
  const baseColor = new THREE.Color(defaultColor);
  const dimColor = new THREE.Color(0x1a2235);

  let slot = 0;
  for (let rp = 0; rp < rowPairs; rp++) {
    const pairCenterY = usableMinY + rp * rowPitch + rowDepth;
    [-1, 1].forEach((side) => {
      const rowY = pairCenterY + side * (rowDepth / 2 + (side < 0 ? 0 : aisleWidth));
      for (let b = 0; b < baysAlongWidth; b++) {
        const x = usableMinX + b * bayWidth + bayWidth / 2;
        for (let t = 0; t < tiers; t++) {
          const lit = Math.random() < taxonomyFillRatio; // representative occupancy
          const p = bimToThree(x, rowY, t * levelHeight + levelHeight / 2);
          dummy.position.set(p.x, p.y, p.z);
          dummy.scale.setScalar(lit ? 1 : 0.001);
          dummy.updateMatrix();
          inst.setMatrixAt(slot, dummy.matrix);
          const c = lit ? baseColor : dimColor;
          colorAttr.setXYZ(slot, c.r, c.g, c.b);
          slot++;
        }
      }
    });
  }
  inst.instanceColor = colorAttr;
  inst.instanceMatrix.needsUpdate = true;
  group.add(inst);

  // rack frame outline (cheap wireframe box per row-pair, not per-bay)
  const frameMat = new THREE.LineBasicMaterial({ color: defaultColor, transparent: true, opacity: 0.5 });
  for (let rp = 0; rp < rowPairs; rp++) {
    const pairCenterY = usableMinY + rp * rowPitch + rowDepth;
    const p0 = bimToThree(usableMinX, pairCenterY - rowDepth, 0);
    const p1 = bimToThree(usableMaxX, pairCenterY - rowDepth, rackHeight);
    const box = new THREE.Box3(new THREE.Vector3(Math.min(p0.x, p1.x), 0, Math.min(p0.z, p1.z)), new THREE.Vector3(Math.max(p0.x, p1.x), rackHeight, Math.max(p0.z, p1.z)));
    group.add(new THREE.Box3Helper(box, frameMat.color));
  }
}

function buildVnaWireGuidance(group, zone) {
  const [minX, minY] = zone.min_point;
  const [maxX, maxY] = zone.max_point;
  const mat = new THREE.LineBasicMaterial({ color: 0xffd166 });
  const inset = 1.5;
  const rowPitch = 1.2 * 2 + 1.65;
  const n = Math.max(1, Math.floor((maxY - minY - inset * 2) / rowPitch));
  for (let i = 0; i < n; i++) {
    const y = minY + inset + i * rowPitch + 1.2;
    const p0 = bimToThree(minX + inset, y, 0.02);
    const p1 = bimToThree(maxX - inset, y, 0.02);
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([p0, p1]), mat));
  }
}

function buildVasZone(group, zone, vasSpec) {
  group.add(zoneFootprintMesh(zone, 0xffd166, 0.08));
  const { width, depth, height } = vasSpec.workbench_dims_m;
  const count = vasSpec.workbench_count;
  const cols = 5, rows = Math.ceil(count / cols);
  const inset = 1.5;
  const [minX, minY] = zone.min_point;
  const usableW = (zone.max_point[0] - minX) - inset * 2;
  const usableD = (zone.max_point[1] - minY) - inset * 2;
  const colSpacing = usableW / cols, rowSpacing = usableD / rows;
  const geo = new THREE.BoxGeometry(width, height, depth);
  const mat = new THREE.MeshStandardMaterial({ color: 0xd9c389 });
  let placed = 0;
  for (let r = 0; r < rows && placed < count; r++) {
    for (let c = 0; c < cols && placed < count; c++) {
      const x = minX + inset + colSpacing * (c + 0.5);
      const y = minY + inset + rowSpacing * (r + 0.5);
      const p = bimToThree(x, y, height / 2);
      const bench = new THREE.Mesh(geo, mat);
      bench.position.set(p.x, p.y, p.z);
      group.add(bench);
      placed++;
    }
  }
  const sign = labelSprite("VAS · KITTING / HSA LABELING / CROSS-DOCK", 0.8, "#ffd166");
  const c = bimToThree((minX + zone.max_point[0]) / 2, (minY + zone.max_point[1]) / 2, 6);
  sign.position.set(c.x, c.y, c.z);
  group.add(sign);
}

function buildColdChainEnclosure(group, zone, skusByZone) {
  const [minX, minY] = zone.min_point;
  const [maxX, maxY] = zone.max_point;
  const h = zone.max_point[2];
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xd7dde3, metalness: 0.6, roughness: 0.3 });
  const w = maxX - minX, d = maxY - minY;
  const center = bimToThree(minX + w / 2, minY + d / 2, h / 2);

  const shell = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), wallMat.clone());
  shell.material.transparent = true;
  shell.material.opacity = 0.35;
  shell.position.set(center.x, center.y, center.z);
  group.add(shell);

  // two air-locked rapid-roll doors on the wall facing the main floor (-X side)
  const doorMat = new THREE.MeshStandardMaterial({ color: 0x8fa3bd, metalness: 0.7, roughness: 0.2 });
  [d * 0.3, d * 0.7].forEach((dz) => {
    const door = new THREE.Mesh(new THREE.BoxGeometry(0.15, 3.2, 2.4), doorMat);
    const p = bimToThree(minX, minY + dz, 1.6);
    door.position.set(p.x, p.y, p.z);
    group.add(door);
  });

  // overhead cooling units
  for (let i = 0; i < 4; i++) {
    const cu = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.6, 1.5), new THREE.MeshStandardMaterial({ color: 0xb8c4d0 }));
    const p = bimToThree(minX + w * (0.25 + 0.5 * (i % 2)), minY + d * (0.25 + 0.5 * Math.floor(i / 2)), h - 0.6);
    cu.position.set(p.x, p.y, p.z);
    group.add(cu);
  }

  // ultra-cold freezer cabinets (deep violet, illustrative SKU -- the
  // pilot dataset has no -70/-80C SKU; structural slot is still shown)
  const freezerMat = new THREE.MeshStandardMaterial({ color: 0x4a4a55, metalness: 0.5, roughness: 0.4 });
  const violetMat = new THREE.MeshStandardMaterial({ color: TAXONOMY_COLOR.ULTRA_COLD_VACCINE });
  for (let i = 0; i < 3; i++) {
    const cab = new THREE.Mesh(new THREE.BoxGeometry(1.4, 2.0, 1.0), freezerMat);
    const p = bimToThree(minX + 3 + i * 2.0, minY + 4, 1.0);
    cab.position.set(p.x, p.y, p.z);
    group.add(cab);
    const vial = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.3, 0.6), violetMat);
    vial.position.set(p.x, p.y + 1.15, p.z);
    group.add(vial);
  }

  // chilled racking (ruby red) -- populated from live cold-chain SKUs
  const chilledSkus = skusByZone["COLD_CHAIN_ENCLOSURE"] || [];
  const redMat = new THREE.MeshStandardMaterial({ color: TAXONOMY_COLOR.STANDARD_VACCINE });
  chilledSkus.forEach((s, i) => {
    const fillRatio = Math.min(1, (s.pallet_count || 0) / Math.max(1, s.pallet_capacity || 1));
    const slots = 4;
    for (let lvl = 0; lvl < slots; lvl++) {
      const lit = lvl < Math.round(fillRatio * slots);
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.5, 0.9), lit ? redMat : new THREE.MeshStandardMaterial({ color: 0x2a2030 }));
      const p = bimToThree(minX + w - 4 - i * 1.6, minY + d - 4, 0.4 + lvl * 0.6);
      box.position.set(p.x, p.y, p.z);
      group.add(box);
    }
  });

  const sign = labelSprite("COLD CHAIN · VNV · -70°C TO 8°C", 0.9, "#7fd1ff");
  sign.position.set(center.x, h + 1.5, center.z);
  group.add(sign);
}

/**
 * Builds the full GDP Singapore Ramp-Up BIM model into `group`.
 * `bim` is the parsed warehouse_bim_layout.json. `skus` is the live
 * /api/state skus map (sku_id -> state) used to colour/fill pallets.
 */
export function buildModelB(group, bim, skus) {
  clearGroup(group);
  buildEnvelopeShell(group, bim);
  buildColumns(group, bim);

  const zones = bim.level_1_zones;
  const skusByZone = { SAS_AUTOMATED_STORAGE: [], VNA_VERY_NARROW_AISLE: [], COLD_CHAIN_ENCLOSURE: [] };
  Object.entries(skus || {}).forEach(([id, s]) => {
    const { zone } = classifySku(s.category, s.storage_mode);
    if (skusByZone[zone]) skusByZone[zone].push({ ...s, id });
  });

  const aggregateFill = (zoneSkus) => {
    if (!zoneSkus.length) return 0.4;
    const ratios = zoneSkus.map((s) => Math.min(1, (s.pallet_count || 0) / Math.max(1, s.pallet_capacity || 1)));
    return ratios.reduce((a, b) => a + b, 0) / ratios.length;
  };

  const sas = zones.SAS_AUTOMATED_STORAGE;
  group.add(zoneFootprintMesh(sas, TAXONOMY_COLOR.PHARMA, 0.1));
  buildRackField(group, sas, {
    aisleWidth: sas.crane_aisle_width_m, tiers: sas.rack_tiers, rackHeight: sas.rack_height_m,
    taxonomyFillRatio: aggregateFill(skusByZone.SAS_AUTOMATED_STORAGE), defaultColor: TAXONOMY_COLOR.PHARMA,
  });
  group.add(labelSprite("SAS · AUTOMATED PHARMA STORAGE", 0.9, "#2ecc71"));
  group.children[group.children.length - 1].position.copy(bimToThree(30, 150, sas.rack_height_m + 2));

  const vna = zones.VNA_VERY_NARROW_AISLE;
  group.add(zoneFootprintMesh(vna, TAXONOMY_COLOR.MEDICAL_DEVICE, 0.1));
  buildRackField(group, vna, {
    aisleWidth: vna.aisle_width_m, tiers: vna.rack_tiers, rackHeight: vna.rack_height_m,
    taxonomyFillRatio: aggregateFill(skusByZone.VNA_VERY_NARROW_AISLE), defaultColor: TAXONOMY_COLOR.MEDICAL_DEVICE,
  });
  buildVnaWireGuidance(group, vna);
  const vnaSign = labelSprite("VNA · DEVICES + BULK PPE", 0.9, "#6fb7ff");
  vnaSign.position.copy(bimToThree(90, 150, vna.rack_height_m + 2));
  group.add(vnaSign);

  buildVasZone(group, zones.VAS_VALUE_ADDED_SERVICES, zones.VAS_VALUE_ADDED_SERVICES);
  buildColdChainEnclosure(group, zones.COLD_CHAIN_ENCLOSURE, skusByZone);
}

// ---------------------------------------------------------------------
// Model C - Compliance Bounding-Box Schematic (flat, top-down)
// ---------------------------------------------------------------------

export function buildModelC(group, bim, skus) {
  clearGroup(group);
  const { x: bx, y: by } = bim.building_envelope.footprint_m;

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(bx, by),
    new THREE.MeshStandardMaterial({ color: 0x0c1322, roughness: 1 })
  );
  floor.rotation.x = -Math.PI / 2;
  group.add(floor);

  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([
      bimToThree(0, 0), bimToThree(bx, 0), bimToThree(bx, by), bimToThree(0, by),
    ]),
    new THREE.LineBasicMaterial({ color: 0x7d8aa3 })
  );
  group.add(outline);

  const colorByZoneKey = {
    SAS_AUTOMATED_STORAGE: TAXONOMY_COLOR.PHARMA,
    VNA_VERY_NARROW_AISLE: TAXONOMY_COLOR.MEDICAL_DEVICE,
    VAS_VALUE_ADDED_SERVICES: 0xffd166,
    COLD_CHAIN_ENCLOSURE: TAXONOMY_COLOR.STANDARD_VACCINE,
  };

  Object.entries(bim.level_1_zones).forEach(([key, zone]) => {
    const mesh = zoneFootprintMesh(zone, colorByZoneKey[key] || 0x888888, 0.45);
    mesh.position.y = 0.04;
    group.add(mesh);
    const [minX, minY] = zone.min_point, [maxX, maxY] = zone.max_point;
    const outlinePts = [
      bimToThree(minX, minY), bimToThree(maxX, minY), bimToThree(maxX, maxY), bimToThree(minX, maxY),
    ];
    group.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(outlinePts), new THREE.LineBasicMaterial({ color: colorByZoneKey[key] || 0xffffff })));
    const c = bimToThree((minX + maxX) / 2, (minY + maxY) / 2, 0.5);
    const label = labelSprite(key.replace(/_/g, " "), 0.55);
    label.position.set(c.x, 1.2, c.z);
    group.add(label);
  });

  // structural columns as flat dots
  const radius = bim.structural_columns.radius_m;
  const positions = bim.structural_columns.positions;
  const colGeo = new THREE.CylinderGeometry(radius, radius, 0.4, 8);
  const colMat = new THREE.MeshStandardMaterial({ color: 0x4a5872 });
  const inst = new THREE.InstancedMesh(colGeo, colMat, positions.length);
  const dummy = new THREE.Object3D();
  positions.forEach((col, i) => {
    const p = bimToThree(col.center[0], col.center[1]);
    dummy.position.set(p.x, 0.2, p.z);
    dummy.updateMatrix();
    inst.setMatrixAt(i, dummy.matrix);
  });
  inst.instanceMatrix.needsUpdate = true;
  group.add(inst);

  // egress walkway band (1.5m inset from the building perimeter)
  const inset = bim.compliance.egress_walkway_width_m;
  const egressPts = [
    bimToThree(inset, inset), bimToThree(bx - inset, inset), bimToThree(bx - inset, by - inset), bimToThree(inset, by - inset),
    bimToThree(inset, inset),
  ];
  const egressLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(egressPts),
    new THREE.LineDashedMaterial({ color: 0xffd166, dashSize: 1.5, gapSize: 0.8 })
  );
  egressLine.computeLineDistances();
  group.add(egressLine);

  const title = labelSprite("GDP COMPLIANCE BOUNDING-BOX SCHEMATIC · LEVEL 1", 1.0, "#e6edf3");
  title.position.set(0, 4, -by / 2 - 8);
  group.add(title);
}
