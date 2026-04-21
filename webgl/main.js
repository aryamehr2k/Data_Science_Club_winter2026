import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const canvasStruct = document.getElementById('canvasStruct');
const canvasEmbed = document.getElementById('canvasEmbed');
const canvasResidue = document.getElementById('canvasResidue');

const viewMode = document.getElementById('viewMode');
const pointSize = document.getElementById('pointSize');
const edgeOpacity = document.getElementById('edgeOpacity');
const showEdges = document.getElementById('showEdges');
const showBackbone = document.getElementById('showBackbone');
const showFocus = document.getElementById('showFocus');
const backboneWidth = document.getElementById('backboneWidth');
const backboneQuality = document.getElementById('backboneQuality');
const showSpheres = document.getElementById('showSpheres');
const resetBtn = document.getElementById('reset');
const meta = document.getElementById('meta');

const residueSelect = document.getElementById('residueSelect');
const wtAA = document.getElementById('wtAA');
const mutAA = document.getElementById('mutAA');
const scoreValue = document.getElementById('scoreValue');
const riskLabel = document.getElementById('riskLabel');
const riskMeta = document.getElementById('riskMeta');
const riskFill = document.getElementById('riskFill');
const pymolBtn = document.getElementById('pymolBtn');
const addMutBtn = document.getElementById('addMutBtn');
const cartList = document.getElementById('cartList');
const combinedDamage = document.getElementById('combinedDamage');
const combinedLabel = document.getElementById('combinedLabel');
const combinedFill = document.getElementById('combinedFill');
const clearCartBtn = document.getElementById('clearCartBtn');
const cartNote = document.getElementById('cartNote');

const mutationCart = new Map();
let cartMulti = null;
let apiAvailable = false;

const rendererStruct = new THREE.WebGLRenderer({ canvas: canvasStruct, antialias: true });
const rendererEmbed = new THREE.WebGLRenderer({ canvas: canvasEmbed, antialias: true });
const rendererResidue = new THREE.WebGLRenderer({ canvas: canvasResidue, antialias: true });

rendererStruct.setPixelRatio(window.devicePixelRatio);
rendererEmbed.setPixelRatio(window.devicePixelRatio);
rendererResidue.setPixelRatio(window.devicePixelRatio);
rendererStruct.setClearColor(0x0b0f1a, 1);
rendererEmbed.setClearColor(0x0b0f1a, 1);
rendererResidue.setClearColor(0x0b0f1a, 1);
rendererStruct.outputColorSpace = THREE.SRGBColorSpace;
rendererEmbed.outputColorSpace = THREE.SRGBColorSpace;
rendererResidue.outputColorSpace = THREE.SRGBColorSpace;
rendererStruct.toneMapping = THREE.ACESFilmicToneMapping;
rendererEmbed.toneMapping = THREE.ACESFilmicToneMapping;
rendererResidue.toneMapping = THREE.ACESFilmicToneMapping;
rendererStruct.toneMappingExposure = 1.1;
rendererEmbed.toneMappingExposure = 1.0;
rendererResidue.toneMappingExposure = 1.0;

const sceneStruct = new THREE.Scene();
const sceneEmbed = new THREE.Scene();
const sceneResidue = new THREE.Scene();
sceneStruct.fog = new THREE.Fog(0x0b0f1a, 40, 160);
sceneEmbed.fog = new THREE.Fog(0x0b0f1a, 40, 160);

const cameraStruct = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
const cameraEmbed = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
const cameraResidue = new THREE.PerspectiveCamera(45, 1, 0.1, 200);

cameraStruct.position.set(0, 0, 80);
cameraEmbed.position.set(0, 0, 80);
cameraResidue.position.set(0, 0, 30);

const controlsStruct = new OrbitControls(cameraStruct, rendererStruct.domElement);
const controlsEmbed = new OrbitControls(cameraEmbed, rendererEmbed.domElement);
const controlsResidue = new OrbitControls(cameraResidue, rendererResidue.domElement);
controlsStruct.enableDamping = true;
controlsEmbed.enableDamping = true;
controlsResidue.enableDamping = true;
controlsResidue.enablePan = false;
controlsStruct.enablePan = false;
controlsEmbed.enablePan = false;
controlsStruct.rotateSpeed = 0.6;
controlsStruct.zoomSpeed = 0.8;
controlsEmbed.rotateSpeed = 0.7;
controlsEmbed.zoomSpeed = 0.9;

const glowTexture = makeGlowTexture();

const ambientStruct = new THREE.AmbientLight(0xffffff, 0.45);
const dirStruct = new THREE.DirectionalLight(0xffffff, 0.65);
dirStruct.position.set(20, 30, 40);
const hemiStruct = new THREE.HemisphereLight(0x8bb8ff, 0x101820, 0.35);
sceneStruct.add(ambientStruct);
sceneStruct.add(dirStruct);
sceneStruct.add(hemiStruct);

const ambientRes = new THREE.AmbientLight(0xffffff, 0.7);
const dirRes = new THREE.DirectionalLight(0xffffff, 0.6);
dirRes.position.set(10, 20, 30);
sceneResidue.add(ambientRes);
sceneResidue.add(dirRes);

let pointsStruct, pointsEmbed, linesStruct, linesEmbed;
let residueGroup = null;
let backboneStruct = null;
let focusGroup = null;
let caSpheres = null;
let adjacency = null;
let structPosCache = null;
let embedPosCache = null;

const AA20 = "ACDEFGHIKLMNPQRSTVWY".split("");
const AA3 = {
  A: "Ala", C: "Cys", D: "Asp", E: "Glu", F: "Phe",
  G: "Gly", H: "His", I: "Ile", K: "Lys", L: "Leu",
  M: "Met", N: "Asn", P: "Pro", Q: "Gln", R: "Arg",
  S: "Ser", T: "Thr", V: "Val", W: "Trp", Y: "Tyr",
};

const ELEMENT_COLORS = {
  C: 0x55c86a,
  N: 0x4c7cff,
  O: 0xff5c5c,
  S: 0xffc14d,
  P: 0xff8ad6,
  H: 0xe6e6e6,
};

const COVALENT_RADII = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  S: 1.05,
  P: 1.07,
};

let dataCache = null;

function makeGlowTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size/2, size/2, 2, size/2, size/2, size/2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.2, 'rgba(255,255,255,0.7)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.15)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function colorByHop(hops) {
  const min = Math.min(...hops);
  const max = Math.max(...hops);
  const colors = new Float32Array(hops.length * 3);
  for (let i = 0; i < hops.length; i++) {
    const t = (hops[i] - min) / (max - min + 1e-6);
    const r = 0.2 + 0.7 * t;
    const g = 0.4 + 0.4 * (1 - t);
    const b = 0.9 - 0.7 * t;
    colors[i * 3 + 0] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  return colors;
}

function buildPoints(positions, colors, mutIndex, isEmbed = false) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: parseFloat(pointSize.value) * (isEmbed ? 1.4 : 1.0),
    vertexColors: true,
    transparent: true,
    opacity: isEmbed ? 0.95 : 0.9,
    depthWrite: true,
    blending: isEmbed ? THREE.AdditiveBlending : THREE.NormalBlending,
    map: glowTexture,
    sizeAttenuation: true,
  });

  const pts = new THREE.Points(geom, mat);

  // mutation star
  const starGeom = new THREE.SphereGeometry(1.6, 16, 16);
  const starMat = new THREE.MeshBasicMaterial({ color: 0xff3b30 });
  const star = new THREE.Mesh(starGeom, starMat);
  star.position.fromArray(positions, mutIndex * 3);
  pts.add(star);
  pts.userData.highlight = star;

  return pts;
}

function buildLines(positions, edges) {
  const segments = new Float32Array(edges.length * 2 * 3);
  for (let i = 0; i < edges.length; i++) {
    const [u, v] = edges[i];
    segments.set(positions.slice(u * 3, u * 3 + 3), i * 6);
    segments.set(positions.slice(v * 3, v * 3 + 3), i * 6 + 3);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0xa0a6b8, transparent: true, opacity: parseFloat(edgeOpacity.value) });
  return new THREE.LineSegments(geom, mat);
}

function buildBackboneTube(positions, radius = 1.1, quality = "med") {
  const pts = [];
  for (let i = 0; i < positions.length; i += 3) {
    pts.push(new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]));
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  curve.curveType = 'centripetal';
  const qualityMap = {
    low: { tubular: Math.max(pts.length * 2, 140), radial: 12 },
    med: { tubular: Math.max(pts.length * 4, 260), radial: 20 },
    high: { tubular: Math.max(pts.length * 6, 400), radial: 28 },
  };
  const q = qualityMap[quality] || qualityMap.med;
  const geom = new THREE.TubeGeometry(curve, q.tubular, radius, q.radial, false);
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x3fa9f5,
    transparent: true,
    opacity: 0.55,
    roughness: 0.35,
    metalness: 0.1,
    clearcoat: 0.7,
    clearcoatRoughness: 0.25,
  });
  return new THREE.Mesh(geom, mat);
}

function buildCASpheres(positions, colors, radius = 0.9) {
  const n = positions.length / 3;
  const geom = new THREE.SphereGeometry(radius, 14, 14);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.05 });
  const mesh = new THREE.InstancedMesh(geom, mat, n);
  const m = new THREE.Matrix4();
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3 + 0];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    m.makeTranslation(x, y, z);
    mesh.setMatrixAt(i, m);
    const c = new THREE.Color(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);
    mesh.setColorAt(i, c);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) {
    mesh.instanceColor.needsUpdate = true;
  }
  return mesh;
}

function updateSphereColors(mesh, colors) {
  if (!mesh) return;
  const n = colors.length / 3;
  for (let i = 0; i < n; i++) {
    const c = new THREE.Color(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);
    mesh.setColorAt(i, c);
  }
  if (mesh.instanceColor) {
    mesh.instanceColor.needsUpdate = true;
  }
}

function toFloat32(positions) {
  const arr = new Float32Array(positions.length * 3);
  positions.forEach((p, i) => {
    arr[i * 3 + 0] = p[0];
    arr[i * 3 + 1] = p[1];
    arr[i * 3 + 2] = p[2];
  });
  return arr;
}

function centerPositions(arr) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < arr.length; i += 3) {
    v.set(arr[i], arr[i + 1], arr[i + 2]);
    box.expandByPoint(v);
  }
  const center = box.getCenter(new THREE.Vector3());
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i += 3) {
    out[i] = arr[i] - center.x;
    out[i + 1] = arr[i + 1] - center.y;
    out[i + 2] = arr[i + 2] - center.z;
  }
  return out;
}

function scalePositions(arr, targetSize) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < arr.length; i += 3) {
    v.set(arr[i], arr[i + 1], arr[i + 2]);
    box.expandByPoint(v);
  }
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  const scale = size > 0 ? targetSize / size : 1;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i += 3) {
    out[i] = (arr[i] - center.x) * scale;
    out[i + 1] = (arr[i + 1] - center.y) * scale;
    out[i + 2] = (arr[i + 2] - center.z) * scale;
  }
  return out;
}

function fitCamera(camera, positions, controls = null) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    v.set(positions[i], positions[i + 1], positions[i + 2]);
    box.expandByPoint(v);
  }
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, size * 0.9 + 30)));
  camera.lookAt(center);
  if (controls) {
    controls.target.copy(center);
    controls.minDistance = Math.max(10, size * 0.3);
    controls.maxDistance = size * 3.0 + 50;
    controls.update();
  }
}

function buildResidueModel(atoms) {
  const group = new THREE.Group();
  if (!atoms || atoms.length === 0) {
    return group;
  }

  const positions = atoms.map((a) => new THREE.Vector3(a.x, a.y, a.z));
  const center = new THREE.Vector3();
  positions.forEach((p) => center.add(p));
  center.divideScalar(positions.length);

  let maxDist = 0;
  positions.forEach((p) => {
    p.sub(center);
    maxDist = Math.max(maxDist, p.length());
  });
  const scale = maxDist > 0 ? 12 / maxDist : 1;
  positions.forEach((p) => p.multiplyScalar(scale));

  const atomMeshes = [];
  for (let i = 0; i < atoms.length; i++) {
    const elem = (atoms[i].e || "C").toUpperCase();
    const color = ELEMENT_COLORS[elem] || 0xffffff;
    const radius = (COVALENT_RADII[elem] || 0.7) * 0.95;
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 18, 18),
      new THREE.MeshStandardMaterial({ color, roughness: 0.25, metalness: 0.05 })
    );
    sphere.position.copy(positions[i]);
    group.add(sphere);
    atomMeshes.push({ pos: positions[i], elem });
  }

  const bondMat = new THREE.MeshStandardMaterial({ color: 0xb6bfda, roughness: 0.5, metalness: 0.0 });
  for (let i = 0; i < atomMeshes.length; i++) {
    for (let j = i + 1; j < atomMeshes.length; j++) {
      const a = atomMeshes[i];
      const b = atomMeshes[j];
      const r1 = COVALENT_RADII[a.elem] || 0.7;
      const r2 = COVALENT_RADII[b.elem] || 0.7;
      const maxBond = (r1 + r2) * 1.25;
      const dist = a.pos.distanceTo(b.pos);
      if (dist > 0.4 && dist < maxBond) {
        const cyl = new THREE.Mesh(
          new THREE.CylinderGeometry(0.12, 0.12, dist, 8),
          bondMat
        );
        const mid = new THREE.Vector3().addVectors(a.pos, b.pos).multiplyScalar(0.5);
        cyl.position.copy(mid);
        const dir = new THREE.Vector3().subVectors(b.pos, a.pos).normalize();
        cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        group.add(cyl);
      }
    }
  }

  return group;
}

function updateResidueView(idx) {
  if (!dataCache || !dataCache.residue_atoms) return;
  const atoms = dataCache.residue_atoms[String(idx)] || [];
  if (residueGroup) sceneResidue.remove(residueGroup);
  residueGroup = buildResidueModel(atoms);
  sceneResidue.add(residueGroup);
  cameraResidue.position.set(0, 0, 30);
  cameraResidue.lookAt(0, 0, 0);
}

function updateHighlights(idx) {
  const posStruct = pointsStruct?.geometry?.attributes?.position?.array;
  const posEmbed = pointsEmbed?.geometry?.attributes?.position?.array;
  if (pointsStruct?.userData?.highlight && posStruct) {
    pointsStruct.userData.highlight.position.fromArray(posStruct, idx * 3);
  }
  if (pointsEmbed?.userData?.highlight && posEmbed) {
    pointsEmbed.userData.highlight.position.fromArray(posEmbed, idx * 3);
  }
}

function buildAdjacency(numNodes, edges) {
  const adj = Array.from({ length: numNodes }, () => []);
  edges.forEach(([u, v]) => {
    adj[u].push(v);
  });
  return adj;
}

function computeHops(adj, start) {
  const n = adj.length;
  const dist = new Int32Array(n);
  dist.fill(-1);
  dist[start] = 0;
  const q = [start];
  let head = 0;
  while (head < q.length) {
    const u = q[head++];
    for (const v of adj[u]) {
      if (dist[v] === -1) {
        dist[v] = dist[u] + 1;
        q.push(v);
      }
    }
  }
  return dist;
}

function colorsFromHops(hops) {
  const min = Math.min(...hops);
  const max = Math.max(...hops);
  const colors = new Float32Array(hops.length * 3);
  for (let i = 0; i < hops.length; i++) {
    const t = (hops[i] - min) / (max - min + 1e-6);
    const r = 0.2 + 0.7 * t;
    const g = 0.4 + 0.4 * (1 - t);
    const b = 0.9 - 0.7 * t;
    colors[i * 3 + 0] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  return colors;
}

function applyColors(points, colors) {
  if (!points) return;
  const attr = points.geometry.attributes.color;
  attr.array.set(colors);
  attr.needsUpdate = true;
  if (points === pointsStruct && caSpheres) {
    updateSphereColors(caSpheres, colors);
  }
}

function updateFocusOverlay(idx, damage, extraIdx = []) {
  if (!showFocus.checked || !structPosCache) {
    if (focusGroup) focusGroup.visible = false;
    return;
  }
  if (!focusGroup) {
    focusGroup = new THREE.Group();
    sceneStruct.add(focusGroup);
  }
  focusGroup.visible = true;
  focusGroup.clear();

  const base = new THREE.Color(0xff3b30);
  const glow = new THREE.Color(0xffa23b);
  const cartColor = new THREE.Color(0xffa23b);
  const pos = structPosCache;

  const main = new THREE.Mesh(
    new THREE.SphereGeometry(1.8 + damage * 2.0, 18, 18),
    new THREE.MeshBasicMaterial({ color: base, transparent: true, opacity: 0.9 })
  );
  main.position.fromArray(pos, idx * 3);
  focusGroup.add(main);

  if (adjacency) {
    const hops = computeHops(adjacency, idx);
    for (let i = 0; i < hops.length; i++) {
      if (i === idx) continue;
      if (hops[i] > 2) continue;
      const w = damage * (1 / (hops[i] + 1));
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.8 + w * 1.2, 12, 12),
        new THREE.MeshBasicMaterial({ color: glow, transparent: true, opacity: 0.5 })
      );
      s.position.fromArray(pos, i * 3);
      focusGroup.add(s);
    }
  }

  for (const j of extraIdx) {
    if (j === idx) continue;
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(1.5, 16, 16),
      new THREE.MeshBasicMaterial({ color: cartColor, transparent: true, opacity: 0.85 })
    );
    m.position.fromArray(pos, j * 3);
    focusGroup.add(m);
  }
}

function updateMutationCard(idx, mut) {
  if (!dataCache) return;
  const stats = dataCache.score_stats || { mean: 0, std: 1 };
  const yStats = dataCache.y_stats || { mean: stats.mean, std: stats.std };
  const key = `${idx}:${mut}`;
  const entry = dataCache.mutMap.get(key);
  const hit = entry ? { score: entry.score, source: "measured" } : localScoreFor(idx, mut);

  if (!hit) {
    scoreValue.textContent = "No data";
    riskLabel.textContent = "Unknown impact";
    riskLabel.style.color = "#9fb0dd";
    riskFill.style.width = "0%";
    riskMeta.textContent = "No measured or trustworthy predicted score available.";
    return;
  }

  const score = hit.score;
  const std = stats.std || 1;
  const z = (score - stats.mean) / std;

  let label = "Low/Neutral risk";
  let color = "#7dd97d";
  if (z < -1.0) {
    label = "High risk";
    color = "#ff5c5c";
  } else if (z < -0.5) {
    label = "Elevated risk";
    color = "#ffb347";
  } else if (z > 0.7) {
    label = "Potential gain";
    color = "#5fb1ff";
  }

  const damage = Math.max(0, Math.min(1, 1 / (1 + Math.exp(z))));
  const pct = Math.round(damage * 100);
  const source = hit.source === "measured" ? "Measured" : "Predicted";
  scoreValue.textContent = `${source} Score ${score.toFixed(2)}`;
  riskLabel.textContent = label;
  riskLabel.style.color = color;
  riskFill.style.width = `${pct}%`;
  riskMeta.textContent = `${source} impact • ${z.toFixed(2)}σ vs dataset mean • est. damage ${pct}%`;

  if (adjacency) {
    const hops = Array.from(computeHops(adjacency, idx));
    const colors = colorsFromHops(hops);
    for (let i = 0; i < hops.length; i++) {
      if (hops[i] <= 2) {
        const t = damage * (1 / (hops[i] + 1));
        colors[i * 3 + 0] = colors[i * 3 + 0] * (1 - t) + 1.0 * t;
        colors[i * 3 + 1] = colors[i * 3 + 1] * (1 - t);
        colors[i * 3 + 2] = colors[i * 3 + 2] * (1 - t);
      }
    }
    applyColors(pointsStruct, colors);
    applyColors(pointsEmbed, colors);
  }

  const extra = Array.from(mutationCart.values()).map((c) => c.residue_idx);
  updateFocusOverlay(idx, damage, extra);
}

function cartKey(idx, mut) {
  return `${idx}:${mut}`;
}

function localScoreFor(idx, mut) {
  if (!dataCache) return null;
  const k = cartKey(idx, mut);
  const m = dataCache.mutMap?.get(k);
  if (m) return { score: m.score, source: "measured" };
  const p = dataCache.predMap?.get(k);
  if (p != null) {
    const stats = dataCache.score_stats || { min: 0, max: 1 };
    const span = Math.max(1, stats.max - stats.min);
    const lo = stats.min - span;
    const hi = stats.max + span;
    if (p >= lo && p <= hi) return { score: p, source: "predicted" };
  }
  return null;
}

function labelFromZ(z) {
  if (z < -1.0) return { label: "High risk", color: "#ff5c5c" };
  if (z < -0.5) return { label: "Elevated risk", color: "#ffb347" };
  if (z > 0.7) return { label: "Potential gain", color: "#5fb1ff" };
  return { label: "Low/Neutral risk", color: "#7dd97d" };
}

async function refreshCombinedFromApi() {
  if (!apiAvailable || mutationCart.size === 0) return false;
  const payload = {
    mutations: Array.from(mutationCart.values()).map((c) => ({
      residue_idx: c.residue_idx,
      mut_aa: c.mut,
    })),
  };
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) return false;
    const body = await r.json();
    cartMulti = body;
    applyCombined(body);
    return true;
  } catch {
    return false;
  }
}

function refreshCombinedLocal() {
  const stats = dataCache?.score_stats || { mean: 0, std: 1 };
  const std = stats.std || 1;
  let zSum = 0;
  let n = 0;
  const missing = [];
  for (const c of mutationCart.values()) {
    const hit = localScoreFor(c.residue_idx, c.mut);
    if (!hit) {
      missing.push(`${c.wt}${c.resseq}${c.mut}`);
      continue;
    }
    const z = (hit.score - stats.mean) / std;
    zSum += z;
    n += 1;
  }
  if (n === 0) {
    cartMulti = null;
    applyCombined(null, missing);
    return;
  }
  const zMean = zSum / n;
  const damage = 1.0 / (1.0 + Math.exp(zMean));
  const { label, color } = labelFromZ(zMean);
  cartMulti = {
    n_valid: n,
    aggregate_z_mean: zMean,
    aggregate_z_sum: zSum,
    combined_damage_pct: damage * 100,
    risk_label: label,
    risk_color: color,
    missing,
    note: "Combined damage is an additive approximation.",
  };
  applyCombined(cartMulti, missing);
}

function applyCombined(body, missing = []) {
  if (!body || body.n_valid === 0) {
    combinedDamage.textContent = "—";
    combinedLabel.textContent = "Add mutations to compute combined damage";
    combinedLabel.style.color = "";
    combinedFill.style.width = "0%";
    cartNote.textContent = missing.length ? `no data: ${missing.join(", ")}` : "";
    return;
  }
  const pct = Math.round(body.combined_damage_pct);
  combinedDamage.textContent = `${pct}%`;
  combinedLabel.textContent =
    `${body.risk_label} · ${body.n_valid} muts · ⟨z⟩=${body.aggregate_z_mean.toFixed(2)}`;
  combinedLabel.style.color = body.risk_color || "#ffcf7a";
  combinedFill.style.width = `${pct}%`;
  const notes = [];
  if (body.missing?.length) notes.push(`no data: ${body.missing.join(", ")}`);
  notes.push(body.note || "");
  cartNote.textContent = notes.filter(Boolean).join(" · ");
}

function renderCart() {
  cartList.innerHTML = "";
  if (mutationCart.size === 0) {
    const empty = document.createElement("div");
    empty.className = "cart-empty";
    empty.textContent = 'No mutations added. Use "Add to Set" to combine mutations.';
    cartList.appendChild(empty);
    applyCombined(null);
    if (dataCache) {
      const idx = parseInt(residueSelect.value, 10);
      updateMutationCard(idx, mutAA.value);
    }
    return;
  }
  for (const [key, c] of mutationCart.entries()) {
    const row = document.createElement("div");
    row.className = "cart-row";
    const label = document.createElement("span");
    label.className = "cart-label";
    label.textContent = `${c.wt}${c.resseq}${c.mut}`;
    const src = document.createElement("span");
    src.className = `cart-src cart-src-${c.source || "unknown"}`;
    src.textContent = c.source || "";
    const rm = document.createElement("button");
    rm.className = "cart-remove";
    rm.textContent = "×";
    rm.title = "remove";
    rm.addEventListener("click", () => {
      mutationCart.delete(key);
      renderCart();
      refreshCombined();
    });
    row.appendChild(label);
    row.appendChild(src);
    row.appendChild(rm);
    cartList.appendChild(row);
  }
  if (dataCache) {
    const idx = parseInt(residueSelect.value, 10);
    updateMutationCard(idx, mutAA.value);
  }
}

async function refreshCombined() {
  const apiOk = await refreshCombinedFromApi();
  if (!apiOk) refreshCombinedLocal();
}

function addCurrentMutation() {
  if (!dataCache) return;
  const idx = parseInt(residueSelect.value, 10);
  const r = dataCache.residues.find((rr) => rr.idx === idx);
  if (!r) return;
  const mut = mutAA.value;
  if (mut === r.aa) return;
  const key = cartKey(idx, mut);
  const hit = localScoreFor(idx, mut);
  mutationCart.set(key, {
    residue_idx: idx,
    resseq: r.resseq,
    wt: r.aa,
    mut,
    source: hit?.source,
  });
  renderCart();
  refreshCombined();
}

async function detectApi() {
  try {
    const r = await fetch("/api/health");
    apiAvailable = r.ok;
  } catch {
    apiAvailable = false;
  }
}

function buildPymolScript(resseq, wt, mut) {
  const pdbPath = (dataCache?.meta?.pdb_path || "../data/structures/TP53_RCSB.pdb");
  const chainId = (dataCache?.meta?.chain_id || "A");
  const label = `${wt}${resseq}${mut}`;
  return [
    "reinitialize",
    `load ${pdbPath}, protein`,
    "hide everything",
    "show cartoon, protein",
    `select mut_site, chain ${chainId} and resi ${resseq}`,
    "show sticks, mut_site",
    "color yellow, mut_site",
    "set stick_radius, 0.18",
    "bg_color black",
    `label mut_site and name CA, "${label}"`,
    "zoom mut_site, 10",
    "wizard mutagenesis",
    `refresh_wizard`,
  ].join("\n");
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function initInspector(data) {
  dataCache = data;
  dataCache.mutMap = new Map();
  data.mutations.forEach((m) => {
    dataCache.mutMap.set(`${m.idx}:${m.mut}`, m);
  });
  dataCache.predMap = new Map();
  if (data.predictions) {
    Object.entries(data.predictions).forEach(([k, v]) => {
      dataCache.predMap.set(k, v);
    });
  }

  residueSelect.innerHTML = "";
  data.residues.forEach((r) => {
    const opt = document.createElement("option");
    const aa3 = AA3[r.aa] || r.aa;
    opt.value = r.idx;
    opt.textContent = `${r.aa}${r.resseq} (${aa3})`;
    residueSelect.appendChild(opt);
  });

  mutAA.innerHTML = "";
  AA20.forEach((aa) => {
    const opt = document.createElement("option");
    opt.value = aa;
    opt.textContent = aa;
    mutAA.appendChild(opt);
  });

  const startIdx = data.mut ?? 0;
  residueSelect.value = String(startIdx);
  const res = data.residues.find((r) => r.idx === startIdx) || data.residues[0];
  wtAA.value = res.aa;
  mutAA.value = res.aa;

  updateHighlights(startIdx);
  updateResidueView(startIdx);
  updateMutationCard(startIdx, mutAA.value);

  residueSelect.addEventListener("change", () => {
    const idx = parseInt(residueSelect.value, 10);
    const r = data.residues.find((rr) => rr.idx === idx);
    if (r) {
      wtAA.value = r.aa;
      mutAA.value = r.aa;
    }
    updateHighlights(idx);
    updateResidueView(idx);
    updateMutationCard(idx, mutAA.value);
  });

  mutAA.addEventListener("change", () => {
    const idx = parseInt(residueSelect.value, 10);
    updateMutationCard(idx, mutAA.value);
  });

  pymolBtn.addEventListener("click", () => {
    const idx = parseInt(residueSelect.value, 10);
    const r = data.residues.find((rr) => rr.idx === idx);
    if (!r) return;
    const script = buildPymolScript(r.resseq, r.aa, mutAA.value);
    downloadText(`pymol_${r.aa}${r.resseq}${mutAA.value}.pml`, script);
  });

  addMutBtn.addEventListener("click", addCurrentMutation);
  clearCartBtn.addEventListener("click", () => {
    mutationCart.clear();
    renderCart();
    refreshCombined();
  });
  renderCart();
}

async function loadData() {
  const res = await fetch('data.json');
  const data = await res.json();

  meta.textContent = `model: ${data.meta.label} | nodes: ${data.meta.nodes} | edges: ${data.meta.edges} | checkpoint: ${data.meta.checkpoint}`;

  const structPos = centerPositions(toFloat32(data.struct));
  const embedPosRaw = centerPositions(toFloat32(data.embed));
  const embedPos = scalePositions(embedPosRaw, 60);
  const colors = colorByHop(data.hop);

  structPosCache = structPos;
  embedPosCache = embedPos;
  adjacency = buildAdjacency(data.meta.nodes, data.edges);

  pointsStruct = buildPoints(structPos, colors, data.mut, false);
  pointsEmbed = buildPoints(embedPos, colors, data.mut, true);

  linesStruct = buildLines(structPos, data.edges);
  linesEmbed = buildLines(embedPos, data.edges);
  backboneStruct = buildBackboneTube(structPos, parseFloat(backboneWidth.value), backboneQuality.value);
  caSpheres = buildCASpheres(structPos, colors, 0.9);

  sceneStruct.add(pointsStruct);
  sceneEmbed.add(pointsEmbed);
  sceneStruct.add(linesStruct);
  sceneEmbed.add(linesEmbed);
  sceneStruct.add(backboneStruct);
  sceneStruct.add(caSpheres);
  backboneStruct.visible = showBackbone.checked;
  caSpheres.visible = showSpheres.checked;

  fitCamera(cameraStruct, structPos, controlsStruct);
  fitCamera(cameraEmbed, embedPos, controlsEmbed);

  initInspector(data);
}

function getCanvasRect(canvas) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width > 10 && rect.height > 10) {
    return rect;
  }
  const parent = canvas.parentElement;
  if (parent) {
    const p = parent.getBoundingClientRect();
    return {
      width: Math.max(300, p.width - 12),
      height: Math.max(200, p.height - 32),
    };
  }
  return { width: 640, height: 480 };
}

function resize() {
  const rectStruct = getCanvasRect(canvasStruct);
  const rectEmbed = getCanvasRect(canvasEmbed);
  const rectResidue = getCanvasRect(canvasResidue);

  rendererStruct.setSize(rectStruct.width, rectStruct.height, false);
  rendererEmbed.setSize(rectEmbed.width, rectEmbed.height, false);
  rendererResidue.setSize(rectResidue.width, rectResidue.height, false);

  cameraStruct.aspect = rectStruct.width / rectStruct.height;
  cameraEmbed.aspect = rectEmbed.width / rectEmbed.height;
  cameraStruct.updateProjectionMatrix();
  cameraEmbed.updateProjectionMatrix();
  cameraResidue.aspect = rectResidue.width / rectResidue.height;
  cameraResidue.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controlsStruct.update();
  controlsEmbed.update();
  controlsResidue.update();

  rendererStruct.render(sceneStruct, cameraStruct);
  rendererEmbed.render(sceneEmbed, cameraEmbed);
  rendererResidue.render(sceneResidue, cameraResidue);
}

pointSize.addEventListener('input', () => {
  if (pointsStruct) pointsStruct.material.size = parseFloat(pointSize.value);
  if (pointsEmbed) pointsEmbed.material.size = parseFloat(pointSize.value) * 1.4;
});

edgeOpacity.addEventListener('input', () => {
  if (linesStruct) linesStruct.material.opacity = parseFloat(edgeOpacity.value);
  if (linesEmbed) linesEmbed.material.opacity = parseFloat(edgeOpacity.value);
});

showEdges.addEventListener('change', () => {
  if (linesStruct) linesStruct.visible = showEdges.checked;
  if (linesEmbed) linesEmbed.visible = showEdges.checked;
});

showBackbone.addEventListener('change', () => {
  if (backboneStruct) backboneStruct.visible = showBackbone.checked;
});

showFocus.addEventListener('change', () => {
  if (focusGroup) focusGroup.visible = showFocus.checked;
});

function rebuildBackbone() {
  if (!structPosCache) return;
  if (backboneStruct) {
    sceneStruct.remove(backboneStruct);
    backboneStruct.geometry.dispose();
    backboneStruct.material.dispose();
  }
  backboneStruct = buildBackboneTube(structPosCache, parseFloat(backboneWidth.value), backboneQuality.value);
  sceneStruct.add(backboneStruct);
  backboneStruct.visible = showBackbone.checked;
}

backboneWidth.addEventListener('input', rebuildBackbone);
backboneQuality.addEventListener('change', rebuildBackbone);

showSpheres.addEventListener('change', () => {
  if (caSpheres) caSpheres.visible = showSpheres.checked;
});

viewMode.addEventListener('change', () => {
  const v = viewMode.value;
  document.querySelectorAll('.panel')[0].style.display = (v === 'embed') ? 'none' : 'block';
  document.querySelectorAll('.panel')[1].style.display = (v === 'struct') ? 'none' : 'block';
  resize();
});

resetBtn.addEventListener('click', () => {
  if (pointsStruct) fitCamera(cameraStruct, pointsStruct.geometry.attributes.position.array, controlsStruct);
  if (pointsEmbed) fitCamera(cameraEmbed, pointsEmbed.geometry.attributes.position.array, controlsEmbed);
  cameraResidue.position.set(0, 0, 30);
  cameraResidue.lookAt(0, 0, 0);
});

window.addEventListener('resize', resize);

detectApi().finally(() => {
  loadData().then(() => {
    resize();
    animate();
  });
});
