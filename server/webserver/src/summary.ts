import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SLAMConnection } from './services/SLAMConnection';
import type { SLAMUpdate, ResolvedBeacon } from './types';
import './summary.css';

/**
 * SLAM Summary Page
 *
 * Connects to the SLAM server directly and requests the global map.
 *
 *   - Left: 2D overhead minimap + AI chatbox
 *   - Right: Full 3D point cloud viewer (with beacons + detection boxes)
 *   - Bottom: Detection debug pipeline (CLIP + SAM3)
 */

// ── Types ──

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface SamMaskDiag {
  score: number;
  box_2d: number[];
  mask_image: string;
  above_sam_threshold: boolean;
  sam_threshold_used: number;
  has_3d_box: boolean;
  bbox_3d: any | null;
  dedup_kept?: boolean;
}

interface FrameDiag {
  submap_id: number;
  frame_idx: number;
  query: string;
  clip_similarity: number;
  above_threshold: boolean;
  clip_threshold_used: number;
  sam_threshold_used: number;
  thumbnail: string | null;
  resolution?: string;
  sam_masks: SamMaskDiag[];
  sam_error?: string;
  detections_before_dedup: any[];
}

interface DebugDetectResponse {
  queries: string[];
  clip_thresholds: Record<string, number>;
  sam_thresholds: Record<string, number>;
  frames: FrameDiag[];
  raw_detection_count: number;
  deduped_detection_count: number;
  detections: any[];
  total_frames_scanned: number;
  query_time_ms: number;
  error?: string;
}

// ── Globals ──

let snapshot: SLAMUpdate | null = null;
let connection: SLAMConnection;
let lastPointCount = 0;

// Three.js state
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls;
let pointCloud: THREE.Points;
let pointCloudGeometry: THREE.BufferGeometry;
let pointCloudMaterial: THREE.PointsMaterial;
let cameraGroup: THREE.Group;
let beaconGroup: THREE.Group;
let detectionBoxGroup: THREE.Group;
let detectionLabelGroup: THREE.Group;
let gridHelper: THREE.GridHelper;
let sceneRoot: THREE.Group;
let showGrid = true;

// Chat
const chatHistory: ChatMessage[] = [];

// Detection debug state
let lastDetectResponse: DebugDetectResponse | null = null;
let currentDetFilter: 'all' | 'above' | 'sam' | 'detected' = 'all';
let lastDetections: any[] = []; // final detections for overlay

// Beacon colors (match SceneManager)
const BEACON_COLORS = [0xFFD700, 0xFF6B6B, 0x4ECDC4, 0xA78BFA, 0xFB923C, 0x34D399];
// Detection box colors
const BOX_COLORS = [
  0x4da6ff, 0xff6b6b, 0x51cf66, 0xfcc419, 0xcc5de8,
  0xff922b, 0x20c997, 0xf06595, 0x5c7cfa, 0x94d82d,
];

// ── Init ──

document.addEventListener('DOMContentLoaded', () => {
  init3DViewer();
  setupChat();
  setupDetection();
  setupViewerControls();
  showConnectingMessage();

  const serverUrl = import.meta.env.VITE_SERVER_URL || window.location.origin;
  connection = new SLAMConnection(serverUrl);

  connection.onUpdate((data: SLAMUpdate) => {
    if (!data.points || data.points.length === 0) {
      showNoDataMessage();
      return;
    }
    snapshot = data;
    updateStats();

    // Only rebuild 3D scene when point count changes (avoid constant re-renders)
    if (data.n_points !== lastPointCount) {
      lastPointCount = data.n_points;
      loadPointCloud();
      loadCameras();
    }

    // Always update beacons (they can change without point count changes)
    updateBeacons(data.resolved_beacons ?? []);

    drawMinimap();
    hideStatusMessage();

    // Enable detect button once we have map data
    const detectBtn = document.getElementById('detectBtn') as HTMLButtonElement | null;
    if (detectBtn) detectBtn.disabled = false;
  });

  connection.onConnect(() => {
    console.log('\u2705 Summary page connected to SLAM server');
  });

  connection.onDisconnect(() => {
    console.log('\u274c Summary page disconnected from SLAM server');
    if (!snapshot) showNoDataMessage();
  });

  connection.onError(() => {
    if (!snapshot) showNoDataMessage();
  });

  connection.connect();

  // Listen for detection debug results on the raw socket
  const checkSocket = setInterval(() => {
    const sock = (connection as any).socket;
    if (sock) {
      sock.on('debug_detect_results', handleDebugDetectResults);
      clearInterval(checkSocket);
    }
  }, 200);
});

// ── Status Messages ──

function showConnectingMessage(): void {
  const statusEl = document.getElementById('connectionStatusMsg');
  if (statusEl) {
    statusEl.textContent = 'Connecting to SLAM server...';
    statusEl.style.display = 'block';
    return;
  }
  const topbar = document.querySelector('.topbar-stats');
  if (topbar) {
    const span = document.createElement('span');
    span.id = 'connectionStatusMsg';
    span.style.cssText = 'color: #fcc419; font-size: 0.85rem;';
    span.textContent = 'Connecting to SLAM server...';
    topbar.prepend(span);
  }
}

function hideStatusMessage(): void {
  const el = document.getElementById('connectionStatusMsg');
  if (el) el.style.display = 'none';
}

function showNoDataMessage(): void {
  const el = document.getElementById('connectionStatusMsg');
  if (el) {
    el.textContent = 'No scan data available \u2014 start a scan in the viewer first';
    el.style.cssText = 'color: #ff6b6b; font-size: 0.85rem; display: block;';
  }
}

function updateStats(): void {
  if (!snapshot) return;
  const el = (id: string) => document.getElementById(id);
  const pts = el('statPoints');
  const cams = el('statCameras');
  const subs = el('statSubmaps');
  if (pts) pts.textContent = `${formatNumber(snapshot.n_points)} points`;
  if (cams) cams.textContent = `${snapshot.n_cameras} cameras`;
  if (subs) subs.textContent = `${snapshot.num_submaps} submaps`;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

// ── 3D Viewer ──

function createCircleTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const half = size / 2;
  ctx.beginPath();
  ctx.arc(half, half, half - 2, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

function init3DViewer(): void {
  const container = document.getElementById('viewer3d');
  if (!container) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 1000);
  camera.position.set(0, 3, 8);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dLight = new THREE.DirectionalLight(0xffffff, 0.3);
  dLight.position.set(5, 10, 5);
  scene.add(dLight);

  sceneRoot = new THREE.Group();
  sceneRoot.rotation.x = Math.PI;
  scene.add(sceneRoot);

  gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
  scene.add(gridHelper);

  // Point cloud
  pointCloudGeometry = new THREE.BufferGeometry();
  pointCloudMaterial = new THREE.PointsMaterial({
    size: 0.02,
    vertexColors: true,
    sizeAttenuation: true,
    map: createCircleTexture(),
    alphaTest: 0.5,
    transparent: true,
    depthWrite: true,
  });
  pointCloud = new THREE.Points(pointCloudGeometry, pointCloudMaterial);
  sceneRoot.add(pointCloud);

  // Camera group
  cameraGroup = new THREE.Group();
  sceneRoot.add(cameraGroup);

  // Beacon group
  beaconGroup = new THREE.Group();
  sceneRoot.add(beaconGroup);

  // Detection groups
  detectionBoxGroup = new THREE.Group();
  detectionLabelGroup = new THREE.Group();
  sceneRoot.add(detectionBoxGroup);
  sceneRoot.add(detectionLabelGroup);

  // Resize observer
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  ro.observe(container);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

function loadPointCloud(): void {
  if (!snapshot || !snapshot.points.length) return;
  const positions = new Float32Array(snapshot.points.flat());
  const colors = new Float32Array(snapshot.colors.flat());
  pointCloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  pointCloudGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  pointCloudGeometry.computeBoundingSphere();
}

function loadCameras(): void {
  if (!snapshot || !snapshot.camera_positions.length) return;

  // Clear old cameras
  while (cameraGroup.children.length > 0) {
    const child = cameraGroup.children[0];
    if ('geometry' in child) (child as any).geometry?.dispose();
    if ('material' in child) {
      const mat = (child as any).material;
      if (Array.isArray(mat)) mat.forEach((m: THREE.Material) => m.dispose());
      else mat?.dispose();
    }
    cameraGroup.remove(child);
  }

  const trajectoryPoints: THREE.Vector3[] = [];

  for (let i = 0; i < snapshot.camera_positions.length; i++) {
    const pos = snapshot.camera_positions[i];
    const rot = snapshot.camera_rotations[i];
    if (!pos || pos.length !== 3) continue;

    const position = new THREE.Vector3(pos[0], pos[1], pos[2]);
    trajectoryPoints.push(position.clone());

    if (rot && rot.length === 3) {
      const isLatest = i === snapshot.camera_positions.length - 1;
      const frustum = createCameraFrustum(isLatest);
      const rotMatrix = new THREE.Matrix4();
      rotMatrix.set(
        rot[0][0], rot[0][1], rot[0][2], pos[0],
        rot[1][0], rot[1][1], rot[1][2], pos[1],
        rot[2][0], rot[2][1], rot[2][2], pos[2],
        0, 0, 0, 1
      );
      frustum.applyMatrix4(rotMatrix);
      cameraGroup.add(frustum);
    }
  }

  if (trajectoryPoints.length > 1) {
    const geo = new THREE.BufferGeometry().setFromPoints(trajectoryPoints);
    const mat = new THREE.LineBasicMaterial({ color: 0x4da6ff, linewidth: 2, transparent: true, opacity: 0.6 });
    cameraGroup.add(new THREE.Line(geo, mat));
  }
}

function createCameraFrustum(isLatest: boolean): THREE.Group {
  const group = new THREE.Group();
  // Smaller frustums (reduced from s=0.15, d=0.3)
  const s = 0.06;
  const d = 0.12;
  const color = isLatest ? 0x4da6ff : 0xffffff;

  const corners = [
    new THREE.Vector3(-s, -s * 0.75, d),
    new THREE.Vector3(s, -s * 0.75, d),
    new THREE.Vector3(s, s * 0.75, d),
    new THREE.Vector3(-s, s * 0.75, d),
  ];
  const origin = new THREE.Vector3(0, 0, 0);

  const linesMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: isLatest ? 0.9 : 0.3 });

  for (const c of corners) {
    const g = new THREE.BufferGeometry().setFromPoints([origin, c]);
    group.add(new THREE.Line(g, linesMat));
  }

  const rectPts = [...corners, corners[0]];
  const rectGeo = new THREE.BufferGeometry().setFromPoints(rectPts);
  group.add(new THREE.Line(rectGeo, linesMat));

  return group;
}

// ── Beacons in 3D Viewer ──

function updateBeacons(resolvedBeacons: ResolvedBeacon[]): void {
  // Clear old beacons
  while (beaconGroup.children.length > 0) {
    const child = beaconGroup.children[0];
    child.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    });
    beaconGroup.remove(child);
  }

  for (const rb of resolvedBeacons) {
    const color = BEACON_COLORS[(rb.beacon_id - 1) % BEACON_COLORS.length];
    const group = new THREE.Group();

    // Vertical pillar
    const pillarHeight = Math.abs(rb.y) + 0.5;
    const pillarGeom = new THREE.CylinderGeometry(0.008, 0.008, pillarHeight, 6);
    const pillarMat = new THREE.MeshBasicMaterial({ color, opacity: 0.5, transparent: true });
    const pillar = new THREE.Mesh(pillarGeom, pillarMat);
    pillar.position.set(0, -pillarHeight / 2, 0);
    group.add(pillar);

    // Diamond marker
    const diamondGeom = new THREE.OctahedronGeometry(0.06);
    const diamondMat = new THREE.MeshBasicMaterial({ color, opacity: 0.9, transparent: true });
    const diamond = new THREE.Mesh(diamondGeom, diamondMat);
    diamond.scale.set(1, 1.5, 1);
    group.add(diamond);

    // Horizontal ring
    const ringGeom = new THREE.RingGeometry(0.08, 0.1, 24);
    const ringMat = new THREE.MeshBasicMaterial({ color, opacity: 0.3, transparent: true, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(0, -0.01, 0);
    group.add(ring);

    // Point light glow
    const light = new THREE.PointLight(color, 0.5, 1.5);
    light.position.set(0, 0.1, 0);
    group.add(light);

    group.position.set(rb.x, rb.y, rb.z);
    group.userData.beaconId = rb.beacon_id;
    beaconGroup.add(group);
  }
}

// ── Detection overlays on 3D Viewer ──

function updateDetectionOverlays(detections: any[]): void {
  clearGroup(detectionBoxGroup);
  clearGroup(detectionLabelGroup);

  if (!detections || detections.length === 0) return;

  // Assign color per unique query
  const queryColors = new Map<string, THREE.Color>();
  let colorIdx = 0;
  for (const det of detections) {
    if (det.bounding_box && !queryColors.has(det.query)) {
      queryColors.set(det.query, new THREE.Color(BOX_COLORS[colorIdx % BOX_COLORS.length]));
      colorIdx++;
    }
  }

  for (const det of detections) {
    const bb = det.bounding_box;
    if (!bb || !bb.center || !bb.extent || !bb.rotation) continue;

    const color = queryColors.get(det.query) || new THREE.Color(0xffffff);

    // Wireframe OBB
    const boxGeo = new THREE.BoxGeometry(bb.extent[0], bb.extent[1], bb.extent[2]);
    const edgesGeo = new THREE.EdgesGeometry(boxGeo);
    boxGeo.dispose();
    const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 2, opacity: 0.9, transparent: true });
    const wireframe = new THREE.LineSegments(edgesGeo, lineMat);

    const r = bb.rotation;
    const m = new THREE.Matrix4();
    m.set(
      r[0][0], r[0][1], r[0][2], 0,
      r[1][0], r[1][1], r[1][2], 0,
      r[2][0], r[2][1], r[2][2], 0,
      0, 0, 0, 1
    );
    wireframe.applyMatrix4(m);
    wireframe.position.set(bb.center[0], bb.center[1], bb.center[2]);
    detectionBoxGroup.add(wireframe);

    // Label sprite
    const halfY = bb.extent[1] / 2;
    const label = createLabelSprite(det.query, color);
    label.position.set(bb.center[0], bb.center[1] + halfY + 0.15, bb.center[2]);
    detectionLabelGroup.add(label);
  }
}

function createLabelSprite(text: string, color: THREE.Color): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  canvas.width = 256;
  canvas.height = 64;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
  const r = 8;
  ctx.beginPath();
  ctx.moveTo(r, 0); ctx.lineTo(256 - r, 0);
  ctx.quadraticCurveTo(256, 0, 256, r);
  ctx.lineTo(256, 64 - r); ctx.quadraticCurveTo(256, 64, 256 - r, 64);
  ctx.lineTo(r, 64); ctx.quadraticCurveTo(0, 64, 0, 64 - r);
  ctx.lineTo(0, r); ctx.quadraticCurveTo(0, 0, r, 0);
  ctx.closePath(); ctx.fill();

  ctx.strokeStyle = `#${color.getHexString()}`;
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 26px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 32);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.6, 0.15, 1);
  return sprite;
}

function clearGroup(group: THREE.Group): void {
  while (group.children.length > 0) {
    const child = group.children[0];
    if ('geometry' in child) (child as any).geometry?.dispose();
    if ('material' in child) {
      const mat = (child as any).material;
      if (Array.isArray(mat)) mat.forEach((m: THREE.Material) => m.dispose());
      else mat?.dispose();
    }
    group.remove(child);
  }
}

function exportPLY(): void {
  if (!snapshot || !snapshot.points.length) return;
  const n = snapshot.points.length;
  let header = 'ply\nformat ascii 1.0\n';
  header += `element vertex ${n}\n`;
  header += 'property float x\nproperty float y\nproperty float z\n';
  header += 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
  const lines: string[] = [header];
  for (let i = 0; i < n; i++) {
    const p = snapshot.points[i];
    const c = snapshot.colors[i];
    const cr = Math.round(Math.min(1, Math.max(0, c[0])) * 255);
    const cg = Math.round(Math.min(1, Math.max(0, c[1])) * 255);
    const cb = Math.round(Math.min(1, Math.max(0, c[2])) * 255);
    lines.push(`${p[0].toFixed(6)} ${p[1].toFixed(6)} ${p[2].toFixed(6)} ${cr} ${cg} ${cb}\n`);
  }
  const blob = new Blob(lines, { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `slam-summary-${Date.now()}.ply`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Viewer Controls ──

function setupViewerControls(): void {
  const resetBtn = document.getElementById('resetViewBtn');
  const gridBtn = document.getElementById('toggleGridBtn');
  const dloadBtn = document.getElementById('downloadPlyBtn');
  const slider = document.getElementById('pointSizeSlider') as HTMLInputElement | null;

  resetBtn?.addEventListener('click', () => {
    camera.position.set(0, 3, 8);
    controls.target.set(0, 0, 0);
    controls.update();
  });

  gridBtn?.addEventListener('click', () => {
    showGrid = !showGrid;
    gridHelper.visible = showGrid;
    gridBtn.classList.toggle('active', showGrid);
  });

  dloadBtn?.addEventListener('click', () => exportPLY());

  slider?.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    pointCloudMaterial.size = v;
  });
}

// ── 2D Minimap ──

function drawMinimap(): void {
  const canvas = document.getElementById('summaryMinimap') as HTMLCanvasElement | null;
  if (!canvas || !snapshot) return;

  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, w, h);

  // Gather all positions for bounds
  const allX: number[] = [];
  const allZ: number[] = [];

  for (const p of snapshot.camera_positions) {
    if (p && p.length >= 3) { allX.push(p[0]); allZ.push(p[2]); }
  }

  const stride = Math.max(1, Math.floor(snapshot.points.length / 5000));
  for (let i = 0; i < snapshot.points.length; i += stride) {
    const p = snapshot.points[i];
    if (p && p.length >= 3) { allX.push(p[0]); allZ.push(p[2]); }
  }

  if (allX.length === 0) return;

  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const minZ = Math.min(...allZ);
  const maxZ = Math.max(...allZ);
  const rangeX = maxX - minX || 1;
  const rangeZ = maxZ - minZ || 1;

  const margin = 30;
  const usableW = w - margin * 2;
  const usableH = h - margin * 2;
  const scale = Math.min(usableW / rangeX, usableH / rangeZ);

  const toScreenX = (x: number) => margin + (x - minX) * scale + (usableW - rangeX * scale) / 2;
  const toScreenY = (z: number) => margin + (z - minZ) * scale + (usableH - rangeZ * scale) / 2;

  // Draw point cloud (sampled)
  ctx.globalAlpha = 0.3;
  for (let i = 0; i < snapshot.points.length; i += stride) {
    const p = snapshot.points[i];
    const c = snapshot.colors[i];
    if (!p || p.length < 3) continue;
    const sx = toScreenX(p[0]);
    const sy = toScreenY(p[2]);
    const cr = Math.round(Math.min(1, c[0]) * 255);
    const cg = Math.round(Math.min(1, c[1]) * 255);
    const cb = Math.round(Math.min(1, c[2]) * 255);
    ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
    ctx.fillRect(sx - 1, sy - 1, 2, 2);
  }
  ctx.globalAlpha = 1.0;

  // Draw camera trajectory
  if (snapshot.camera_positions.length > 1) {
    ctx.strokeStyle = 'rgba(77, 166, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let first = true;
    for (const pos of snapshot.camera_positions) {
      if (!pos || pos.length < 3) continue;
      const sx = toScreenX(pos[0]);
      const sy = toScreenY(pos[2]);
      if (first) { ctx.moveTo(sx, sy); first = false; }
      else { ctx.lineTo(sx, sy); }
    }
    ctx.stroke();
  }

  // Draw camera dots
  for (let i = 0; i < snapshot.camera_positions.length; i++) {
    const pos = snapshot.camera_positions[i];
    if (!pos || pos.length < 3) continue;
    const sx = toScreenX(pos[0]);
    const sy = toScreenY(pos[2]);
    const isLast = i === snapshot.camera_positions.length - 1;
    ctx.beginPath();
    ctx.arc(sx, sy, isLast ? 5 : 2, 0, Math.PI * 2);
    ctx.fillStyle = isLast ? '#4da6ff' : 'rgba(77, 166, 255, 0.6)';
    ctx.fill();
  }

  // ── Draw beacons on minimap ──
  const beacons = snapshot.resolved_beacons ?? [];
  for (const rb of beacons) {
    const bx = toScreenX(rb.x);
    const by = toScreenY(rb.z);
    const bc = BEACON_COLORS[(rb.beacon_id - 1) % BEACON_COLORS.length];
    const hex = '#' + bc.toString(16).padStart(6, '0');

    // Outer glow ring
    ctx.beginPath();
    ctx.arc(bx, by, 10, 0, Math.PI * 2);
    ctx.fillStyle = hex + '33';
    ctx.fill();

    // Diamond shape
    ctx.beginPath();
    ctx.moveTo(bx, by - 7);
    ctx.lineTo(bx + 5, by);
    ctx.lineTo(bx, by + 7);
    ctx.lineTo(bx - 5, by);
    ctx.closePath();
    ctx.fillStyle = hex;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('B' + rb.beacon_id, bx, by - 11);
  }

  // ── Draw detection bounding boxes on minimap ──
  if (lastDetections.length > 0) {
    const queryColors = new Map<string, string>();
    let cIdx = 0;
    for (const det of lastDetections) {
      if (det.bounding_box && !queryColors.has(det.query)) {
        const dc = BOX_COLORS[cIdx % BOX_COLORS.length];
        queryColors.set(det.query, '#' + dc.toString(16).padStart(6, '0'));
        cIdx++;
      }
    }

    for (const det of lastDetections) {
      const bb = det.bounding_box;
      if (!bb || !bb.center) continue;

      const cx = toScreenX(bb.center[0]);
      const cy = toScreenY(bb.center[2]);
      const detColor = queryColors.get(det.query) || '#ffffff';

      // Draw as a filled circle + label
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = detColor + '88';
      ctx.fill();
      ctx.strokeStyle = detColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // If has extent, draw projected box on XZ plane
      if (bb.extent) {
        const hw = (bb.extent[0] / 2) * scale;
        const hd = (bb.extent[2] / 2) * scale;
        ctx.strokeStyle = detColor + 'aa';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(cx - hw, cy - hd, hw * 2, hd * 2);
      }

      // Label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(det.query, cx, cy - 10);
    }
  }

  // Legend
  ctx.fillStyle = '#666';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Overhead View (XZ plane)', 10, h - 8);
}

// ── Chat ──

function setupChat(): void {
  const form = document.getElementById('chatForm') as HTMLFormElement | null;
  const input = document.getElementById('chatInput') as HTMLInputElement | null;
  if (!form || !input) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    addChatMessage('user', msg);
    chatHistory.push({ role: 'user', content: msg });

    const typingEl = addTypingIndicator();

    try {
      const reply = await queryDedalus(msg);
      typingEl.remove();
      addChatMessage('assistant', reply);
      chatHistory.push({ role: 'assistant', content: reply });
    } catch (err) {
      typingEl.remove();
      const errMsg = err instanceof Error ? err.message : 'Failed to get response';
      addChatMessage('assistant', 'Error: ' + errMsg);
    }
  });
}

function addChatMessage(role: 'user' | 'assistant', content: string): void {
  const container = document.getElementById('chatMessages');
  if (!container) return;

  const msgEl = document.createElement('div');
  msgEl.className = 'chat-message ' + role;
  msgEl.innerHTML = '<div class="chat-bubble">' + escapeHtml(content) + '</div>';
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
}

function addTypingIndicator(): HTMLElement {
  const container = document.getElementById('chatMessages');
  const el = document.createElement('div');
  el.className = 'chat-message assistant';
  el.innerHTML = '<div class="chat-bubble"><span class="chat-typing"><span></span><span></span><span></span></span></div>';
  container?.appendChild(el);
  container!.scrollTop = container!.scrollHeight;
  return el;
}

async function queryDedalus(userMsg: string): Promise<string> {
  const apiKey = ((import.meta as any).env as any).VITE_DEDALUS_API_KEY as string;
  if (!apiKey) throw new Error('API key not configured');

  const contextParts: string[] = [];
  if (snapshot) {
    contextParts.push('The user has completed a 3D SLAM scan with ' + snapshot.n_points + ' points, ' + snapshot.n_cameras + ' camera frames, and ' + snapshot.num_submaps + ' submaps.');
  }
  if (lastDetections.length > 0) {
    const objs = lastDetections.map((d: any) => d.query).filter((v: string, i: number, a: string[]) => a.indexOf(v) === i);
    contextParts.push('Objects detected in the scan: ' + objs.join(', ') + '.');
  }

  const systemPrompt = 'You are a helpful assistant for a 3D SLAM mapping system. You help users understand their scanned environment and the objects detected within it. Keep responses concise (2-4 sentences). ' + contextParts.join(' ');

  const messages = [
    { role: 'system', content: systemPrompt },
    ...chatHistory.slice(-10),
    { role: 'user', content: userMsg },
  ];

  const response = await fetch('https://api.dedaluslabs.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + apiKey,
    },
    body: JSON.stringify({
      model: 'gpt-4',
      messages,
      temperature: 0.5,
      max_tokens: 512,
    }),
  });

  if (!response.ok) throw new Error('API error: ' + response.statusText);

  const data = await response.json();
  return data.choices?.[0]?.message?.content ?? 'No response received.';
}

// ── Detection Debug Pipeline ──

function setupDetection(): void {
  const input = document.getElementById('detectionInput') as HTMLTextAreaElement | null;
  const btn = document.getElementById('detectBtn') as HTMLButtonElement | null;

  if (!input || !btn) return;

  btn.addEventListener('click', () => doDetection());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doDetection();
  });
}

function doDetection(): void {
  const input = document.getElementById('detectionInput') as HTMLTextAreaElement | null;
  const clipInput = document.getElementById('clipThreshold') as HTMLInputElement | null;
  const samInput = document.getElementById('samThreshold') as HTMLInputElement | null;
  const btn = document.getElementById('detectBtn') as HTMLButtonElement | null;
  const btnText = document.getElementById('detectBtnText') as HTMLSpanElement | null;

  if (!input) return;
  const raw = input.value.trim();
  if (!raw) return;

  const queries = raw.split(/[\n,]+/).map(s => s.trim()).filter(Boolean);
  if (queries.length === 0) return;

  const clipThreshold = parseFloat(clipInput?.value ?? '0.2') || 0.2;
  const samThreshold = parseFloat(samInput?.value ?? '0.3') || 0.3;

  showDetStatus(
    'Running full detection pipeline for [' + queries.join(', ') + '] (CLIP>=' + clipThreshold + ', SAM>=' + samThreshold + ')...',
    'info'
  );

  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = 'Searching...';

  const sock = (connection as any).socket;
  if (!sock) {
    showDetStatus('Not connected to server', 'error');
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = 'Search';
    return;
  }

  sock.emit('debug_detect', {
    queries,
    clip_thresholds: { default: clipThreshold },
    sam_thresholds: { default: samThreshold },
  });
}

function handleDebugDetectResults(data: DebugDetectResponse): void {
  const btn = document.getElementById('detectBtn') as HTMLButtonElement | null;
  const btnText = document.getElementById('detectBtnText') as HTMLSpanElement | null;
  if (btn) btn.disabled = false;
  if (btnText) btnText.textContent = 'Search';

  if (data.error) {
    showDetStatus('Error: ' + data.error, 'error');
    return;
  }

  lastDetectResponse = data;
  currentDetFilter = 'all';

  const { queries, clip_thresholds, sam_thresholds, frames, raw_detection_count,
    deduped_detection_count, detections, total_frames_scanned, query_time_ms } = data;

  const clipDefault = clip_thresholds?.default ?? 0.2;
  const samDefault = sam_thresholds?.default ?? 0.3;
  const aboveCount = frames.filter(f => f.above_threshold).length;
  const samCount = frames.filter(f => f.sam_masks.length > 0).length;

  showDetStatus(
    '[' + queries.join(', ') + '] -- ' + total_frames_scanned + ' frame-query combos | ' +
    aboveCount + ' above CLIP (' + clipDefault + ') | ' +
    samCount + ' with SAM masks (SAM>=' + samDefault + ') | ' +
    raw_detection_count + ' raw -> ' + deduped_detection_count + ' deduped | ' +
    query_time_ms + 'ms',
    deduped_detection_count > 0 ? 'success' : 'warn'
  );

  // Store final detections for overlays
  lastDetections = detections;

  // Update 3D overlays
  updateDetectionOverlays(detections);
  // Redraw minimap with detections
  drawMinimap();

  // Build filter buttons
  const filterBtns = document.getElementById('detFilterBtns');
  if (filterBtns) {
    filterBtns.classList.remove('hidden');
    filterBtns.innerHTML = '';
    const filters: { key: 'all' | 'above' | 'sam' | 'detected'; label: string; count: number }[] = [
      { key: 'all', label: 'All frames', count: frames.length },
      { key: 'above', label: 'Above threshold', count: aboveCount },
      { key: 'sam', label: 'SAM masks found', count: samCount },
      { key: 'detected', label: 'Valid detections', count: raw_detection_count },
    ];
    for (const f of filters) {
      const fbtn = document.createElement('button');
      fbtn.className = 'det-filter-btn' + (f.key === 'all' ? ' active' : '');
      fbtn.textContent = f.label + ' (' + f.count + ')';
      fbtn.addEventListener('click', () => {
        currentDetFilter = f.key;
        filterBtns.querySelectorAll('.det-filter-btn').forEach(b => b.classList.remove('active'));
        fbtn.classList.add('active');
        renderDetFrameGrid(data.frames);
      });
      filterBtns.appendChild(fbtn);
    }
  }

  const emptyState = document.getElementById('detEmptyState');
  const resultsWrap = document.getElementById('detResultsWrap');
  if (emptyState) emptyState.classList.add('hidden');
  if (resultsWrap) resultsWrap.classList.remove('hidden');

  renderDetFrameGrid(frames);
  renderDetFinalDetections(detections, queries);

  // Notify chat context
  if (deduped_detection_count > 0) {
    const objNames = detections.map((d: any) => d.query).filter((v: string, i: number, a: string[]) => a.indexOf(v) === i);
    addChatMessage('assistant', 'Detection found ' + deduped_detection_count + ' object(s): ' + objNames.join(', ') + '. Locations are shown on both viewers.');
    chatHistory.push({ role: 'assistant', content: 'Detected objects: ' + objNames.join(', ') });
  }
}

function renderDetFrameGrid(frames: FrameDiag[]): void {
  const grid = document.getElementById('detResultsGrid');
  if (!grid) return;

  const filtered = frames.filter(f => {
    if (currentDetFilter === 'above') return f.above_threshold;
    if (currentDetFilter === 'sam') return f.sam_masks.length > 0;
    if (currentDetFilter === 'detected') return f.detections_before_dedup.length > 0;
    return true;
  });

  grid.innerHTML = '';

  if (filtered.length === 0) {
    grid.innerHTML = '<p class="det-no-results">No frames match this filter.</p>';
    return;
  }

  for (const f of filtered) {
    const card = document.createElement('div');
    const aboveCls = f.above_threshold ? 'above' : 'below';
    const hasMasks = f.sam_masks.length > 0;
    const hasDets = f.detections_before_dedup.length > 0;
    card.className = 'det-frame-card ' + aboveCls + (hasMasks ? ' has-masks' : '') + (hasDets ? ' has-dets' : '');

    const simPct = Math.min(Math.max(f.clip_similarity * 100, 0), 100);
    const simColor = similarityColor(f.clip_similarity);

    let imgHtml = f.thumbnail
      ? '<img src="data:image/jpeg;base64,' + f.thumbnail + '" />'
      : '<div class="det-fc-placeholder">No image</div>';

    let badgesHtml = '';
    if (hasMasks) badgesHtml += '<span class="det-fc-badge det-fc-badge-sam">' + f.sam_masks.length + ' mask' + (f.sam_masks.length > 1 ? 's' : '') + '</span>';
    if (!f.above_threshold) badgesHtml += '<span class="det-fc-badge det-fc-badge-below">below</span>';

    card.innerHTML =
      '<div class="det-fc-img">' + imgHtml +
        '<span class="det-fc-sim" style="background:' + simColor + '">' + simPct.toFixed(1) + '%</span>' +
        badgesHtml +
      '</div>' +
      '<div class="det-fc-meta">' +
        '<div class="det-fc-row"><span>Query</span><span class="det-mono">' + escapeHtml(f.query) + '</span></div>' +
        '<div class="det-fc-row"><span>Submap</span><span>' + f.submap_id + '</span></div>' +
        '<div class="det-fc-row"><span>Frame</span><span>' + f.frame_idx + '</span></div>' +
        '<div class="det-fc-row"><span>CLIP sim</span><span class="det-mono" style="color:' + simColor + '">' + f.clip_similarity.toFixed(4) + '</span></div>' +
        '<div class="det-fc-row"><span>CLIP thresh</span><span class="det-mono">' + f.clip_threshold_used + '</span></div>' +
        '<div class="det-fc-row"><span>SAM thresh</span><span class="det-mono">' + f.sam_threshold_used + '</span></div>' +
        (f.resolution ? '<div class="det-fc-row"><span>Res</span><span class="det-mono">' + f.resolution + '</span></div>' : '') +
        (f.sam_error ? '<div class="det-fc-row det-fc-error"><span>SAM error</span><span>' + escapeHtml(f.sam_error) + '</span></div>' : '') +
        '<div class="det-fc-bar"><div class="det-fc-bar-fill" style="width:' + simPct + '%;background:' + simColor + '"></div></div>' +
      '</div>' +
      (hasMasks ? renderSamMasksInline(f.sam_masks) : '');

    grid.appendChild(card);
  }
}

function renderSamMasksInline(masks: SamMaskDiag[]): string {
  let html = '<div class="det-fc-masks">';
  for (const m of masks) {
    const maskColor = !m.above_sam_threshold ? '#f87171'
      : m.dedup_kept === true ? '#4ade80'
      : m.dedup_kept === false ? '#facc15'
      : m.has_3d_box ? '#4ade80' : '#facc15';

    let detailHtml = '';
    if (!m.above_sam_threshold) {
      detailHtml = '<span style="color:#f87171">below SAM threshold (' + m.sam_threshold_used + ')</span>';
    } else {
      detailHtml = m.has_3d_box ? '3D box: valid' : '3D box: none (insufficient points)';
    }

    html += '<div class="det-fc-mask-item">';
    html += '<img src="data:image/png;base64,' + m.mask_image + '" />';
    html += '<div class="det-fc-mask-info">';
    html += '<span class="det-fc-mask-score" style="color:' + maskColor + '">SAM ' + (m.score * 100).toFixed(1) + '%</span>';
    html += '<span class="det-fc-mask-detail">' + detailHtml + '</span>';
    if (m.bbox_3d) {
      html += '<span class="det-fc-mask-detail det-mono">ext: [' + m.bbox_3d.extent.map((v: number) => v.toFixed(3)).join(', ') + ']</span>';
    }
    if (m.dedup_kept === true) html += '<span class="det-fc-mask-detail" style="color:#4ade80">kept after dedup</span>';
    if (m.dedup_kept === false) html += '<span class="det-fc-mask-detail" style="color:#facc15">discarded by dedup</span>';
    html += '</div></div>';
  }
  html += '</div>';
  return html;
}

function renderDetFinalDetections(detections: any[], _queries: string[]): void {
  const section = document.getElementById('detFinalDetections');
  if (!section) return;

  section.innerHTML = '';
  if (detections.length === 0) {
    section.innerHTML = '<h3 class="section-title">Final Detections (after dedup)</h3><p class="det-no-results">No detections survived deduplication / validation.</p>';
    return;
  }

  let html = '<h3 class="section-title">Final Detections (after dedup): ' + detections.length + '</h3>';
  html += '<div class="det-list">';
  for (const d of detections) {
    const bb = d.bounding_box;
    html += '<div class="det-card"><div class="det-header">';
    html += '<span class="det-query">' + escapeHtml(d.query) + '</span>';
    html += '<span class="det-conf">' + ((d.confidence ?? 0) * 100).toFixed(1) + '%</span>';
    html += '</div><div class="det-body">';
    html += '<div class="det-fc-row"><span>Submap</span><span>' + d.matched_submap + '</span></div>';
    html += '<div class="det-fc-row"><span>Frame</span><span>' + d.matched_frame + '</span></div>';
    if (bb) {
      html += '<div class="det-fc-row"><span>Center</span><span class="det-mono">[' + bb.center.map((v: number) => v.toFixed(3)).join(', ') + ']</span></div>';
      html += '<div class="det-fc-row"><span>Extent</span><span class="det-mono">[' + bb.extent.map((v: number) => v.toFixed(3)).join(', ') + ']</span></div>';
    }
    html += '</div></div>';
  }
  html += '</div>';
  section.innerHTML = html;
}

function similarityColor(sim: number): string {
  if (sim >= 0.3) return '#4ade80';
  if (sim >= 0.2) return '#94d82d';
  if (sim >= 0.15) return '#facc15';
  if (sim >= 0.1) return '#ff922b';
  return '#f87171';
}

function showDetStatus(msg: string, kind: 'info' | 'success' | 'warn' | 'error'): void {
  const bar = document.getElementById('detStatusBar');
  if (!bar) return;
  bar.textContent = msg;
  bar.className = 'det-status-bar det-status-' + kind;
}

// ── Utilities ──

function escapeHtml(str: string): string {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
