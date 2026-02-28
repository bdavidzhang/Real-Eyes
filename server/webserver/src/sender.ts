import { io, Socket } from 'socket.io-client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './sender.css';

/**
 * Camera Streamer with AR Beacon Overlay
 *
 * Beacon lifecycle:
 * 1. User places beacon → flash on screen for 3 seconds, then fade away
 * 2. Beacon stays invisible on camera while pending server resolution
 * 3. Once server resolves world position (submap processed), beacon reappears
 *    on the camera feed using SLAM camera pose projection
 *
 * AR positioning uses the SLAM-computed camera rotation matrix and position
 * (received via slam_update) to project beacon 3D world coordinates into the
 * phone's 2D camera frame. This eliminates magnetometer jitter and correctly
 * handles walk-through (beacon appears behind camera when you pass it).
 */

const SERVER_URL = import.meta.env.VITE_SERVER_URL || window.location.origin;

// ── DOM Elements ──
const video = document.getElementById('video') as HTMLVideoElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const arOverlay = document.getElementById('arOverlay') as HTMLCanvasElement;
const arCtx = arOverlay.getContext('2d')!;
const statusDot = document.getElementById('statusDot') as HTMLElement;
const statusText = document.getElementById('statusText') as HTMLElement;
const fpsText = document.getElementById('fpsText') as HTMLElement;
const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;
const minimapCanvas = document.getElementById('minimapCanvas') as HTMLCanvasElement;
const minimapCtx = minimapCanvas.getContext('2d')!;
const minimapContainer = document.getElementById('minimap-container') as HTMLElement;
const minimapToggle = document.getElementById('minimapToggle') as HTMLButtonElement;
const handModeBtn = document.getElementById('handModeBtn') as HTMLButtonElement;
const buttonModeBtn = document.getElementById('buttonModeBtn') as HTMLButtonElement;
const placeBeaconBtn = document.getElementById('placeBeaconBtn') as HTMLButtonElement;
const clearBeaconsBtn = document.getElementById('clearBeaconsBtn') as HTMLButtonElement;
const beaconStatus = document.getElementById('beaconStatus') as HTMLElement;

// ── State ──
let socket: Socket | null = null;
let streaming = false;
let streamInterval: ReturnType<typeof setInterval> | null = null;
let framesSent = 0;
let frameSentCounter = 0;
let lastFpsUpdate = Date.now();

// ══════════════════════════════════════════
// ── Minimap State ──
// ══════════════════════════════════════════

/** All camera positions received from SLAM (x/z birds-eye) */
const cameraPath: { x: number; z: number }[] = [];

/** Minimap mode: 'follow' = user-centered trail, 'overview' = fit everything */
let minimapMode: 'follow' | 'overview' = 'follow';

/** Zoom radius in world units for follow mode */
const MINIMAP_FOLLOW_RADIUS = 2.0;

// ══════════════════════════════════════════
// ── SLAM Camera Pose (for AR projection) ──
// ══════════════════════════════════════════

/** Latest camera position from SLAM */
let lastCamPos: [number, number, number] | null = null;

/** Latest camera rotation matrix (3x3, camera-to-world) from SLAM */
let lastCamRot: number[][] | null = null;

/**
 * Gyro-interpolation state.
 * SLAM updates arrive every ~7s. Between updates we use the gyroscope
 * (DeviceOrientation) to continuously adjust the projection so beacons
 * track the phone's rotation at 60 fps.
 *
 * On each slam_update we snapshot the current device orientation.
 * Between updates: delta = current_ori - snapshot_ori, applied as an
 * additional rotation on the SLAM camera-space coordinates.
 */
let slamOriSnapshot: DeviceOri = { alpha: 0, beta: 0, gamma: 0 };

/** Approximate camera horizontal FOV in radians */
const CAMERA_HFOV_RAD = (65 * Math.PI) / 180;
/** Approximate camera vertical FOV in radians */
const CAMERA_VFOV_RAD = (50 * Math.PI) / 180;
const DEG2RAD = Math.PI / 180;

// ══════════════════════════════════════════
// ── DeviceOrientation (gyro interpolation) ──
// ══════════════════════════════════════════

interface DeviceOri {
  alpha: number;
  beta: number;
  gamma: number;
}

let smoothOrientation: DeviceOri = { alpha: 0, beta: 0, gamma: 0 };
let orientationAvailable = false;
const ORI_SMOOTH = 0.15;

function initDeviceOrientation(): void {
  const doe = DeviceOrientationEvent as any;
  if (typeof doe.requestPermission === 'function') {
    doe.requestPermission()
      .then((state: string) => {
        if (state === 'granted') {
          window.addEventListener('deviceorientation', handleOrientation, true);
          orientationAvailable = true;
        }
      })
      .catch(() => console.warn('DeviceOrientation permission denied'));
  } else {
    window.addEventListener('deviceorientation', handleOrientation, true);
  }
}

function handleOrientation(event: DeviceOrientationEvent): void {
  if (event.alpha !== null) {
    const raw = {
      alpha: event.alpha,
      beta: event.beta ?? 0,
      gamma: event.gamma ?? 0,
    };
    orientationAvailable = true;
    smoothOrientation.alpha = smoothAngle(smoothOrientation.alpha, raw.alpha, ORI_SMOOTH);
    smoothOrientation.beta += ORI_SMOOTH * (raw.beta - smoothOrientation.beta);
    smoothOrientation.gamma += ORI_SMOOTH * (raw.gamma - smoothOrientation.gamma);
  }
}

function smoothAngle(current: number, target: number, factor: number): number {
  const delta = ((target - current) % 360 + 540) % 360 - 180;
  return (current + factor * delta + 360) % 360;
}

// ══════════════════════════════════════════
// ── Beacon System ──
// ══════════════════════════════════════════

type BeaconMode = 'hand' | 'button';
let beaconMode: BeaconMode = 'hand';
let nextBeaconId = 1;

type BeaconDisplayState = 'flash' | 'hidden' | 'active';

interface Beacon {
  id: number;
  name: string;
  x: number;
  y: number;
  z: number;
  pending: boolean;
  frameNumber: number;
  placementOri: DeviceOri;
  color: string;
  displayState: BeaconDisplayState;
  placedAt: number;
  flashOpacity: number;
}

const beacons: Beacon[] = [];

const BEACON_COLORS = [
  '#FFD700', '#FF6B6B', '#4ECDC4', '#A78BFA', '#FB923C', '#34D399',
];

const FLASH_DURATION_MS = 3000;

// Hand tracking
let handLandmarker: HandLandmarker | null = null;
let handDetectionActive = false;
let lastHandDetected = false;
let handCooldown = false;
const HAND_COOLDOWN_MS = 3000;

// ══════════════════════════════════════════
// ── Camera Init ──
// ══════════════════════════════════════════

async function initCamera(): Promise<void> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    resizeArOverlay();
    window.addEventListener('resize', resizeArOverlay);
    console.log(`Camera initialized: ${video.videoWidth}x${video.videoHeight}`);
  } catch (error) {
    console.error('Camera error:', error);
    updateStatus('error', `Camera error: ${(error as Error).message}`);
  }
}

function resizeArOverlay(): void {
  arOverlay.width = window.innerWidth;
  arOverlay.height = window.innerHeight;
}

// ── MediaPipe Hand Landmarker Init ──
async function initHandTracking(): Promise<void> {
  try {
    beaconStatus.textContent = 'Loading hand model...';
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
    });
    console.log('Hand landmarker loaded');
    beaconStatus.textContent = 'Hand tracking ready';
    setTimeout(() => {
      if (beaconStatus.textContent === 'Hand tracking ready') beaconStatus.textContent = '';
    }, 2000);
  } catch (error) {
    console.error('Hand tracking init error:', error);
    beaconStatus.textContent = 'Hand tracking unavailable';
  }
}

function detectHand(): boolean {
  if (!handLandmarker || !video.videoWidth) return false;
  try {
    const result = handLandmarker.detectForVideo(video, performance.now());
    return result.landmarks.length > 0;
  } catch {
    return false;
  }
}

function checkHandForBeacon(): void {
  if (beaconMode !== 'hand' || !handDetectionActive || handCooldown) return;
  const handVisible = detectHand();
  if (handVisible && !lastHandDetected) {
    placeBeacon();
    handCooldown = true;
    beaconStatus.textContent = `Cooldown (${HAND_COOLDOWN_MS / 1000}s)...`;
    setTimeout(() => {
      handCooldown = false;
      if (beaconMode === 'hand') beaconStatus.textContent = 'Ready - show hand';
    }, HAND_COOLDOWN_MS);
  }
  lastHandDetected = handVisible;
}

// ── Status UI ──
function updateStatus(
  state: 'connected' | 'disconnected' | 'connecting' | 'error',
  text?: string,
): void {
  statusDot.className = 'status-dot';
  switch (state) {
    case 'connected':
      statusDot.classList.add('dot-connected');
      statusText.textContent = text ?? 'Connected - Streaming';
      break;
    case 'connecting':
      statusDot.classList.add('dot-connecting');
      statusText.textContent = text ?? 'Connecting...';
      break;
    case 'error':
      statusDot.classList.add('dot-error');
      statusText.textContent = text ?? 'Connection Error';
      break;
    default:
      statusDot.classList.add('dot-disconnected');
      statusText.textContent = text ?? 'Disconnected';
      break;
  }
}

// ── FPS ──
function updateFps(): void {
  framesSent++;
  const now = Date.now();
  const elapsed = now - lastFpsUpdate;
  if (elapsed >= 1000) {
    fpsText.textContent = `${(framesSent / (elapsed / 1000)).toFixed(1)} fps`;
    fpsText.style.display = 'inline';
    framesSent = 0;
    lastFpsUpdate = now;
  }
}

// ── Frame Capture & Send ──
function sendFrame(): void {
  if (!streaming || !socket?.connected) return;
  try {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    frameSentCounter++;
    socket.emit('frame', { image: base64, timestamp: Date.now() / 1000 });
    updateFps();
    if (frameSentCounter % 5 === 0) checkHandForBeacon();
  } catch (error) {
    console.error('Frame send error:', error);
  }
}

// ══════════════════════════════════════════
// ── Beacon Placement ──
// ══════════════════════════════════════════

function placeBeacon(): void {
  const id = nextBeaconId++;
  const name = `B${id}`;
  const color = BEACON_COLORS[(id - 1) % BEACON_COLORS.length];
  const estimatedPos = cameraPath.length > 0 ? cameraPath[cameraPath.length - 1] : { x: 0, z: 0 };

  const beacon: Beacon = {
    id,
    name,
    x: estimatedPos.x,
    z: estimatedPos.z,
    y: 0,
    pending: true,
    frameNumber: frameSentCounter,
    placementOri: { ...smoothOrientation },
    color,
    displayState: 'flash',
    placedAt: Date.now(),
    flashOpacity: 1,
  };

  beacons.push(beacon);
  beaconStatus.textContent = `${name} placed`;
  drawMinimap();

  if (socket?.connected) {
    socket.emit('place_beacon', { beacon_id: id, frame_number: frameSentCounter });
  }

  setTimeout(() => {
    if (beacon.displayState === 'flash') beacon.displayState = 'hidden';
  }, FLASH_DURATION_MS);

  console.log(`📍 ${name} placed | frame=${frameSentCounter}`);
}

function clearAllBeacons(): void {
  beacons.length = 0;
  nextBeaconId = 1;
  beaconStatus.textContent = 'Beacons cleared';
  drawMinimap();
  if (socket?.connected) socket.emit('clear_beacons');
  setTimeout(() => {
    if (beaconStatus.textContent === 'Beacons cleared') beaconStatus.textContent = '';
  }, 2000);
}

// ══════════════════════════════════════════
// ── SLAM Camera Projection ──
// ══════════════════════════════════════════

/**
 * Project a 3D world point into the phone's camera frame using the latest
 * SLAM camera pose. Returns { x, y, depth } in screen coordinates,
 * or null if no camera pose is available.
 *
 * The camera extrinsics from SLAM define a camera-to-world transform:
 *   p_world = R * p_cam + t
 *
 * So to go world→camera:
 *   p_cam = R^T * (p_world - t)
 *
 * Then perspective divide + FOV-based focal length → screen coords.
 */
/**
 * Compute the signed angular difference (in degrees) from angle a to b,
 * handling 0/360 wraparound.  Result in [-180, +180].
 */
function angleDelta(a: number, b: number): number {
  return ((b - a) % 360 + 540) % 360 - 180;
}

/**
 * Project a 3D world point into the phone's camera frame.
 *
 * Uses the SLAM camera pose (updated every ~7s) plus a gyroscope delta
 * to interpolate rotation continuously between SLAM updates.
 *
 *   SLAM transform (camera-to-world):  p_world = R * p_cam + t
 *   Inverse (world-to-camera):          p_cam  = R^T * (p_world - t)
 *
 * After computing p_cam from the SLAM pose, we rotate it by the
 * gyro delta (how much the phone has turned since the last slam_update)
 * so the projection tracks the phone at 60 fps.
 */
function projectToScreen(
  wx: number,
  wy: number,
  wz: number,
): { sx: number; sy: number; depth: number } | null {
  if (!lastCamPos || !lastCamRot) return null;

  const R = lastCamRot;
  const t = lastCamPos;

  // World → SLAM camera space
  const dx = wx - t[0];
  const dy = wy - t[1];
  const dz = wz - t[2];

  let cx = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz;
  let cy = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz;
  let cz = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz;

  // ── Gyro-delta interpolation ──
  // Compute how much the phone has rotated since the last SLAM update
  // and apply the inverse rotation so the beacon tracks the phone in real-time.
  if (orientationAvailable) {
    // Yaw delta (left-right panning) — most important for tracking
    const deltaYawDeg = angleDelta(slamOriSnapshot.alpha, smoothOrientation.alpha);
    // Pitch delta (tilting phone up/down)
    const deltaPitchDeg = smoothOrientation.beta - slamOriSnapshot.beta;

    const yaw = -deltaYawDeg * DEG2RAD;   // negate: phone turned right → point shifts left
    const pitch = -deltaPitchDeg * DEG2RAD;

    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch);
    const sinP = Math.sin(pitch);

    // Rotate around camera Y-axis (yaw)
    const cx2 = cx * cosY + cz * sinY;
    const cz2 = -cx * sinY + cz * cosY;

    // Rotate around camera X-axis (pitch)
    const cy2 = cy * cosP - cz2 * sinP;
    const cz3 = cy * sinP + cz2 * cosP;

    cx = cx2;
    cy = cy2;
    cz = cz3;
  }

  const depth = cz;

  const screenW = arOverlay.width;
  const screenH = arOverlay.height;
  const fx = (screenW / 2) / Math.tan(CAMERA_HFOV_RAD / 2);
  const fy = (screenH / 2) / Math.tan(CAMERA_VFOV_RAD / 2);

  if (Math.abs(depth) < 0.001) return null;

  const sx = screenW / 2 + (cx / depth) * fx;
  const sy = screenH / 2 + (cy / depth) * fy;

  return { sx, sy, depth };
}

// ══════════════════════════════════════════
// ── AR Overlay ──
// ══════════════════════════════════════════

function drawArOverlay(): void {
  const w = arOverlay.width;
  const h = arOverlay.height;
  arCtx.clearRect(0, 0, w, h);

  if (beacons.length === 0) return;

  const now = Date.now();

  for (const beacon of beacons) {
    // Update flash opacity
    if (beacon.displayState === 'flash') {
      const elapsed = now - beacon.placedAt;
      if (elapsed < FLASH_DURATION_MS * 0.5) {
        beacon.flashOpacity = 1;
      } else {
        beacon.flashOpacity = Math.max(
          0,
          1 - (elapsed - FLASH_DURATION_MS * 0.5) / (FLASH_DURATION_MS * 0.5),
        );
      }
    }

    const shouldDraw = beacon.displayState === 'flash' || beacon.displayState === 'active';
    if (!shouldDraw) continue;

    // ── Flash phase: show at screen center (immediate feedback, no projection)
    if (beacon.displayState === 'flash') {
      const opacity = beacon.flashOpacity;
      drawBeaconMarker(beacon, w / 2, h / 2 - 40, opacity, 1.0);
      continue;
    }

    // ── Active phase: use SLAM projection
    const proj = projectToScreen(beacon.x, beacon.y, beacon.z);
    if (!proj) continue; // no camera pose yet

    // Point is behind camera → user has walked past the beacon
    if (proj.depth <= 0) {
      // Show edge indicator pointing backwards
      drawEdgeIndicator(beacon, w / 2, h + 60, w, h, 0.6);
      continue;
    }

    // Distance from camera to beacon (in world units)
    const dist = proj.depth;

    // Scale marker: appears large when close, small when far
    // Full size at dist <= 0.5m, shrinks to 0.25x at dist >= 5m
    const beaconScale = Math.max(0.25, Math.min(1.2, 1.0 / (dist * 0.5 + 0.3)));

    const onScreen =
      proj.sx > -60 && proj.sx < w + 60 && proj.sy > -60 && proj.sy < h + 60;

    if (onScreen) {
      drawBeaconMarker(beacon, proj.sx, proj.sy, 1.0, beaconScale);

      // Distance label
      arCtx.save();
      arCtx.font = `bold ${Math.round(11 * beaconScale)}px Inter, sans-serif`;
      arCtx.textAlign = 'center';
      arCtx.fillStyle = '#ffffff99';
      arCtx.shadowColor = '#000';
      arCtx.shadowBlur = 3;
      arCtx.fillText(
        `${dist.toFixed(1)}m`,
        proj.sx,
        proj.sy + 36 * beaconScale,
      );
      arCtx.shadowBlur = 0;
      arCtx.restore();
    } else {
      drawEdgeIndicator(beacon, proj.sx, proj.sy, w, h, 0.8);
    }
  }
}

function drawBeaconMarker(
  beacon: Beacon,
  sx: number,
  sy: number,
  opacity: number,
  scale: number,
): void {
  arCtx.save();
  arCtx.globalAlpha = opacity;

  const baseSize = 14 * scale;

  // Outer glow
  const glowRadius = 30 * scale;
  const gradient = arCtx.createRadialGradient(sx, sy, 0, sx, sy, glowRadius);
  gradient.addColorStop(0, beacon.color + '60');
  gradient.addColorStop(1, beacon.color + '00');
  arCtx.fillStyle = gradient;
  arCtx.fillRect(sx - glowRadius, sy - glowRadius, glowRadius * 2, glowRadius * 2);

  // Diamond shape
  arCtx.beginPath();
  arCtx.moveTo(sx, sy - baseSize);
  arCtx.lineTo(sx + baseSize * 0.7, sy);
  arCtx.lineTo(sx, sy + baseSize);
  arCtx.lineTo(sx - baseSize * 0.7, sy);
  arCtx.closePath();
  arCtx.fillStyle = beacon.color;
  arCtx.fill();
  arCtx.strokeStyle = '#ffffff';
  arCtx.lineWidth = 1.5 * scale;
  arCtx.stroke();

  // Stalk below diamond
  const stalkLen = 20 * scale;
  arCtx.beginPath();
  arCtx.moveTo(sx, sy + baseSize);
  arCtx.lineTo(sx, sy + baseSize + stalkLen);
  arCtx.strokeStyle = beacon.color + 'AA';
  arCtx.lineWidth = 2 * scale;
  arCtx.stroke();

  // Pulsing ring (active only)
  if (beacon.displayState === 'active') {
    const pulse = (Math.sin(Date.now() / 400) + 1) / 2;
    arCtx.beginPath();
    arCtx.arc(sx, sy, (18 + pulse * 8) * scale, 0, Math.PI * 2);
    arCtx.strokeStyle = beacon.color;
    arCtx.globalAlpha = opacity * 0.3 * (1 - pulse);
    arCtx.lineWidth = 2 * scale;
    arCtx.stroke();
    arCtx.globalAlpha = opacity;
  }

  // Label
  arCtx.font = `bold ${Math.round(12 * scale)}px Inter, sans-serif`;
  arCtx.textAlign = 'center';
  arCtx.fillStyle = '#ffffff';
  arCtx.shadowColor = '#000000';
  arCtx.shadowBlur = 4;
  arCtx.fillText(beacon.name, sx, sy - baseSize - 6 * scale);
  arCtx.shadowBlur = 0;

  // Flash label
  if (beacon.displayState === 'flash') {
    arCtx.font = `${Math.round(9 * scale)}px Inter, sans-serif`;
    arCtx.fillStyle = '#ffffff88';
    arCtx.fillText('placing...', sx, sy + baseSize + stalkLen + 14 * scale);
  }

  arCtx.restore();
}

function drawEdgeIndicator(
  beacon: Beacon,
  sx: number,
  sy: number,
  w: number,
  h: number,
  opacity: number,
): void {
  const margin = 30;
  const cx = Math.max(margin, Math.min(w - margin, sx));
  const cy = Math.max(margin, Math.min(h - margin, sy));
  const angle = Math.atan2(sy - cy, sx - cx);

  arCtx.save();
  arCtx.globalAlpha = opacity * 0.7;
  arCtx.translate(cx, cy);
  arCtx.rotate(angle);
  arCtx.beginPath();
  arCtx.moveTo(10, 0);
  arCtx.lineTo(-6, -6);
  arCtx.lineTo(-6, 6);
  arCtx.closePath();
  arCtx.fillStyle = beacon.color;
  arCtx.fill();
  arCtx.restore();

  arCtx.save();
  arCtx.globalAlpha = opacity;
  arCtx.font = 'bold 10px Inter, sans-serif';
  arCtx.textAlign = 'center';
  arCtx.fillStyle = beacon.color;
  arCtx.fillText(beacon.name, cx, cy - 14);
  arCtx.restore();
}

// AR animation loop
let arAnimationId: number | null = null;

function startArLoop(): void {
  function loop() {
    drawArOverlay();
    arAnimationId = requestAnimationFrame(loop);
  }
  loop();
}

function stopArLoop(): void {
  if (arAnimationId !== null) {
    cancelAnimationFrame(arAnimationId);
    arAnimationId = null;
  }
  arCtx.clearRect(0, 0, arOverlay.width, arOverlay.height);
}

// ══════════════════════════════════════════
// ── Minimap Drawing ──
// ══════════════════════════════════════════

function drawMinimap(): void {
  const w = minimapCanvas.width;
  const h = minimapCanvas.height;
  const pad = 16;

  minimapCtx.clearRect(0, 0, w, h);
  minimapCtx.fillStyle = 'rgba(0, 0, 0, 0.85)';
  minimapCtx.fillRect(0, 0, w, h);

  if (cameraPath.length === 0 && beacons.length === 0) {
    minimapCtx.fillStyle = '#666';
    minimapCtx.font = '11px Inter, sans-serif';
    minimapCtx.textAlign = 'center';
    minimapCtx.fillText('Waiting for data...', w / 2, h / 2);
    return;
  }

  const userPos = cameraPath.length > 0 ? cameraPath[cameraPath.length - 1] : { x: 0, z: 0 };

  let centerX: number;
  let centerZ: number;
  let viewRadius: number;

  if (minimapMode === 'follow') {
    // ── Follow mode: user centered, fixed zoom
    centerX = userPos.x;
    centerZ = userPos.z;
    viewRadius = MINIMAP_FOLLOW_RADIUS;
  } else {
    // ── Overview mode: fit everything
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of cameraPath) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.z < minZ) minZ = p.z;
      if (p.z > maxZ) maxZ = p.z;
    }
    for (const b of beacons) {
      if (b.x < minX) minX = b.x;
      if (b.x > maxX) maxX = b.x;
      if (b.z < minZ) minZ = b.z;
      if (b.z > maxZ) maxZ = b.z;
    }
    centerX = (minX + maxX) / 2;
    centerZ = (minZ + maxZ) / 2;
    viewRadius = Math.max((maxX - minX) / 2, (maxZ - minZ) / 2, 0.5) * 1.3;
  }

  const drawableSize = w - pad * 2;
  const scale = drawableSize / (viewRadius * 2);
  const toX = (x: number) => w / 2 + (x - centerX) * scale;
  const toY = (z: number) => h / 2 + (z - centerZ) * scale;

  // Grid
  minimapCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  minimapCtx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const gy = pad + (i / 4) * drawableSize;
    minimapCtx.beginPath(); minimapCtx.moveTo(pad, gy); minimapCtx.lineTo(w - pad, gy); minimapCtx.stroke();
    const gx = pad + (i / 4) * drawableSize;
    minimapCtx.beginPath(); minimapCtx.moveTo(gx, pad); minimapCtx.lineTo(gx, h - pad); minimapCtx.stroke();
  }

  // Path trail (faded gradient for follow mode)
  if (cameraPath.length > 1) {
    if (minimapMode === 'follow') {
      // Draw with fading trail effect: recent = bright, old = dim
      const total = cameraPath.length;
      const trailLen = Math.min(total, 200);
      const startIdx = total - trailLen;

      for (let i = startIdx + 1; i < total; i++) {
        const progress = (i - startIdx) / trailLen; // 0=old, 1=recent
        minimapCtx.beginPath();
        minimapCtx.moveTo(toX(cameraPath[i - 1].x), toY(cameraPath[i - 1].z));
        minimapCtx.lineTo(toX(cameraPath[i].x), toY(cameraPath[i].z));
        minimapCtx.strokeStyle = `rgba(255, 255, 255, ${0.1 + progress * 0.6})`;
        minimapCtx.lineWidth = 1 + progress * 1;
        minimapCtx.stroke();
      }
    } else {
      // Overview: full path with uniform style
      minimapCtx.beginPath();
      minimapCtx.moveTo(toX(cameraPath[0].x), toY(cameraPath[0].z));
      for (let i = 1; i < cameraPath.length; i++) {
        minimapCtx.lineTo(toX(cameraPath[i].x), toY(cameraPath[i].z));
      }
      minimapCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      minimapCtx.lineWidth = 1.5;
      minimapCtx.lineJoin = 'round';
      minimapCtx.stroke();

      // Start dot
      minimapCtx.beginPath();
      minimapCtx.arc(toX(cameraPath[0].x), toY(cameraPath[0].z), 3, 0, Math.PI * 2);
      minimapCtx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      minimapCtx.fill();
    }
  }

  // User position indicator (always at center in follow mode)
  if (cameraPath.length > 0) {
    const ux = toX(userPos.x);
    const uy = toY(userPos.z);

    // Direction indicator (small triangle) - uses heading from path direction
    let heading = 0;
    if (cameraPath.length >= 2) {
      const prev = cameraPath[cameraPath.length - 2];
      const cur = cameraPath[cameraPath.length - 1];
      heading = Math.atan2(cur.z - prev.z, cur.x - prev.x);
    }

    minimapCtx.save();
    minimapCtx.translate(ux, uy);
    minimapCtx.rotate(heading);

    // Direction triangle
    minimapCtx.beginPath();
    minimapCtx.moveTo(7, 0);
    minimapCtx.lineTo(-4, -4);
    minimapCtx.lineTo(-4, 4);
    minimapCtx.closePath();
    minimapCtx.fillStyle = '#ffffff';
    minimapCtx.fill();
    minimapCtx.restore();

    // User dot (on top of triangle)
    minimapCtx.beginPath();
    minimapCtx.arc(ux, uy, 4, 0, Math.PI * 2);
    minimapCtx.fillStyle = '#ffffff';
    minimapCtx.fill();
    minimapCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    minimapCtx.lineWidth = 1;
    minimapCtx.stroke();
  }

  // Beacons
  for (const b of beacons) {
    const bx = toX(b.x);
    const by = toY(b.z);

    // Only draw if on canvas
    if (bx < -10 || bx > w + 10 || by < -10 || by > h + 10) continue;

    const sz = 5;
    minimapCtx.beginPath();
    minimapCtx.moveTo(bx, by - sz);
    minimapCtx.lineTo(bx + sz, by);
    minimapCtx.lineTo(bx, by + sz);
    minimapCtx.lineTo(bx - sz, by);
    minimapCtx.closePath();
    minimapCtx.fillStyle = b.pending ? b.color + '66' : b.color + 'CC';
    minimapCtx.strokeStyle = b.color;
    minimapCtx.fill();
    minimapCtx.lineWidth = 1;
    minimapCtx.stroke();

    minimapCtx.fillStyle = b.color;
    minimapCtx.font = '9px Inter, sans-serif';
    minimapCtx.textAlign = 'center';
    minimapCtx.fillText(b.name, bx, by - sz - 4);
  }

  // Mode indicator
  minimapCtx.fillStyle = 'rgba(255, 255, 255, 0.25)';
  minimapCtx.font = '9px Inter, sans-serif';
  minimapCtx.textAlign = 'left';
  minimapCtx.fillText(minimapMode === 'follow' ? '📍 Tap to expand' : '🗺️ Tap to follow', pad, h - 6);

  if (beacons.length > 0) {
    minimapCtx.textAlign = 'right';
    minimapCtx.fillStyle = 'rgba(255, 200, 50, 0.5)';
    minimapCtx.fillText(`${beacons.length} beacon${beacons.length > 1 ? 's' : ''}`, w - pad, h - 6);
  }
}

// ══════════════════════════════════════════
// ── Connection & Streaming ──
// ══════════════════════════════════════════

function startStreaming(): void {
  if (streaming) return;
  updateStatus('connecting');

  socket = io(SERVER_URL, { transports: ['websocket', 'polling'], reconnection: true });

  socket.on('connect', () => {
    updateStatus('connected');
    socket!.emit('start_slam');
    streaming = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    handDetectionActive = true;
    streamInterval = setInterval(sendFrame, 100);
    startArLoop();
  });

  socket.on('slam_update', (data: {
    camera_positions?: number[][];
    camera_rotations?: number[][][];
    resolved_beacons?: Array<{ beacon_id: number; x: number; y: number; z: number }>;
  }) => {
    if (data.camera_positions && data.camera_positions.length > 0) {
      cameraPath.length = 0;
      for (const pos of data.camera_positions) {
        if (pos && pos.length >= 3) cameraPath.push({ x: pos[0], z: pos[2] });
      }
      // Store latest camera position
      const last = data.camera_positions[data.camera_positions.length - 1];
      if (last && last.length >= 3) {
        lastCamPos = [last[0], last[1], last[2]];
      }
    }

    // Store latest camera rotation matrix (3x3)
    if (data.camera_rotations && data.camera_rotations.length > 0) {
      const lastRot = data.camera_rotations[data.camera_rotations.length - 1];
      if (lastRot && lastRot.length === 3) {
        lastCamRot = lastRot;
      }
    }

    // Snapshot current device orientation for gyro-delta interpolation.
    // Between this slam_update and the next (~7s later), the gyro delta
    // from this snapshot drives continuous AR tracking.
    if (orientationAvailable) {
      slamOriSnapshot = { ...smoothOrientation };
    }

    if (data.resolved_beacons) {
      for (const rb of data.resolved_beacons) {
        const beacon = beacons.find((b) => b.id === rb.beacon_id);
        if (beacon) {
          beacon.x = rb.x;
          beacon.y = rb.y ?? 0;
          beacon.z = rb.z;
          beacon.pending = false;
          if (beacon.displayState !== 'active') {
            beacon.displayState = 'active';
            console.log(`✅ ${beacon.name} resolved → active (SLAM projection)`);
          }
        }
      }
    }

    drawMinimap();
  });

  socket.on('slam_reset', () => {
    cameraPath.length = 0;
    beacons.length = 0;
    nextBeaconId = 1;
    lastCamPos = null;
    lastCamRot = null;
    slamOriSnapshot = { alpha: 0, beta: 0, gamma: 0 };
    drawMinimap();
  });

  socket.on('disconnect', () => {
    updateStatus('disconnected');
    stopStreaming();
  });

  socket.on('frame_received', (data: { status: string }) => {
    if (data.status === 'queue_full') console.warn('Server queue full');
  });

  socket.on('connect_error', (error: Error) => {
    console.error('Connection error:', error);
    updateStatus('error');
  });
}

function stopStreaming(): void {
  streaming = false;
  handDetectionActive = false;
  if (streamInterval) { clearInterval(streamInterval); streamInterval = null; }
  stopArLoop();
  if (socket) { socket.emit('stop_slam'); socket.disconnect(); socket = null; }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  fpsText.style.display = 'none';
  updateStatus('disconnected', 'Stopped');
  cameraPath.length = 0;
  lastCamPos = null;
  lastCamRot = null;
  slamOriSnapshot = { alpha: 0, beta: 0, gamma: 0 };
  drawMinimap();
}

// ══════════════════════════════════════════
// ── UI Event Handlers ──
// ══════════════════════════════════════════

// Minimap toggle (minimize)
minimapToggle.addEventListener('click', (e) => {
  e.stopPropagation();
  const isMinimized = minimapContainer.classList.contains('minimized');
  minimapContainer.classList.toggle('minimized', !isMinimized);
  minimapToggle.textContent = isMinimized ? '−' : '+';
  if (isMinimized) drawMinimap();
});

// Minimap click: toggle follow ↔ overview
minimapCanvas.addEventListener('click', () => {
  minimapMode = minimapMode === 'follow' ? 'overview' : 'follow';

  if (minimapMode === 'overview') {
    // Expand canvas for overview
    minimapCanvas.width = 300;
    minimapCanvas.height = 300;
    minimapContainer.style.width = '300px';
  } else {
    // Shrink back for follow
    minimapCanvas.width = 200;
    minimapCanvas.height = 200;
    minimapContainer.style.width = '200px';
  }

  drawMinimap();
});

handModeBtn.addEventListener('click', () => {
  beaconMode = 'hand';
  handModeBtn.classList.add('active');
  buttonModeBtn.classList.remove('active');
  placeBeaconBtn.style.display = 'none';
  beaconStatus.textContent = handLandmarker ? 'Ready - show hand' : 'Loading hand model...';
});

buttonModeBtn.addEventListener('click', () => {
  beaconMode = 'button';
  buttonModeBtn.classList.add('active');
  handModeBtn.classList.remove('active');
  placeBeaconBtn.style.display = 'block';
  beaconStatus.textContent = '';
});

placeBeaconBtn.addEventListener('click', () => {
  if (!streaming) { beaconStatus.textContent = 'Start streaming first'; return; }
  placeBeacon();
});

clearBeaconsBtn.addEventListener('click', () => clearAllBeacons());

startBtn.addEventListener('click', startStreaming);
stopBtn.addEventListener('click', stopStreaming);

// ── Init ──
window.addEventListener('DOMContentLoaded', () => {
  initCamera();
  initDeviceOrientation();
  initHandTracking();
  drawMinimap();
});
