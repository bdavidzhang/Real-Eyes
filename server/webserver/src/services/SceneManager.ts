import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SLAMUpdate, ControlsConfig, ResolvedBeacon, DetectionResult } from '../types';

// Color palette for detected object bounding boxes
const BOX_COLORS = [
  0x4da6ff, 0xff6b6b, 0x51cf66, 0xfcc419, 0xcc5de8,
  0xff922b, 0x20c997, 0xf06595, 0x5c7cfa, 0x94d82d,
];

/**
 * Manages Three.js scene, camera, and rendering
 */
export class SceneManager {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;

  // Scene objects
  private pointCloud: THREE.Points;
  private pointCloudGeometry: THREE.BufferGeometry;
  private pointCloudMaterial: THREE.PointsMaterial;
  private cameraGroup: THREE.Group;
  private beaconGroup: THREE.Group;
  private detectionGroup: THREE.Group;
  private detectionBoxGroup: THREE.Group;
  private detectionLabelGroup: THREE.Group;
  private gridHelper: THREE.GridHelper;
  private axesHelper: THREE.AxesHelper;
  private sceneRoot: THREE.Group;  // Parent group for 180° rotation

  // Beacon colors (match sender)
  private static BEACON_COLORS = [
    0xFFD700, 0xFF6B6B, 0x4ECDC4, 0xA78BFA, 0xFB923C, 0x34D399,
  ];
  private knownBeaconIds = new Set<number>();

  // Configuration
  private config: ControlsConfig = {
    showCameras: true,
    showPoints: true,
    showGrid: true,
    showAxes: true,
    pointSize: 0.02,
    cameraSize: 1.0,
    followCamera: false,
    flipY: false,
    showDetectionBoxes: true,
    showDetectionLabels: true,
  };

  private animationId: number | null = null;

  constructor(container: HTMLElement) {
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    // Camera setup
    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.01,
      1000
    );
    this.camera.position.set(0, 3, 8);

    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 0, 0);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight.position.set(5, 10, 5);
    this.scene.add(directionalLight);

    // Scene root group — rotated 180° around X so the scene appears right-side-up
    this.sceneRoot = new THREE.Group();
    this.sceneRoot.rotation.x = Math.PI;
    this.scene.add(this.sceneRoot);

    // Grid
    this.gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    this.scene.add(this.gridHelper);  // grid stays in scene root (not rotated)

    // Axes
    this.axesHelper = new THREE.AxesHelper(1);
    this.scene.add(this.axesHelper);  // axes stay in scene root (not rotated)

    // Point cloud - use circle sprite to match Viser's round point rendering
    this.pointCloudGeometry = new THREE.BufferGeometry();
    this.pointCloudMaterial = new THREE.PointsMaterial({
      size: this.config.pointSize,
      vertexColors: true,
      sizeAttenuation: true,
      map: this.createCircleTexture(),
      alphaTest: 0.5,
      transparent: true,
      depthWrite: true,
    });
    this.pointCloud = new THREE.Points(
      this.pointCloudGeometry,
      this.pointCloudMaterial
    );
    this.sceneRoot.add(this.pointCloud);

    // Camera group
    this.cameraGroup = new THREE.Group();
    this.sceneRoot.add(this.cameraGroup);

    // Beacon group
    this.beaconGroup = new THREE.Group();
    this.sceneRoot.add(this.beaconGroup);
    // Detection bounding box group
    this.detectionGroup = new THREE.Group();
    this.detectionBoxGroup = new THREE.Group();
    this.detectionLabelGroup = new THREE.Group();
    this.detectionGroup.add(this.detectionBoxGroup);
    this.detectionGroup.add(this.detectionLabelGroup);
    this.sceneRoot.add(this.detectionGroup);

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());

    // Start animation loop
    this.animate();
  }

  /**
   * Update visualization with new SLAM data
   */
  updateVisualization(data: SLAMUpdate): void {
    try {
      console.log('📊 Processing visualization data:', {
        points: data.n_points,
        cameras: data.n_cameras,
        submaps: data.num_submaps,
        loops: data.num_loops,
      });

      // Update point cloud
      if (data.points && data.points.length > 0) {
        const positions = new Float32Array(data.points.flat());
        const colors = new Float32Array(data.colors.flat());

        this.pointCloudGeometry.setAttribute(
          'position',
          new THREE.BufferAttribute(positions, 3)
        );
        this.pointCloudGeometry.setAttribute(
          'color',
          new THREE.BufferAttribute(colors, 3)
        );
        this.pointCloudGeometry.computeBoundingSphere();

        console.log(`✅ Point cloud updated: ${positions.length / 3} points`);
      }

      // Update cameras
      if (data.camera_positions && data.camera_positions.length > 0) {
        this.updateCameras(data.camera_positions, data.camera_rotations);
      }

      // Update beacons
      if (data.resolved_beacons && data.resolved_beacons.length > 0) {
        this.updateBeacons(data.resolved_beacons);
      }

      // Follow latest camera if enabled
      if (
        this.config.followCamera &&
        data.camera_positions &&
        data.camera_positions.length > 0
      ) {
        this.followLatestCamera(data.camera_positions);
      }
    } catch (error) {
      console.error('❌ Visualization error:', error);
    }
  }

  /**
   * Update camera frustums and trajectory
   */
  private updateCameras(
    positions: number[][],
    rotations: number[][][]
  ): void {
    // Clear old cameras
    while (this.cameraGroup.children.length > 0) {
      const child = this.cameraGroup.children[0];
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((mat) => mat.dispose());
        } else {
          child.material.dispose();
        }
      }
      this.cameraGroup.remove(child);
    }

    const trajectoryPoints: THREE.Vector3[] = [];

    for (let i = 0; i < positions.length; i++) {
      const pos = positions[i];
      const rot = rotations[i];

      if (!pos || pos.length !== 3 || !rot || rot.length !== 3) {
        console.warn(`Invalid camera data at index ${i}`);
        continue;
      }

      const position = new THREE.Vector3(pos[0], pos[1], pos[2]);
      trajectoryPoints.push(position.clone());

      // Create rotation matrix
      const rotMatrix = new THREE.Matrix4();
      rotMatrix.set(
        rot[0][0], rot[0][1], rot[0][2], pos[0],
        rot[1][0], rot[1][1], rot[1][2], pos[1],
        rot[2][0], rot[2][1], rot[2][2], pos[2],
        0, 0, 0, 1
      );

      // Show frustum for every camera frame
      const isLatest = i === positions.length - 1;
      const frustum = this.createCameraFrustum(isLatest);
      frustum.applyMatrix4(rotMatrix);
      this.cameraGroup.add(frustum);
    }

    // Draw trajectory line
    if (trajectoryPoints.length > 1) {
      const trajectoryGeometry = new THREE.BufferGeometry().setFromPoints(
        trajectoryPoints
      );
      const trajectoryMaterial = new THREE.LineBasicMaterial({
        color: 0xcccccc,
        linewidth: 2,
        opacity: 0.6,
        transparent: true,
      });
      const trajectoryLine = new THREE.Line(
        trajectoryGeometry,
        trajectoryMaterial
      );
      this.cameraGroup.add(trajectoryLine);
    }

    this.cameraGroup.visible = this.config.showCameras;
    console.log(
      `✅ Cameras updated: ${positions.length} total, showing ${trajectoryPoints.length}`
    );
  }

  /**
   * Create camera frustum visualization
   */
  private createCameraFrustum(isLatest = false): THREE.Group {
    const group = new THREE.Group();

    // Camera cone/pyramid
    const geometry = new THREE.ConeGeometry(
      0.05 * this.config.cameraSize,
      0.15 * this.config.cameraSize,
      4
    );
    const material = new THREE.MeshBasicMaterial({
      color: isLatest ? 0xffffff : 0xaaaaaa,
      wireframe: true,
      opacity: 0.6,
      transparent: true,
    });
    const cone = new THREE.Mesh(geometry, material);
    cone.rotation.x = Math.PI / 2;
    group.add(cone);

    // Camera center sphere
    const sphereGeom = new THREE.SphereGeometry(0.02 * this.config.cameraSize);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: isLatest ? 0xffffff : 0xaaaaaa,
      opacity: 0.7,
      transparent: true,
    });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    group.add(sphere);

    return group;
  }

  /**
   * Update 3D beacon markers in the scene.
   * Each beacon is a vertical pillar with a glowing diamond on top.
   * Only adds new beacons (avoids re-creating existing ones, but updates position).
   */
  updateBeacons(resolvedBeacons: ResolvedBeacon[]): void {
    // Clear and rebuild — simple and correct since beacon count is small
    while (this.beaconGroup.children.length > 0) {
      const child = this.beaconGroup.children[0];
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else {
          child.material.dispose();
        }
      }
      // Groups need recursive cleanup
      if (child instanceof THREE.Group) {
        child.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry.dispose();
            if (Array.isArray(obj.material)) {
              obj.material.forEach((m) => m.dispose());
            } else {
              obj.material.dispose();
            }
          }
        });
      }
      this.beaconGroup.remove(child);
    }
    this.knownBeaconIds.clear();

    for (const rb of resolvedBeacons) {
      if (this.knownBeaconIds.has(rb.beacon_id)) continue;
      this.knownBeaconIds.add(rb.beacon_id);

      const color = SceneManager.BEACON_COLORS[(rb.beacon_id - 1) % SceneManager.BEACON_COLORS.length];
      const group = new THREE.Group();

      // Vertical pillar (thin cylinder from ground to beacon y)
      const pillarHeight = Math.abs(rb.y) + 0.5;
      const pillarGeom = new THREE.CylinderGeometry(0.008, 0.008, pillarHeight, 6);
      const pillarMat = new THREE.MeshBasicMaterial({
        color,
        opacity: 0.5,
        transparent: true,
      });
      const pillar = new THREE.Mesh(pillarGeom, pillarMat);
      pillar.position.set(0, -pillarHeight / 2, 0);
      group.add(pillar);

      // Diamond marker at beacon position
      const diamondGeom = new THREE.OctahedronGeometry(0.06);
      const diamondMat = new THREE.MeshBasicMaterial({
        color,
        opacity: 0.9,
        transparent: true,
      });
      const diamond = new THREE.Mesh(diamondGeom, diamondMat);
      diamond.scale.set(1, 1.5, 1);
      group.add(diamond);

      // Outer ring (pulsing effect via scale animation is handled in animate)
      const ringGeom = new THREE.RingGeometry(0.08, 0.1, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color,
        opacity: 0.3,
        transparent: true,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.rotation.x = -Math.PI / 2; // horizontal ring
      ring.position.set(0, -0.01, 0);
      group.add(ring);

      // Point light glow
      const light = new THREE.PointLight(color, 0.5, 1.5);
      light.position.set(0, 0.1, 0);
      group.add(light);

      group.position.set(rb.x, rb.y, rb.z);
      group.userData.beaconId = rb.beacon_id;
      this.beaconGroup.add(group);
    }

    if (resolvedBeacons.length > 0) {
      console.log(`📍 ${resolvedBeacons.length} beacons rendered in 3D scene`);
    }
  }

  /**
   * Follow the latest camera position
   * Positions the viewer camera behind and above the target
   */
  private followLatestCamera(positions: number[][]): void {
    if (positions.length === 0) return;

    const lastPos = positions[positions.length - 1];
    const targetPos = new THREE.Vector3(lastPos[0], lastPos[1], lastPos[2]);

    // Position the viewer camera behind and above the target
    const offsetPos = new THREE.Vector3(
      targetPos.x - 3,
      targetPos.y + 2,
      targetPos.z
    );
    this.camera.position.lerp(offsetPos, 0.05);
    this.controls.target.lerp(targetPos, 0.05);
  }

  /**
   * Create a circular sprite texture for point rendering (matches Viser's circle point_shape)
   */
  private createCircleTexture(): THREE.Texture {
    const size = 64;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    // Draw filled circle
    const center = size / 2;
    const radius = size / 2 - 1;
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  }

  /**
   * Update 3D bounding boxes for detected objects.
   * Called each time a slam_update arrives with detections.
   */
  updateDetections(detections: DetectionResult[]): void {
    // Dispose old bounding boxes
    this.clearDetectionGroup();

    if (!detections || detections.length === 0) return;

    // Assign a consistent color per unique query string
    const queryColors = new Map<string, THREE.Color>();
    let colorIdx = 0;
    for (const det of detections) {
      if (det.success && !queryColors.has(det.query)) {
        queryColors.set(det.query, new THREE.Color(BOX_COLORS[colorIdx % BOX_COLORS.length]));
        colorIdx++;
      }
    }

    for (const det of detections) {
      if (!det.success || !det.bounding_box) continue;
      const bb = det.bounding_box;
      if (!bb.center || !bb.extent || !bb.rotation) continue;

      const color = queryColors.get(det.query) || new THREE.Color(0xffffff);

      // Build wireframe box from center + extent + rotation
      const wireframe = this.createOBBWireframe(bb, color);
      this.detectionBoxGroup.add(wireframe);

      // Label sprite above the box
      const halfY = bb.extent[1] / 2;
      const cx = bb.center[0];
      const cz = bb.center[2];
      const maxY = bb.center[1] + halfY;
      const label = this.createLabelSprite(det.query, color);
      label.position.set(cx, maxY + 0.15, cz);
      this.detectionLabelGroup.add(label);
    }

    // Apply current visibility
    this.detectionBoxGroup.visible = this.config.showDetectionBoxes;
    this.detectionLabelGroup.visible = this.config.showDetectionLabels;
  }

  /**
   * Remove all detection bounding boxes from the scene.
   */
  clearDetections(): void {
    this.clearDetectionGroup();
  }

  private clearDetectionGroup(): void {
    for (const group of [this.detectionBoxGroup, this.detectionLabelGroup]) {
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
  }

  private createOBBWireframe(bb: { center: number[]; extent: number[]; rotation: number[][] }, color: THREE.Color): THREE.LineSegments {
    // Use Three.js BoxGeometry + EdgesGeometry — no corner ordering issues
    const boxGeo = new THREE.BoxGeometry(bb.extent[0], bb.extent[1], bb.extent[2]);
    const edgesGeo = new THREE.EdgesGeometry(boxGeo);
    boxGeo.dispose();

    const material = new THREE.LineBasicMaterial({ color, linewidth: 2, opacity: 0.9, transparent: true });
    const lineSegments = new THREE.LineSegments(edgesGeo, material);

    // Apply OBB rotation (3x3 matrix)
    const r = bb.rotation;
    const m = new THREE.Matrix4();
    m.set(
      r[0][0], r[0][1], r[0][2], 0,
      r[1][0], r[1][1], r[1][2], 0,
      r[2][0], r[2][1], r[2][2], 0,
      0, 0, 0, 1,
    );
    lineSegments.applyMatrix4(m);

    // Position at OBB center
    lineSegments.position.set(bb.center[0], bb.center[1], bb.center[2]);

    return lineSegments;
  }

  private createLabelSprite(text: string, color: THREE.Color): THREE.Sprite {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.width = 256;
    canvas.height = 64;

    // Rounded rect background
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
    sprite.scale.set(0.8, 0.2, 1);
    return sprite;
  }

  /**
   * Reset camera view
   */
  resetView(): void {
    this.camera.position.set(0, 3, 8);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    console.log('🔄 View reset');
  }

  /**
   * Clear all visualization data
   */
  clearScene(): void {
    // Clear point cloud
    this.pointCloudGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([]), 3)
    );
    this.pointCloudGeometry.setAttribute(
      'color',
      new THREE.BufferAttribute(new Float32Array([]), 3)
    );

    // Clear cameras
    while (this.cameraGroup.children.length > 0) {
      const child = this.cameraGroup.children[0];
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((mat) => mat.dispose());
        } else {
          child.material.dispose();
        }
      }
      this.cameraGroup.remove(child);
    }

    // Clear beacons
    this.updateBeacons([]);

    console.log('🗑️  Scene cleared');
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ControlsConfig>): void {
    this.config = { ...this.config, ...newConfig };

    // Apply config changes
    this.cameraGroup.visible = this.config.showCameras;
    this.pointCloud.visible = this.config.showPoints;
    this.gridHelper.visible = this.config.showGrid;
    this.axesHelper.visible = this.config.showAxes;
    this.pointCloudMaterial.size = this.config.pointSize;
    this.detectionBoxGroup.visible = this.config.showDetectionBoxes;
    this.detectionLabelGroup.visible = this.config.showDetectionLabels;

    // Toggle 180° rotation (default is rotated; toggle removes it)
    this.sceneRoot.rotation.x = this.config.flipY ? 0 : Math.PI;
  }

  /**
   * Get current configuration
   */
  getConfig(): ControlsConfig {
    return { ...this.config };
  }

  /**
   * Get renderer canvas
   */
  getCanvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  /**
   * Animation loop
   */
  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  /**
   * Handle window resize
   */
  private onWindowResize(): void {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }

    this.clearScene();
    this.clearDetectionGroup();
    this.pointCloudGeometry.dispose();
    this.pointCloudMaterial.dispose();
    this.gridHelper.geometry.dispose();
    if (Array.isArray(this.gridHelper.material)) {
      this.gridHelper.material.forEach((mat) => mat.dispose());
    } else {
      this.gridHelper.material.dispose();
    }
    this.axesHelper.dispose();
    this.renderer.dispose();
    this.controls.dispose();

    window.removeEventListener('resize', () => this.onWindowResize());
  }

  /**
   * Export the current point cloud as a PLY file and trigger download.
   * Returns false if no points are available.
   */
  exportPointCloudPLY(): boolean {
    const posAttr = this.pointCloudGeometry.getAttribute('position') as THREE.BufferAttribute | null;
    const colAttr = this.pointCloudGeometry.getAttribute('color') as THREE.BufferAttribute | null;

    if (!posAttr || posAttr.count === 0) {
      return false;
    }

    const numPoints = posAttr.count;
    const positions = posAttr.array as Float32Array;
    const colors = colAttr ? (colAttr.array as Float32Array) : null;

    // Build PLY file
    let header = 'ply\n';
    header += 'format ascii 1.0\n';
    header += `element vertex ${numPoints}\n`;
    header += 'property float x\n';
    header += 'property float y\n';
    header += 'property float z\n';
    header += 'property uchar red\n';
    header += 'property uchar green\n';
    header += 'property uchar blue\n';
    header += 'end_header\n';

    const lines: string[] = [header];

    for (let i = 0; i < numPoints; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      let r = 128, g = 128, b = 128;
      if (colors) {
        r = Math.round(Math.min(1, Math.max(0, colors[i * 3])) * 255);
        g = Math.round(Math.min(1, Math.max(0, colors[i * 3 + 1])) * 255);
        b = Math.round(Math.min(1, Math.max(0, colors[i * 3 + 2])) * 255);
      }
      lines.push(`${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)} ${r} ${g} ${b}\n`);
    }

    const blob = new Blob(lines, { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vggt-slam-pointcloud-${Date.now()}.ply`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    return true;
  }
}
