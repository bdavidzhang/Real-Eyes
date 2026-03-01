import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

/**
 * Interactive 3D scene for the landing page hero section
 * Shows a Gaussian splat reconstruction with camera poses
 */
export class HeroScene {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private pointCloud: THREE.Points | null = null;
  private cameraFrustums: THREE.Group;
  private animationId: number | null = null;
  private container: HTMLElement;
  private userInteracted: boolean = false;
  private viewportOffset: number = 0; // 0 = centered, target = left offset in pixels
  private targetViewportOffset: number = 0;

  constructor(container: HTMLElement) {
    this.container = container;

    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);
    this.scene.fog = new THREE.Fog(0x0a0a0a, 8, 25);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    this.camera.position.set(5, 3, 5);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 15;
    this.controls.maxPolarAngle = Math.PI / 1.5;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.5;

    // Camera frustums group
    this.cameraFrustums = new THREE.Group();
    this.scene.add(this.cameraFrustums);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 5);
    this.scene.add(dirLight);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-5, 5, -5);
    this.scene.add(dirLight2);

    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x333333, 0x1a1a1a);
    gridHelper.position.y = -1;
    this.scene.add(gridHelper);

    // Load the Gaussian splat
    this.loadGaussianSplat();

    // Handle resize
    window.addEventListener('resize', () => this.onResize());

    // Start animation
    this.animate();
  }

  private async loadGaussianSplat(): Promise<void> {
    const loader = new PLYLoader();

    try {
      console.log('Loading Gaussian splat...');

      const geometry = await loader.loadAsync('/assets/Gaussian_splat.ply');

      console.log('Gaussian splat loaded successfully');

      // Center the geometry and flip y-axis
      geometry.computeBoundingBox();
      const boundingBox = geometry.boundingBox!;
      const center = new THREE.Vector3();
      boundingBox.getCenter(center);
      // Translate to center and flip y-axis by negating y translation
      geometry.translate(-center.x, center.y, -center.z);

      // Scale to flip y-axis
      geometry.scale(1, -1, 1);

      // Calculate positioning before creating the point cloud
      const size = new THREE.Vector3();
      boundingBox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);

      // Create material for the point cloud - start invisible
      const material = new THREE.PointsMaterial({
        size: 0.02,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0, // Start invisible
      });

      this.pointCloud = new THREE.Points(geometry, material);
      this.scene.add(this.pointCloud);

      // Adjust camera - more zoomed in, centered view
      const fov = this.camera.fov * (Math.PI / 180);
      let cameraZ = Math.abs(maxDim / Math.tan(fov / 2));
      cameraZ *= 0.6; // Zoom in more

      // Position camera to view the point cloud centered in the canvas
      this.camera.position.set(cameraZ * 0.5, cameraZ * 0.4, cameraZ * 0.8);

      // Set rotation center (target) at origin
      this.camera.lookAt(0, 0, 0);
      this.controls.target.set(0, 0, 0);
      this.controls.update();

      // Add camera frustums along a trajectory
      this.createCameraTrajectory();

    } catch (error) {
      console.error('Error loading Gaussian splat:', error);
      // Fallback to a simple demo scene if loading fails
      this.createFallbackScene();
    }
  }

  private createFallbackScene(): void {
    console.log('Creating fallback scene...');

    // Simple cube as fallback
    const geometry = new THREE.BoxGeometry(2, 2, 2);
    const material = new THREE.MeshStandardMaterial({
      color: 0x808080,
      wireframe: true
    });
    const cube = new THREE.Mesh(geometry, material);
    this.scene.add(cube);

    this.createCameraTrajectory();
  }

  private createCameraTrajectory(): void {
    // Create a path of camera frustums
    const frustumPositions = [
      new THREE.Vector3(-2, 0.5, 2),
      new THREE.Vector3(-1, 0.5, 1),
      new THREE.Vector3(0, 0.5, 0),
      new THREE.Vector3(1, 0.5, -0.5),
      new THREE.Vector3(2, 0.5, -1),
    ];

    frustumPositions.forEach((pos, i) => {
      const frustum = this.createCameraFrustum();
      frustum.position.copy(pos);

      // Look towards the center
      frustum.lookAt(0, 0, -1);

      // Color based on position in trajectory (gradient from white to gray)
      const t = i / (frustumPositions.length - 1);
      const color = new THREE.Color().setHSL(0, 0, 0.9 - t * 0.4);
      frustum.children.forEach(child => {
        if (child instanceof THREE.LineSegments) {
          (child.material as THREE.LineBasicMaterial).color = color;
        }
      });

      this.cameraFrustums.add(frustum);
    });

    // Draw trajectory line
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(frustumPositions);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      opacity: 0.3,
      transparent: true
    });
    const line = new THREE.Line(lineGeometry, lineMaterial);
    this.cameraFrustums.add(line);
  }

  private createCameraFrustum(): THREE.Group {
    const group = new THREE.Group();

    // Create a simple camera frustum wireframe
    const size = 0.15;
    const depth = 0.25;

    const points = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(-size, -size, -depth),
      new THREE.Vector3(size, -size, -depth),
      new THREE.Vector3(size, size, -depth),
      new THREE.Vector3(-size, size, -depth),
    ];

    const indices = [
      0, 1, 0, 2, 0, 3, 0, 4,
      1, 2, 2, 3, 3, 4, 4, 1
    ];

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometry.setIndex(indices);

    const material = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.6, transparent: true });
    const frustum = new THREE.LineSegments(geometry, material);

    group.add(frustum);
    return group;
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);

    // Smoothly interpolate viewport offset
    this.viewportOffset += (this.targetViewportOffset - this.viewportOffset) * 0.05;

    this.controls.update();

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    if (this.viewportOffset > 1) {
      const offset = Math.round(this.viewportOffset);

      // Clear the entire canvas first (keeps the left side dark)
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, width, height);
      this.renderer.clear();

      // Render the 3D scene only in the right portion
      this.renderer.setScissorTest(true);
      this.renderer.setScissor(offset, 0, width - offset, height);
      this.renderer.setViewport(offset, 0, width - offset, height);
      this.camera.aspect = (width - offset) / height;
      this.camera.updateProjectionMatrix();
    } else {
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    this.renderer.render(this.scene, this.camera);
  };

  private onResize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.renderer.setSize(width, height);

    // Recalculate viewport offset if already interacted
    if (this.userInteracted) {
      this.targetViewportOffset = width * 0.35;
    }
  }

  public fadeIn(): void {
    // Smoothly increase opacity when user starts interacting
    if (this.pointCloud) {
      const material = this.pointCloud.material as THREE.PointsMaterial;
      const startOpacity = material.opacity;
      const targetOpacity = 1.0;
      const duration = 800; // milliseconds
      const startTime = Date.now();

      const animateOpacity = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        material.opacity = startOpacity + (targetOpacity - startOpacity) * eased;

        if (progress < 1) {
          requestAnimationFrame(animateOpacity);
        }
      };

      animateOpacity();
    }
  }

  /**
   * Called when the user first interacts (pointerdown/touchstart).
   * Disables auto-rotate and fades in the point cloud. Safe to call multiple times.
   */
  public startInteraction(): void {
    if (this.userInteracted) return;
    this.userInteracted = true;
    // Stop auto-rotation so the user can control the view
    this.controls.autoRotate = false;
    // Shift the 3D viewport to the right ~35% of the screen width
    this.targetViewportOffset = this.container.clientWidth * 0.35;
    this.fadeIn();
  }

  public dispose(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
    this.controls.dispose();
    this.renderer.dispose();
    if (this.pointCloud) {
      this.pointCloud.geometry.dispose();
      (this.pointCloud.material as THREE.Material).dispose();
    }
    window.removeEventListener('resize', () => this.onResize());
  }
}
