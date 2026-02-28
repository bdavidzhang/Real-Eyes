import { SLAMConnection } from './services/SLAMConnection';
import { SceneManager } from './services/SceneManager';
import { UIManager } from './components/UIManager';
import type { SLAMUpdate } from './types';
import './styles/dashboard.css';

/**
 * Main Application Class
 */
class SLAMViewerApp {
  private connection: SLAMConnection;
  private sceneManager: SceneManager;
  private uiManager: UIManager;

  constructor() {
    console.log('🎨 VGGT-SLAM Viewer initializing...');

    // Initialize managers
    const container = document.getElementById('app');
    if (!container) {
      throw new Error('App container not found');
    }

    this.sceneManager = new SceneManager(container);
    this.uiManager = new UIManager();
    const serverUrl = import.meta.env.VITE_SERVER_URL || `https://${window.location.hostname}:5000`;
    this.connection = new SLAMConnection(serverUrl);

    this.setupConnections();
    this.setupEventHandlers();
    this.loadIncomingTrackingPlan();

    console.log('✅ VGGT-SLAM Viewer ready!');
  }

  /**
   * Setup connection event handlers
   */
  private setupConnections(): void {
    // Connection state changes
    this.connection.onStateChange((state) => {
      const messages = {
        disconnected: 'Disconnected',
        connecting: 'Connecting...',
        connected: 'Connected',
        error: 'Connection Error',
      };

      this.uiManager.updateStatus(
        state,
        messages[state],
        state === 'connected' ? this.connection.getLatency() : undefined
      );
    });

    // Connection established
    this.connection.onConnect(() => {
      console.log('✅ Connected to SLAM server');
      this.uiManager.showNotification('Connected to SLAM server', 'success');
    });

    // Disconnection
    this.connection.onDisconnect(() => {
      console.log('❌ Disconnected from server');
      this.uiManager.showNotification('Disconnected from server', 'error');
    });

    // SLAM data updates
    this.connection.onUpdate((data: SLAMUpdate) => {
      this.handleSLAMUpdate(data);
    });

    // Errors
    this.connection.onError((error: string) => {
      console.error('Connection error:', error);
      this.uiManager.showNotification(error, 'error');
    });

    // Detection preview response
    this.connection.onDetectionPreview((data) => {
      this.uiManager.showDetectionPreview(data);
    });
  }

  /**
   * Setup UI event handlers
   */
  private setupEventHandlers(): void {
    // Connect button
    this.uiManager.onConnect(() => {
      if (!this.connection.isConnected()) {
        this.connection.connect();
      }
    });

    // Disconnect button
    this.uiManager.onDisconnect(() => {
      this.connection.disconnect();
    });

    // Reset button
    this.uiManager.onReset(async () => {
      try {
        this.uiManager.showNotification('Resetting SLAM server...', 'info');
        await this.connection.resetServer();
        this.sceneManager.clearScene();
        this.sceneManager.clearDetections();
        this.uiManager.updateDetectionResults([]);
        this.uiManager.updateStats({
          frames: 0,
          submaps: 0,
          loops: 0,
          points: 0,
          cameras: 0,
          fps: 0,
        });
        this.uiManager.showNotification('SLAM server reset complete', 'success');
      } catch (error) {
        this.uiManager.showNotification('Reset failed', 'error');
      }
    });

    // Reset view button
    this.uiManager.onResetView(() => {
      this.sceneManager.resetView();
      this.uiManager.showNotification('View reset', 'info');
    });

    // Toggle cameras
    this.uiManager.onToggleCameras(() => {
      const config = this.sceneManager.getConfig();
      const newConfig = { showCameras: !config.showCameras };
      this.sceneManager.updateConfig(newConfig);
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });

    // Toggle grid
    this.uiManager.onToggleGrid(() => {
      const config = this.sceneManager.getConfig();
      const newConfig = { showGrid: !config.showGrid };
      this.sceneManager.updateConfig(newConfig);
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });

    // Toggle points
    this.uiManager.onTogglePoints(() => {
      const config = this.sceneManager.getConfig();
      const newConfig = { showPoints: !config.showPoints };
      this.sceneManager.updateConfig(newConfig);
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });

    // Flip Y-axis
    this.uiManager.onFlipY(() => {
      const config = this.sceneManager.getConfig();
      const newConfig = { flipY: !config.flipY };
      this.sceneManager.updateConfig(newConfig);
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });

    // Follow camera
    this.uiManager.onFollowCamera(() => {
      const config = this.sceneManager.getConfig();
      const newConfig = { followCamera: !config.followCamera };
      this.sceneManager.updateConfig(newConfig);
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });

    // Point size change
    this.uiManager.onPointSizeChange((size: number) => {
      this.sceneManager.updateConfig({ pointSize: size });
    });

    // Download point cloud PLY
    this.uiManager.onDownloadSplat(() => {
      const ok = this.sceneManager.exportPointCloudPLY();
      if (ok) {
        this.uiManager.showNotification('Point cloud PLY downloaded', 'success');
      } else {
        this.uiManager.showNotification('No point cloud data to export', 'error');
      }
    });
    // Detection panel: set targets
    this.uiManager.onSetTargets((queries: string[]) => {
      this.connection.setDetectionQueries(queries);
      this.uiManager.updateDetectionQueries(queries);
      this.uiManager.showNotification(`Detecting: ${queries.join(', ')}`, 'info');
    });

    // Detection panel: clear targets
    this.uiManager.onClearTargets(() => {
      this.connection.setDetectionQueries([]);
      this.sceneManager.clearDetections();
      this.uiManager.updateDetectionQueries([]);
      this.uiManager.updateDetectionResults([]);
    });

    // Detection card click: request preview images
    this.uiManager.onDetectionClick((det) => {
      if (det.matched_submap != null && det.matched_frame != null) {
        this.connection.getDetectionPreview(det.matched_submap, det.matched_frame, det.query);
      }
    });

    // Toggle detection labels
    this.uiManager.onToggleDetLabels(() => {
      const config = this.sceneManager.getConfig();
      this.sceneManager.updateConfig({ showDetectionLabels: !config.showDetectionLabels });
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });

    // Toggle detection boxes
    this.uiManager.onToggleDetBoxes(() => {
      const config = this.sceneManager.getConfig();
      this.sceneManager.updateConfig({ showDetectionBoxes: !config.showDetectionBoxes });
      this.uiManager.updateControlStates(this.sceneManager.getConfig());
    });
  }

  /**
   * Handle SLAM update from server
   */
  private handleSLAMUpdate(data: SLAMUpdate): void {
    // Update visualization
    this.sceneManager.updateVisualization(data);

    // Update stats
    this.uiManager.updateStats({
      frames: data.frame_id,
      submaps: data.num_submaps,
      loops: data.num_loops,
      points: data.n_points,
      cameras: data.n_cameras,
      fps: 0, // FPS is calculated in UIManager
    });

    // Update detection overlays if detections are present
    if (data.detections) {
      this.sceneManager.updateDetections(data.detections);
      this.uiManager.updateDetectionResults(data.detections);
    }
    if (data.active_queries) {
      this.uiManager.updateDetectionQueries(data.active_queries);
    }
  }

  /**
   * Load tracking plan from plan page if available
   */
  private loadIncomingTrackingPlan(): void {
    const params = new URLSearchParams(window.location.search);
    const source = params.get('source');

    if (source === 'plan') {
      // Read data from sessionStorage
      const objectsJson = sessionStorage.getItem('trackedObjects');
      const waypoints = sessionStorage.getItem('waypointsEnabled') === 'true';
      const pathfinding = sessionStorage.getItem('pathfindingEnabled') === 'true';

      if (objectsJson) {
        const objects: string[] = JSON.parse(objectsJson);
        console.log('📦 Loaded tracking plan:', { objects, waypoints, pathfinding });

        // Auto-populate detection input
        this.uiManager.populateDetectionInput(objects);

        // Auto-connect and set detection queries
        setTimeout(() => {
          this.connection.connect();

          setTimeout(() => {
            this.connection.setDetectionQueries(objects);
            this.uiManager.updateDetectionQueries(objects);
            console.log('✅ Detection targets set:', objects);
          }, 500);
        }, 100);

        // Clear sessionStorage
        sessionStorage.removeItem('trackedObjects');
        sessionStorage.removeItem('waypointsEnabled');
        sessionStorage.removeItem('pathfindingEnabled');
      }
    }
  }

  /**
   * Cleanup on app destroy
   */
  destroy(): void {
    this.connection.disconnect();
    this.sceneManager.dispose();
  }
}

// Initialize app when DOM is ready
let app: SLAMViewerApp | null = null;

window.addEventListener('DOMContentLoaded', () => {
  app = new SLAMViewerApp();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  app?.destroy();
});
