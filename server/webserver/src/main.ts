import { SLAMConnection } from './services/SLAMConnection';
import { SceneManager } from './services/SceneManager';
import { UIManager } from './components/UIManager';
import { AgentPanel } from './components/AgentPanel';
import type { SLAMUpdate, AgentUICommand, AgentUIResult, DetectionResult } from './types';
import './styles/dashboard.css';
import './styles/agent.css';

/**
 * Main Application Class
 */
class SLAMViewerApp {
  private connection: SLAMConnection;
  private sceneManager: SceneManager;
  private uiManager: UIManager;
  private agentPanel: AgentPanel;
  private totalPoints = 0;
  private totalCameras = 0;
  private trackingSource: 'live' | 'demo' = 'live';
  private demoVideoId: string | null = null;
  private latestDetections: DetectionResult[] = [];

  constructor() {
    console.log('🎨 VGGT-SLAM Viewer initializing...');

    // Initialize managers
    const container = document.getElementById('app');
    if (!container) {
      throw new Error('App container not found');
    }

    this.sceneManager = new SceneManager(container);
    this.uiManager = new UIManager();
    this.agentPanel = new AgentPanel();
    const serverUrl = import.meta.env.VITE_SERVER_URL || window.location.origin;
    this.connection = new SLAMConnection(serverUrl);

    this.setupConnections();
    this.setupEventHandlers();
    this.setupAgentHandlers();
    void this.loadIncomingTrackingPlan();

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

    // Detection preview response — cache thumbnail + show modal if user is waiting
    this.connection.onDetectionPreview((data) => {
      this.uiManager.receivePreviewData(data);
    });

    // Progressive detection results — show everything the server finds.
    // Chips control what's actively being searched; removal is the only reason to hide boxes.
    this.connection.onDetectionPartial((data) => {
      console.debug(
        '[detection_partial]',
        'server_active:', data.active_queries,
        '| dets:', data.detections.length,
        '| is_final:', data.is_final,
      );
      this.latestDetections = data.detections;
      this.sceneManager.updateDetections(data.detections);
      this.uiManager.updateDetectionResults(data.detections);
      // chips stay as-is — do NOT call updateDetectionQueries
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

    const stopDemoBtn = document.getElementById('stopDemoBtn') as HTMLButtonElement | null;
    if (stopDemoBtn) {
      stopDemoBtn.addEventListener('click', async () => {
        try {
          await this.connection.stopDemo();
          this.connection.stopSLAM();
          this.uiManager.showNotification('Demo stopped. Current map is kept.', 'info');
        } catch (error) {
          this.uiManager.showNotification('Failed to stop demo', 'error');
        }
      });
    }

    // Reset button
    this.uiManager.onReset(async () => {
      try {
        this.uiManager.showNotification('Resetting SLAM server...', 'info');
        await this.connection.resetServer();
        this.sceneManager.clearScene();
        this.sceneManager.clearDetections();
        this.totalPoints = 0;
        this.totalCameras = 0;
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
      const oldQueries = this.uiManager.queries;
      const removed = oldQueries.filter((q) => !queries.includes(q));
      const added = queries.filter((q) => !oldQueries.includes(q));
      console.debug(
        '[onSetTargets] old:', oldQueries,
        '| new:', queries,
        '| added:', added,
        '| removed:', removed,
      );
      this.connection.setDetectionQueries(queries);
      this.uiManager.updateDetectionQueries(queries);
      if (removed.length > 0) {
        // User explicitly removed these chips — hide their boxes immediately
        const removedSet = new Set(removed);
        const filtered = this.latestDetections.filter((d) => !removedSet.has(d.query));
        console.debug('[onSetTargets] hiding removed:', removed, '|', this.latestDetections.length, '->', filtered.length);
        this.latestDetections = filtered;
        this.sceneManager.updateDetections(filtered);
        this.uiManager.updateDetectionResults(filtered);
      }
      if (queries.length > 0) {
        this.uiManager.showNotification(`Detecting: ${queries.join(', ')}`, 'info');
      }
    });

    // Detection panel: clear targets
    this.uiManager.onClearTargets(() => {
      console.debug('[onClearTargets] clearing all queries. was:', this.uiManager.queries);
      this.latestDetections = [];
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
   * Setup agent panel event wiring
   */
  private setupAgentHandlers(): void {
    // Agent events from server → agent panel
    this.connection.onAgentThought((data) => {
      this.agentPanel.handleThought(data);
    });

    this.connection.onAgentAction((data) => {
      this.agentPanel.handleAction(data);
    });

    this.connection.onAgentFinding((data) => {
      this.agentPanel.handleFinding(data);
      this.uiManager.showNotification(`Found: ${data.query}`, 'success');
    });

    this.connection.onAgentState((data) => {
      this.agentPanel.handleState(data);
    });

    this.connection.onAgentToolEvent((data) => {
      this.agentPanel.handleToolEvent(data);
    });

    this.connection.onAgentTaskEvent((data) => {
      this.agentPanel.handleTaskEvent(data);
    });

    this.connection.onAgentJobEvent((data) => {
      this.agentPanel.handleJobEvent(data);
    });

    this.connection.onAgentUICommand((cmd) => {
      this.agentPanel.handleUICommand(cmd);
      void this.handleAgentUICommand(cmd);
    });

    // Agent panel → server
    this.agentPanel.onChatSend((message) => {
      this.connection.sendAgentChat(message);
    });

    this.agentPanel.onAgentToggle((enabled) => {
      this.connection.toggleAgent(enabled);
    });

    // Auto-fetch preview thumbnails when detection cards are rendered
    this.uiManager.onAutoFetchPreview((det) => {
      if (det.matched_submap != null && det.matched_frame != null) {
        this.connection.getDetectionPreview(det.matched_submap, det.matched_frame, det.query);
      }
    });

    // Request agent state on connect
    this.connection.onConnect(() => {
      setTimeout(() => this.connection.requestAgentState(), 200);
    });
  }

  private async handleAgentUICommand(cmd: AgentUICommand): Promise<void> {
    const ack = (status: 'ok' | 'error' | 'ignored' | 'timeout', result?: Record<string, unknown>, error?: string) => {
      const uiResult: AgentUIResult = {
        id: cmd.id,
        status,
        result,
        error,
      };
      this.connection.sendAgentUIResult(uiResult);
      this.agentPanel.handleUIResult(uiResult, cmd.name);
    };

    try {
      switch (cmd.name) {
        case 'set_detection_queries': {
          const raw = cmd.args.queries;
          if (!Array.isArray(raw)) {
            ack('ignored', undefined, 'queries must be an array');
            return;
          }
          const queries = raw.map((q) => String(q).trim().toLowerCase()).filter((q) => q.length > 0);
          this.connection.setDetectionQueries(queries);
          this.uiManager.updateDetectionQueries(queries);
          ack('ok', { query_count: queries.length });
          return;
        }
        case 'add_detection_query': {
          const query = String(cmd.args.query ?? '').trim().toLowerCase();
          if (!query) {
            ack('ignored', undefined, 'query is required');
            return;
          }
          const updated = [...new Set([...this.uiManager.queries, query])];
          console.debug('[agent add_detection_query] query:', query, '| new chips:', updated);
          this.uiManager.updateDetectionQueries(updated);
          ack('ok', { query });
          return;
        }
        case 'remove_detection_query': {
          const query = String(cmd.args.query ?? '').trim().toLowerCase();
          if (!query) {
            ack('ignored', undefined, 'query is required');
            return;
          }
          const updated = this.uiManager.queries.filter((q) => q !== query);
          console.debug('[agent remove_detection_query] query:', query, '| remaining chips:', updated);
          this.uiManager.updateDetectionQueries(updated);
          const filtered = this.latestDetections.filter((d) => d.query !== query);
          this.latestDetections = filtered;
          this.sceneManager.updateDetections(filtered);
          this.uiManager.updateDetectionResults(filtered);
          ack('ok', { query });
          return;
        }
        case 'focus_detection': {
          const center = cmd.args.center;
          if (Array.isArray(center) && center.length === 3) {
            const xyz = center.map((v) => Number(v));
            this.sceneManager.focusOnPoint([xyz[0], xyz[1], xyz[2]]);
            ack('ok', { centered: true });
            return;
          }
          const query = typeof cmd.args.query === 'string' ? cmd.args.query : null;
          if (query) {
            const focused = this.sceneManager.focusOnDetectionQuery(query);
            ack(focused ? 'ok' : 'ignored', { query });
            return;
          }
          ack('ignored', undefined, 'missing center or query');
          return;
        }
        case 'show_waypoint': {
          const id = String(cmd.args.waypoint_id ?? 'agent-waypoint');
          const pos = cmd.args.position;
          if (!Array.isArray(pos) || pos.length !== 3) {
            ack('ignored', undefined, 'invalid waypoint position');
            return;
          }
          this.sceneManager.showWaypoint(
            id,
            [Number(pos[0]), Number(pos[1]), Number(pos[2])],
            typeof cmd.args.label === 'string' ? cmd.args.label : undefined,
          );
          ack('ok', { waypoint_id: id });
          return;
        }
        case 'show_path': {
          const rawPoints = cmd.args.points;
          if (!Array.isArray(rawPoints)) {
            ack('ignored', undefined, 'invalid path points');
            return;
          }
          const points = rawPoints
            .filter((pt): pt is unknown[] => Array.isArray(pt) && pt.length === 3)
            .map((pt) => [Number(pt[0]), Number(pt[1]), Number(pt[2])] as [number, number, number]);
          this.sceneManager.showPath(
            points,
            typeof cmd.args.path_id === 'string' ? cmd.args.path_id : 'agent-path',
          );
          ack('ok', { points: points.length });
          return;
        }
        case 'show_toast': {
          const msg = String(cmd.args.message ?? '').trim();
          if (!msg) {
            ack('ignored', undefined, 'empty message');
            return;
          }
          const level = cmd.args.level === 'success' || cmd.args.level === 'error' ? cmd.args.level : 'info';
          this.uiManager.showNotification(msg, level);
          ack('ok');
          return;
        }
        case 'open_detection_preview': {
          const submap = Number(cmd.args.submap_id);
          const frame = Number(cmd.args.frame_idx);
          const query = String(cmd.args.query ?? '').trim();
          if (!Number.isFinite(submap) || !Number.isFinite(frame) || !query) {
            ack('ignored', undefined, 'invalid preview arguments');
            return;
          }
          this.connection.getDetectionPreview(submap, frame, query);
          ack('ok', { submap_id: submap, frame_idx: frame, query });
          return;
        }
        default:
          ack('ignored', undefined, 'unknown command');
      }
    } catch (error) {
      ack('error', undefined, (error as Error).message);
    }
  }

  /**
   * Handle SLAM update from server
   */
  private handleSLAMUpdate(data: SLAMUpdate): void {
    // Update visualization
    this.sceneManager.updateVisualization(data);

    // Track accumulated totals for stats
    if (!data.type || data.type === 'full') {
      this.totalPoints = data.n_points;
      this.totalCameras = data.n_cameras;
    } else {
      this.totalPoints += data.n_points;
      this.totalCameras += data.n_cameras;
    }

    // Update stats
    this.uiManager.updateStats({
      frames: data.frame_id,
      submaps: data.num_submaps,
      loops: data.num_loops,
      points: this.totalPoints,
      cameras: this.totalCameras,
      fps: 0, // FPS is calculated in UIManager
    });

    if (data.detections) {
      console.debug('[slam_update] dets:', data.detections.length, '| server_active:', data.active_queries);
      this.latestDetections = data.detections;
      this.sceneManager.updateDetections(data.detections);
      this.uiManager.updateDetectionResults(data.detections);
    }
  }

  /**
   * Load tracking plan from plan page if available
   */
  private async loadIncomingTrackingPlan(): Promise<void> {
    const params = new URLSearchParams(window.location.search);
    const source = params.get('source');
    const modeParam = params.get('mode');
    const videoParam = params.get('video_id');

    const sessionTrackingSource = sessionStorage.getItem('trackingSource');
    this.trackingSource = (sessionTrackingSource === 'demo' || modeParam === 'demo')
      ? 'demo'
      : 'live';
    this.demoVideoId = sessionStorage.getItem('demoVideoId') || videoParam;

    const viewerModeBadge = document.getElementById('viewerModeBadge') as HTMLDivElement | null;
    const senderLink = document.querySelector('.nav-link[href^="/sender.html"]') as HTMLAnchorElement | null;
    const stopDemoBtn = document.getElementById('stopDemoBtn') as HTMLButtonElement | null;

    if (senderLink) {
      if (this.trackingSource === 'demo' && this.demoVideoId) {
        senderLink.href = `/sender.html?mode=demo&video_id=${encodeURIComponent(this.demoVideoId)}`;
      } else {
        senderLink.href = '/sender.html';
      }
    }
    if (viewerModeBadge) {
      if (this.trackingSource === 'demo') {
        viewerModeBadge.style.display = 'inline-flex';
        viewerModeBadge.textContent = this.demoVideoId
          ? `DEMO MODE - ${this.demoVideoId}`
          : 'DEMO MODE';
      } else {
        viewerModeBadge.style.display = 'none';
      }
    }
    if (stopDemoBtn) {
      stopDemoBtn.style.display = this.trackingSource === 'demo' ? 'inline-flex' : 'none';
    }

    if (source === 'plan') {
      // Read data from sessionStorage
      const objectsJson = sessionStorage.getItem('trackedObjects');
      const waypoints = sessionStorage.getItem('waypointsEnabled') === 'true';
      const pathfinding = sessionStorage.getItem('pathfindingEnabled') === 'true';

      if (objectsJson) {
        const objects: string[] = JSON.parse(objectsJson);
        console.log('📦 Loaded tracking plan:', {
          objects,
          waypoints,
          pathfinding,
          trackingSource: this.trackingSource,
          demoVideoId: this.demoVideoId,
        });

        // Auto-populate detection input
        this.uiManager.populateDetectionInput(objects);

        // Auto-connect and set detection queries
        setTimeout(async () => {
          this.connection.connect();

          try {
            if (this.trackingSource === 'demo' && this.demoVideoId) {
              await this.connection.startDemo(this.demoVideoId, 10);
              this.uiManager.showNotification(`Demo mode: ${this.demoVideoId}`, 'info');
            } else {
              await this.connection.stopDemo();
            }
          } catch (error) {
            console.error('Demo mode request failed:', error);
            this.uiManager.showNotification('Failed to configure demo mode', 'error');
          }

          setTimeout(() => {
            this.connection.setDetectionQueries(objects);
            this.uiManager.updateDetectionQueries(objects);
            console.log('✅ Detection targets set:', objects);

            const userPrompt = sessionStorage.getItem('userPrompt') ?? '';
            if (userPrompt || objects.length > 0) {
              this.connection.setAgentGoal(userPrompt, objects);
              console.log('✅ Agent context set:', { goal: userPrompt, initial_queries: objects });
            }
          }, 500);
        }, 100);

        // Clear sessionStorage
        sessionStorage.removeItem('trackedObjects');
        sessionStorage.removeItem('waypointsEnabled');
        sessionStorage.removeItem('pathfindingEnabled');
        sessionStorage.removeItem('trackingSource');
        sessionStorage.removeItem('demoVideoId');
        sessionStorage.removeItem('userPrompt');
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
