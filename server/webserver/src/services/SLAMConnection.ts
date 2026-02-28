import { io, Socket } from 'socket.io-client';
import type { SLAMUpdate, ConnectionState, DetectionPreview, DetectionPartialResult, AgentThought, AgentAction, AgentFinding, AgentState } from '../types';

/**
 * Manages WebSocket connection to SLAM server
 */
export class SLAMConnection {
  private socket: Socket | null = null;
  private serverUrl: string;
  private connectionState: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private latency = 0;
  private lastPingTime = 0;

  // Event callbacks
  private onConnectCallback?: () => void;
  private onDisconnectCallback?: () => void;
  private onUpdateCallback?: (data: SLAMUpdate) => void;
  private onStateChangeCallback?: (state: ConnectionState) => void;
  private onErrorCallback?: (error: string) => void;
  private onDetectionPreviewCallback?: (data: DetectionPreview) => void;
  private onDetectionPartialCallback?: (data: DetectionPartialResult) => void;

  // Agent callbacks
  private onAgentThoughtCallback?: (data: AgentThought) => void;
  private onAgentActionCallback?: (data: AgentAction) => void;
  private onAgentFindingCallback?: (data: AgentFinding) => void;
  private onAgentStateCallback?: (data: AgentState) => void;

  constructor(serverUrl: string) {
    this.serverUrl = serverUrl;
  }

  /**
   * Connect to SLAM server
   */
  connect(): void {
    console.log('🔌 Connecting to:', this.serverUrl);
    this.setConnectionState('connecting');

    this.socket = io(this.serverUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: this.maxReconnectAttempts,
    });

    this.setupEventHandlers();
  }

  /**
   * Disconnect from server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.setConnectionState('disconnected');
  }

  /**
   * Request current global map from server
   */
  requestGlobalMap(): void {
    if (this.socket && this.isConnected()) {
      console.log('📥 Requesting global map...');
      this.lastPingTime = Date.now();
      this.socket.emit('get_global_map');
    }
  }

  /**
   * Request server reset
   */
  async resetServer(): Promise<void> {
    try {
      const response = await fetch(`${this.serverUrl}/reset`, {
        method: 'POST',
      });

      if (response.ok) {
        console.log('✅ Server reset successful');
      } else {
        throw new Error('Reset failed');
      }
    } catch (error) {
      console.error('❌ Reset error:', error);
      this.onErrorCallback?.('Failed to reset server');
      throw error;
    }
  }

  /**
   * Start demo feeder for a selected local video on the server.
   */
  async startDemo(videoId: string, fps = 10): Promise<void> {
    const response = await fetch(`${this.serverUrl}/api/demo/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ video_id: videoId, fps }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const message = body.error || `Failed to start demo (${response.status})`;
      throw new Error(message);
    }
  }

  /**
   * Stop active demo feeder if one is running.
   */
  async stopDemo(): Promise<void> {
    await fetch(`${this.serverUrl}/api/demo/stop`, {
      method: 'POST',
    });
  }

  /**
   * Ask backend to stop SLAM processing loop.
   */
  stopSLAM(): void {
    if (this.socket && this.isConnected()) {
      this.socket.emit('stop_slam');
    }
  }

  /**
   * Check if currently connected
   */
  isConnected(): boolean {
    return this.connectionState === 'connected' && this.socket?.connected === true;
  }

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Get current latency in ms
   */
  getLatency(): number {
    return this.latency;
  }

  // Event handler setters
  onConnect(callback: () => void): void {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback: () => void): void {
    this.onDisconnectCallback = callback;
  }

  onUpdate(callback: (data: SLAMUpdate) => void): void {
    this.onUpdateCallback = callback;
  }

  onStateChange(callback: (state: ConnectionState) => void): void {
    this.onStateChangeCallback = callback;
  }

  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Send active detection queries to the server for real-time per-submap detection.
   */
  setDetectionQueries(queries: string[]): void {
    if (this.socket && this.isConnected()) {
      console.log('🎯 Setting detection queries:', queries);
      this.socket.emit('set_detection_queries', { queries });
    }
  }

  /**
   * Request on-demand keyframe + SAM 3 mask preview for a specific detection.
   */
  getDetectionPreview(submapId: number, frameIdx: number, query: string): void {
    if (this.socket && this.isConnected()) {
      console.log(`📸 Requesting preview: '${query}' submap ${submapId} frame ${frameIdx}`);
      this.socket.emit('get_detection_preview', {
        submap_id: submapId,
        frame_idx: frameIdx,
        query,
      });
    }
  }

  onDetectionPreview(callback: (data: DetectionPreview) => void): void {
    this.onDetectionPreviewCallback = callback;
  }

  onDetectionPartial(callback: (data: DetectionPartialResult) => void): void {
    this.onDetectionPartialCallback = callback;
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('✅ Connected to SLAM server!');
      this.setConnectionState('connected');
      this.reconnectAttempts = 0;
      this.onConnectCallback?.();

      // Auto-request global map on connect
      setTimeout(() => this.requestGlobalMap(), 100);
    });

    this.socket.on('disconnect', () => {
      console.log('❌ Disconnected from server');
      this.setConnectionState('disconnected');
      this.onDisconnectCallback?.();
    });

    this.socket.on('connected', (data) => {
      console.log('✅ Server ready:', data);
    });

    this.socket.on('slam_update', (data: SLAMUpdate) => {
      this.calculateLatency();
      console.log('📦 SLAM update received');
      this.onUpdateCallback?.(data);
    });

    this.socket.on('global_map', (data: SLAMUpdate) => {
      this.calculateLatency();
      console.log('🗺️  Global map received');
      this.onUpdateCallback?.(data);
    });

    this.socket.on('slam_reset', () => {
      console.log('🔄 Server was reset');
      this.onErrorCallback?.('Server was reset');
    });

    this.socket.on('detection_preview', (data: DetectionPreview) => {
      console.log('📸 Detection preview received');
      this.onDetectionPreviewCallback?.(data);
    });

    this.socket.on('detection_partial', (data: DetectionPartialResult) => {
      console.log(`🔍 Detection partial: ${data.detections.length} detections, final=${data.is_final}`);
      this.onDetectionPartialCallback?.(data);
    });

    // Agent events
    this.socket.on('agent_thought', (data: AgentThought) => {
      this.onAgentThoughtCallback?.(data);
    });

    this.socket.on('agent_action', (data: AgentAction) => {
      this.onAgentActionCallback?.(data);
    });

    this.socket.on('agent_finding', (data: AgentFinding) => {
      this.onAgentFindingCallback?.(data);
    });

    this.socket.on('agent_state', (data: AgentState) => {
      this.onAgentStateCallback?.(data);
    });

    this.socket.on('connect_error', (error) => {
      console.error('❌ Connection error:', error);
      this.reconnectAttempts++;

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.setConnectionState('error');
        this.onErrorCallback?.('Connection failed after multiple attempts');
      }
    });

    this.socket.on('error', (error) => {
      console.error('❌ Socket error:', error);
      this.onErrorCallback?.(error.toString());
    });
  }

  // Agent event callback setters
  onAgentThought(callback: (data: AgentThought) => void): void {
    this.onAgentThoughtCallback = callback;
  }

  onAgentAction(callback: (data: AgentAction) => void): void {
    this.onAgentActionCallback = callback;
  }

  onAgentFinding(callback: (data: AgentFinding) => void): void {
    this.onAgentFindingCallback = callback;
  }

  onAgentState(callback: (data: AgentState) => void): void {
    this.onAgentStateCallback = callback;
  }

  // Agent emit methods
  sendAgentChat(message: string): void {
    if (this.socket && this.isConnected()) {
      this.socket.emit('agent_chat', { message });
    }
  }

  setAgentGoal(goal: string): void {
    if (this.socket && this.isConnected()) {
      this.socket.emit('agent_set_goal', { goal });
    }
  }

  toggleAgent(enabled: boolean): void {
    if (this.socket && this.isConnected()) {
      this.socket.emit('agent_toggle', { enabled });
    }
  }

  requestAgentState(): void {
    if (this.socket && this.isConnected()) {
      this.socket.emit('get_agent_state');
    }
  }

  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.onStateChangeCallback?.(state);
  }

  private calculateLatency(): void {
    if (this.lastPingTime > 0) {
      this.latency = Date.now() - this.lastPingTime;
      this.lastPingTime = 0;
    }
  }
}
