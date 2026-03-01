import type { SLAMStats, ConnectionState, ControlsConfig, DetectionResult, DetectionPreview } from '../types';

// Color palette (must match SceneManager BOX_COLORS)
const BOX_COLORS = [
  '#4da6ff', '#ff6b6b', '#51cf66', '#fcc419', '#cc5de8',
  '#ff922b', '#20c997', '#f06595', '#5c7cfa', '#94d82d',
];

/**
 * Manages all UI components and interactions
 */
export class UIManager {
  // DOM elements
  private statusElement: HTMLElement;
  private statusDot: HTMLElement;
  private statusText: HTMLElement;
  private latencyElement: HTMLElement;

  // Info panel elements
  private frameElement: HTMLElement;
  private submapsElement: HTMLElement;
  private loopsElement: HTMLElement;
  private pointsElement: HTMLElement;
  private camerasElement: HTMLElement;
  private fpsElement: HTMLElement;

  // Control elements
  private connectBtn: HTMLButtonElement;
  private disconnectBtn: HTMLButtonElement;
  private resetBtn: HTMLButtonElement;
  private resetViewBtn: HTMLButtonElement;
  private toggleCamsBtn: HTMLButtonElement;
  private toggleGridBtn: HTMLButtonElement;
  private togglePointsBtn: HTMLButtonElement;
  private flipYBtn: HTMLButtonElement;
  private followCamBtn: HTMLButtonElement;
  private pointSizeSlider: HTMLInputElement;
  private toggleDetLabelsBtn: HTMLButtonElement;
  private toggleDetBoxesBtn: HTMLButtonElement;

  // Settings panel
  private settingsPanel: HTMLElement;
  private settingsBtn: HTMLButtonElement;
  private downloadSplatBtn: HTMLButtonElement;
  private settingsOpen = false;

  // Detection input (now inside agent panel)
  private detectionInput: HTMLInputElement;
  private setTargetsBtn: HTMLButtonElement;
  private clearTargetsBtn: HTMLButtonElement;
  private activeQueriesList: HTMLElement;
  private detectionResultsList: HTMLElement;
  private detectionCountEl: HTMLElement;
  private queryColorMap = new Map<string, string>();
  private currentQueries: string[] = [];

  // Preview cache for instant card thumbnails
  private previewCache = new Map<string, DetectionPreview>();
  private pendingPreviews = new Set<string>();
  private activePreviewKey: string | null = null;
  private onAutoFetchPreviewCallback?: (det: DetectionResult) => void;

  // Preview modal
  private previewOverlay: HTMLElement;
  private previewCloseBtn: HTMLButtonElement;
  private previewTitle: HTMLElement;
  private previewKeyframe: HTMLImageElement;
  private previewMask: HTMLImageElement;
  private previewMeta: HTMLElement;

  // FPS tracking
  private lastUpdate = Date.now();
  private frameCount = 0;

  // Notification queue (prevents DOM churn during bursty events)
  private notificationQueue: Array<{ message: string; type: 'info' | 'success' | 'error' }> = [];
  private activeNotifications: HTMLElement[] = [];
  private notificationFlushScheduled = false;
  private maxQueuedNotifications = 64;
  private maxActiveNotifications = 5;

  // Callbacks
  private onConnectCallback?: () => void;
  private onDisconnectCallback?: () => void;
  private onResetCallback?: () => void;
  private onResetViewCallback?: () => void;
  private onToggleCamerasCallback?: () => void;
  private onToggleGridCallback?: () => void;
  private onTogglePointsCallback?: () => void;
  private onFlipYCallback?: () => void;
  private onFollowCameraCallback?: () => void;
  private onPointSizeChangeCallback?: (size: number) => void;
  private onDownloadSplatCallback?: () => void;
  private onSetTargetsCallback?: (queries: string[]) => void;
  private onClearTargetsCallback?: () => void;
  private onDetectionClickCallback?: (det: DetectionResult) => void;
  private onToggleDetLabelsCallback?: () => void;
  private onToggleDetBoxesCallback?: () => void;

  constructor() {
    // Get status elements
    this.statusElement = this.getElement('status');
    this.statusDot = this.getElement('status-dot');
    this.statusText = this.getElement('status-text');
    this.latencyElement = this.getElement('latency');

    // Get info panel elements
    this.frameElement = this.getElement('frame');
    this.submapsElement = this.getElement('submaps');
    this.loopsElement = this.getElement('loops');
    this.pointsElement = this.getElement('points');
    this.camerasElement = this.getElement('cameras');
    this.fpsElement = this.getElement('fps');

    // Get control elements
    this.connectBtn = this.getElement('connectBtn') as HTMLButtonElement;
    this.disconnectBtn = this.getElement('disconnectBtn') as HTMLButtonElement;
    this.resetBtn = this.getElement('resetBtn') as HTMLButtonElement;
    this.resetViewBtn = this.getElement('resetViewBtn') as HTMLButtonElement;
    this.toggleCamsBtn = this.getElement('toggleCamsBtn') as HTMLButtonElement;
    this.toggleGridBtn = this.getElement('toggleGridBtn') as HTMLButtonElement;
    this.togglePointsBtn = this.getElement('togglePointsBtn') as HTMLButtonElement;
    this.flipYBtn = this.getElement('flipYBtn') as HTMLButtonElement;
    this.followCamBtn = this.getElement('followCamBtn') as HTMLButtonElement;
    this.pointSizeSlider = this.getElement('pointSizeSlider') as HTMLInputElement;
    this.toggleDetLabelsBtn = this.getElement('toggleDetLabelsBtn') as HTMLButtonElement;
    this.toggleDetBoxesBtn = this.getElement('toggleDetBoxesBtn') as HTMLButtonElement;

    // Get settings elements
    this.settingsPanel = this.getElement('settings-panel');
    this.settingsBtn = this.getElement('settingsBtn') as HTMLButtonElement;
    this.downloadSplatBtn = this.getElement('downloadSplatBtn') as HTMLButtonElement;

    // Get detection elements (now inside agent panel)
    this.detectionInput = this.getElement('detectionInput') as HTMLInputElement;
    this.setTargetsBtn = this.getElement('setTargetsBtn') as HTMLButtonElement;
    this.clearTargetsBtn = this.getElement('clearTargetsBtn') as HTMLButtonElement;
    this.activeQueriesList = this.getElement('activeQueriesList');
    this.detectionResultsList = this.getElement('detectionResultsList');
    this.detectionCountEl = this.getElement('detectionCount');

    // Preview modal
    this.previewOverlay = this.getElement('detection-preview-overlay');
    this.previewCloseBtn = this.getElement('closePreviewBtn') as HTMLButtonElement;
    this.previewTitle = this.getElement('previewTitle');
    this.previewKeyframe = this.getElement('previewKeyframe') as HTMLImageElement;
    this.previewMask = this.getElement('previewMask') as HTMLImageElement;
    this.previewMeta = this.getElement('previewMeta');

    this.setupEventListeners();
  }

  private getElement(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) {
      throw new Error(`Element with id '${id}' not found`);
    }
    return element;
  }

  private setupEventListeners(): void {
    this.connectBtn.addEventListener('click', () => {
      this.onConnectCallback?.();
    });

    this.disconnectBtn.addEventListener('click', () => {
      this.onDisconnectCallback?.();
    });

    this.resetBtn.addEventListener('click', () => {
      if (confirm('Reset SLAM data? This will clear the current map.')) {
        this.onResetCallback?.();
      }
    });

    this.resetViewBtn.addEventListener('click', () => {
      this.onResetViewCallback?.();
    });

    this.toggleCamsBtn.addEventListener('click', () => {
      this.onToggleCamerasCallback?.();
    });

    this.toggleGridBtn.addEventListener('click', () => {
      this.onToggleGridCallback?.();
    });

    this.togglePointsBtn.addEventListener('click', () => {
      this.onTogglePointsCallback?.();
    });

    this.flipYBtn.addEventListener('click', () => {
      this.onFlipYCallback?.();
    });

    this.followCamBtn.addEventListener('click', () => {
      this.onFollowCameraCallback?.();
    });

    this.pointSizeSlider.addEventListener('input', (e) => {
      const value = parseFloat((e.target as HTMLInputElement).value);
      // Update the displayed value
      const sliderValue = this.pointSizeSlider.nextElementSibling as HTMLElement;
      if (sliderValue) {
        sliderValue.textContent = value.toFixed(3);
      }
      this.onPointSizeChangeCallback?.(value);
    });

    this.settingsBtn.addEventListener('click', () => {
      this.toggleSettings();
    });

    this.downloadSplatBtn.addEventListener('click', () => {
      this.onDownloadSplatCallback?.();
    });

    this.toggleDetLabelsBtn.addEventListener('click', () => {
      this.onToggleDetLabelsCallback?.();
    });

    this.toggleDetBoxesBtn.addEventListener('click', () => {
      this.onToggleDetBoxesCallback?.();
    });

    this.setTargetsBtn.addEventListener('click', () => {
      this.handleSetTargets();
    });

    this.clearTargetsBtn.addEventListener('click', () => {
      this.detectionInput.value = '';
      this.queryColorMap.clear();
      this.currentQueries = [];
      this.activeQueriesList.innerHTML = '';
      this.detectionResultsList.innerHTML = '';
      this.detectionCountEl.textContent = '0';
      this.onClearTargetsCallback?.();
    });

    // Submit on Enter in inline input
    this.detectionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.handleSetTargets();
      }
    });

    // Event delegation: click × on a query chip to remove it
    this.activeQueriesList.addEventListener('click', (e) => {
      const btn = (e.target as HTMLElement).closest('.query-chip-remove') as HTMLElement | null;
      if (btn?.dataset.query) {
        this.removeQuery(btn.dataset.query);
      }
    });

    // Preview modal close
    this.previewCloseBtn.addEventListener('click', () => this.closePreview());
    this.previewOverlay.addEventListener('click', (e) => {
      if (e.target === this.previewOverlay) this.closePreview();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Escape closes preview modal regardless of focus
      if (e.key === 'Escape') {
        this.closePreview();
        return;
      }
      // Don't steal keys while the user is typing in any input/textarea/select
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
          || (e.target as HTMLElement).isContentEditable) {
        return;
      }
      switch (e.key.toLowerCase()) {
        case 'p':
          this.onTogglePointsCallback?.();
          break;
        case 'c':
          this.onToggleCamerasCallback?.();
          break;
        case 'g':
          this.onToggleGridCallback?.();
          break;
        case 'f':
          this.onFlipYCallback?.();
          break;
        case 'r':
          this.onResetViewCallback?.();
          break;
        case 's':
          this.toggleSettings();
          break;
      }
    });
  }

  /**
   * Update connection status display
   */
  updateStatus(state: ConnectionState, message: string, latency?: number): void {
    this.statusText.textContent = message;

    // Update status dot color
    this.statusDot.className = 'status-dot';
    switch (state) {
      case 'connected':
        this.statusDot.classList.add('status-connected');
        this.statusElement.classList.remove('status-error', 'status-connecting');
        this.statusElement.classList.add('status-success');
        break;
      case 'connecting':
        this.statusDot.classList.add('status-connecting');
        this.statusElement.classList.remove('status-error', 'status-success');
        this.statusElement.classList.add('status-warning');
        break;
      case 'error':
        this.statusDot.classList.add('status-error');
        this.statusElement.classList.remove('status-success', 'status-warning');
        this.statusElement.classList.add('status-error');
        break;
      case 'disconnected':
        this.statusDot.classList.add('status-disconnected');
        this.statusElement.classList.remove('status-success', 'status-warning', 'status-error');
        break;
    }

    // Update latency
    if (latency !== undefined) {
      this.latencyElement.textContent = `${latency}ms`;
      this.latencyElement.style.display = 'inline';
    } else {
      this.latencyElement.style.display = 'none';
    }

    // Update button states based on connection
    const isConnected = state === 'connected';
    this.connectBtn.disabled = isConnected;
    this.disconnectBtn.disabled = !isConnected;
    this.resetBtn.disabled = !isConnected;
    this.toggleCamsBtn.disabled = !isConnected;
    this.followCamBtn.disabled = !isConnected;
    // These are local-only controls, always available
    this.resetViewBtn.disabled = false;
    this.toggleGridBtn.disabled = false;
    this.settingsBtn.disabled = false;
  }

  /**
   * Update statistics display
   */
  updateStats(stats: SLAMStats): void {
    this.frameElement.textContent = stats.frames.toString();
    this.submapsElement.textContent = stats.submaps.toString();
    this.loopsElement.textContent = stats.loops.toString();
    this.pointsElement.textContent = this.formatNumber(stats.points);
    this.camerasElement.textContent = stats.cameras.toString();

    // Update FPS
    this.frameCount++;
    const now = Date.now();
    if (now - this.lastUpdate > 1000) {
      const fps = this.frameCount / ((now - this.lastUpdate) / 1000);
      this.fpsElement.textContent = fps.toFixed(1);
      this.frameCount = 0;
      this.lastUpdate = now;
    }

    // Animate value changes
    this.animateValueChange(this.pointsElement);
  }

  /**
   * Update control button states
   */
  updateControlStates(config: ControlsConfig): void {
    this.toggleCamsBtn.textContent = config.showCameras ? 'ON' : 'OFF';
    this.toggleCamsBtn.classList.toggle('active', config.showCameras);

    this.togglePointsBtn.textContent = config.showPoints ? 'ON' : 'OFF';
    this.togglePointsBtn.classList.toggle('active', config.showPoints);

    this.toggleGridBtn.textContent = config.showGrid ? 'ON' : 'OFF';
    this.toggleGridBtn.classList.toggle('active', config.showGrid);

    this.followCamBtn.textContent = config.followCamera ? 'ON' : 'OFF';
    this.followCamBtn.classList.toggle('active', config.followCamera);

    this.flipYBtn.textContent = config.flipY ? 'ON' : 'OFF';
    this.flipYBtn.classList.toggle('active', config.flipY);
    this.toggleDetLabelsBtn.textContent = config.showDetectionLabels
      ? 'Labels'
      : 'Labels';
    this.toggleDetLabelsBtn.classList.toggle('active', config.showDetectionLabels);
    this.toggleDetBoxesBtn.textContent = config.showDetectionBoxes
      ? 'Boxes'
      : 'Boxes';
    this.toggleDetBoxesBtn.classList.toggle('active', config.showDetectionBoxes);
  }

  /**
   * Show notification message
   */
  showNotification(message: string, type: 'info' | 'success' | 'error' = 'info'): void {
    this.notificationQueue.push({ message, type });
    if (this.notificationQueue.length > this.maxQueuedNotifications) {
      this.notificationQueue = this.notificationQueue.slice(-this.maxQueuedNotifications);
    }
    this.scheduleNotificationFlush();
  }

  private scheduleNotificationFlush(): void {
    if (this.notificationFlushScheduled) {
      return;
    }
    this.notificationFlushScheduled = true;
    requestAnimationFrame(() => {
      this.notificationFlushScheduled = false;
      this.flushNotifications();
    });
  }

  private flushNotifications(): void {
    while (
      this.notificationQueue.length > 0
      && this.activeNotifications.length < this.maxActiveNotifications
    ) {
      const next = this.notificationQueue.shift();
      if (!next) {
        break;
      }

      const notification = document.createElement('div');
      notification.className = `notification notification-${next.type}`;
      notification.textContent = next.message;
      notification.style.right = '24px';
      notification.style.left = 'auto';
      notification.style.transform = 'translateY(0)';

      document.body.appendChild(notification);
      this.activeNotifications.push(notification);
      this.repositionNotifications();

      setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
          notification.remove();
          this.activeNotifications = this.activeNotifications.filter((n) => n !== notification);
          this.repositionNotifications();
          this.flushNotifications();
        }, 300);
      }, 2200);
    }
  }

  private repositionNotifications(): void {
    this.activeNotifications.forEach((notification, index) => {
      notification.style.bottom = `${96 + index * 56}px`;
    });
  }

  /**
   * Toggle settings panel
   */
  private toggleSettings(): void {
    this.settingsOpen = !this.settingsOpen;
    this.settingsPanel.classList.toggle('open', this.settingsOpen);
    this.settingsBtn.classList.toggle('active', this.settingsOpen);
  }

  /**
   * Parse inline input and add new queries to the existing set
   */
  private handleSetTargets(): void {
    const raw = this.detectionInput.value.trim();
    if (!raw) return;
    const newItems = raw.split(/[,\n]/).map(q => q.trim().toLowerCase()).filter(q => q.length > 0);
    if (newItems.length === 0) return;
    this.detectionInput.value = '';
    const merged = [...new Set([...this.currentQueries, ...newItems])];
    this.onSetTargetsCallback?.(merged);
  }

  /**
   * Remove a single query from the active set
   */
  private removeQuery(query: string): void {
    const remaining = this.currentQueries.filter(q => q !== query);
    if (remaining.length === this.currentQueries.length) return;
    if (remaining.length === 0) {
      this.onClearTargetsCallback?.();
    } else {
      this.onSetTargetsCallback?.(remaining);
    }
  }

  /**
   * Update the display of active query chips (with hover-to-remove × button)
   */
  updateDetectionQueries(queries: string[]): void {
    this.currentQueries = [...queries];

    // Assign stable colors
    let idx = 0;
    for (const q of queries) {
      if (!this.queryColorMap.has(q)) {
        this.queryColorMap.set(q, BOX_COLORS[idx % BOX_COLORS.length]);
      }
      idx++;
    }
    // Remove stale colors
    const activeSet = new Set(queries);
    for (const key of this.queryColorMap.keys()) {
      if (!activeSet.has(key)) this.queryColorMap.delete(key);
    }

    this.activeQueriesList.innerHTML = '';
    for (const q of queries) {
      const color = this.queryColorMap.get(q) || '#ffffff';
      const chip = document.createElement('span');
      chip.className = 'query-chip';
      chip.innerHTML = `<span class="query-chip-dot" style="background:${color}"></span>${this.escapeHtml(q)}<button class="query-chip-remove" data-query="${this.escapeHtml(q)}" title="Remove">&times;</button>`;
      this.activeQueriesList.appendChild(chip);
    }
  }

  /**
   * Update the detection results gallery (clickable cards with DET type badge).
   * Cards auto-fetch their keyframe thumbnail and show it instantly on click.
   */
  updateDetectionResults(detections: DetectionResult[]): void {
    const successful = detections.filter(d => d.success);
    this.detectionCountEl.textContent = successful.length.toString();
    this.detectionResultsList.innerHTML = '';

    for (const det of successful) {
      const color = this.queryColorMap.get(det.query) || '#ffffff';
      const conf = det.confidence !== undefined ? `${(det.confidence * 100).toFixed(0)}%` : '';
      const key = this.previewKey(det.matched_submap ?? -1, det.matched_frame ?? -1, det.query);
      const cached = this.previewCache.get(key);
      const imageB64 = cached?.keyframe_image ?? det.keyframe_image;

      const card = document.createElement('figure');
      card.className = 'agent-context-card agent-context-card--det det-card-clickable';
      card.dataset.previewKey = key;
      card.innerHTML = `
        <span class="context-type-tag">DET</span>
        ${imageB64
          ? `<img src="data:image/jpeg;base64,${imageB64}" alt="${this.escapeHtml(det.query)}" />`
          : `<div class="context-card-placeholder" style="--card-color:${color}">
               <span class="context-card-placeholder-label">${this.escapeHtml(det.query)}</span>
             </div>`}
        <figcaption>
          <span>${this.escapeHtml(det.query)}</span>
          ${conf ? `<span class="context-card-conf">${conf}</span>` : ''}
        </figcaption>`;

      card.addEventListener('click', () => {
        const hit = this.previewCache.get(key);
        if (hit) {
          this.showDetectionPreview(hit);
        } else {
          this.showPreviewLoading(det);
          this.onDetectionClickCallback?.(det);
        }
      });
      this.detectionResultsList.appendChild(card);

      // Auto-fetch thumbnail if not yet cached or in-flight
      if (!imageB64 && !this.pendingPreviews.has(key)
          && det.matched_submap != null && det.matched_frame != null) {
        this.pendingPreviews.add(key);
        this.onAutoFetchPreviewCallback?.(det);
      }
    }
  }

  /**
   * Receive a preview response: cache it, update the card thumbnail,
   * and populate the modal if the user is currently waiting for it.
   */
  receivePreviewData(data: DetectionPreview): void {
    const key = this.previewKey(data.submap_id, data.frame_idx, data.query);
    if (!data.error) {
      this.previewCache.set(key, data);
    }
    this.pendingPreviews.delete(key);

    // Update card thumbnail in-place
    if (data.keyframe_image) {
      const card = this.detectionResultsList.querySelector<HTMLElement>(`[data-preview-key="${key}"]`);
      if (card) {
        const placeholder = card.querySelector<HTMLElement>('.context-card-placeholder');
        if (placeholder) {
          const img = document.createElement('img');
          img.src = `data:image/jpeg;base64,${data.keyframe_image}`;
          img.alt = data.query;
          placeholder.replaceWith(img);
        }
      }
    }

    // If modal is open waiting for this exact item, populate it now
    if (this.activePreviewKey === key) {
      this.showDetectionPreview(data);
    }
  }

  private previewKey(submap: number, frame: number, query: string): string {
    return `${submap}_${frame}_${query}`;
  }

  onAutoFetchPreview(callback: (det: DetectionResult) => void): void {
    this.onAutoFetchPreviewCallback = callback;
  }

  /**
   * Show preview modal in loading state while images are fetched
   */
  private showPreviewLoading(det: DetectionResult): void {
    this.activePreviewKey = this.previewKey(det.matched_submap ?? -1, det.matched_frame ?? -1, det.query);
    this.previewTitle.textContent = `"${det.query}" — Submap ${det.matched_submap}, Frame ${det.matched_frame}`;
    this.previewKeyframe.removeAttribute('src');
    this.previewMask.removeAttribute('src');
    this.previewKeyframe.classList.add('loading');
    this.previewMask.classList.add('loading');
    this.previewMeta.textContent = 'Loading preview…';
    this.previewOverlay.classList.remove('hidden');
  }

  /**
   * Populate preview modal with received images
   */
  showDetectionPreview(data: DetectionPreview): void {
    this.activePreviewKey = this.previewKey(data.submap_id, data.frame_idx, data.query);
    this.previewTitle.textContent = `"${data.query}" — Submap ${data.submap_id}, Frame ${data.frame_idx}`;

    if (data.keyframe_image) {
      this.previewKeyframe.src = `data:image/jpeg;base64,${data.keyframe_image}`;
      this.previewKeyframe.classList.remove('loading');
    }
    if (data.mask_image) {
      this.previewMask.src = `data:image/png;base64,${data.mask_image}`;
      this.previewMask.classList.remove('loading');
    } else {
      this.previewMask.classList.remove('loading');
      // Show a placeholder if no mask was produced
      this.previewMask.removeAttribute('src');
    }

    this.previewMeta.textContent = data.error
      ? `Error: ${data.error}`
      : `Query: "${data.query}" | Submap ${data.submap_id} | Frame ${data.frame_idx}`;

    this.previewOverlay.classList.remove('hidden');
  }

  /**
   * Close the preview modal
   */
  closePreview(): void {
    this.activePreviewKey = null;
    this.previewOverlay.classList.add('hidden');
  }

  private escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Format large numbers
   */
  private formatNumber(num: number): string {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  }

  /**
   * Animate value change
   */
  private animateValueChange(element: HTMLElement): void {
    element.classList.add('value-update');
    setTimeout(() => element.classList.remove('value-update'), 300);
  }

  // Event callback setters
  onConnect(callback: () => void): void {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback: () => void): void {
    this.onDisconnectCallback = callback;
  }

  onReset(callback: () => void): void {
    this.onResetCallback = callback;
  }

  onResetView(callback: () => void): void {
    this.onResetViewCallback = callback;
  }

  onToggleCameras(callback: () => void): void {
    this.onToggleCamerasCallback = callback;
  }

  onToggleGrid(callback: () => void): void {
    this.onToggleGridCallback = callback;
  }

  onTogglePoints(callback: () => void): void {
    this.onTogglePointsCallback = callback;
  }

  onFlipY(callback: () => void): void {
    this.onFlipYCallback = callback;
  }

  onFollowCamera(callback: () => void): void {
    this.onFollowCameraCallback = callback;
  }

  onPointSizeChange(callback: (size: number) => void): void {
    this.onPointSizeChangeCallback = callback;
  }

  onDownloadSplat(callback: () => void): void {
    this.onDownloadSplatCallback = callback;
  }

  onSetTargets(callback: (queries: string[]) => void): void {
    this.onSetTargetsCallback = callback;
  }

  onClearTargets(callback: () => void): void {
    this.onClearTargetsCallback = callback;
  }

  onDetectionClick(callback: (det: DetectionResult) => void): void {
    this.onDetectionClickCallback = callback;
  }

  onToggleDetLabels(callback: () => void): void {
    this.onToggleDetLabelsCallback = callback;
  }

  onToggleDetBoxes(callback: () => void): void {
    this.onToggleDetBoxesCallback = callback;
  }

  /**
   * Programmatically populate detection input with objects
   */
  public populateDetectionInput(objects: string[]): void {
    this.updateDetectionQueries(objects);
    console.log('📝 Populated detection queries:', objects);
  }
}
