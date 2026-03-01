import type {
  AgentThought,
  AgentAction,
  AgentFinding,
  AgentState,
  AgentToolEvent,
  AgentTaskEvent,
  AgentUICommand,
  AgentUIResult,
  MissionState,
  AgentTaskState,
} from '../types';

const MAX_FEED_ENTRIES = 200;
const MAX_OBS_CARDS = 12;

export class AgentPanel {
  // Panel root
  private panel: HTMLElement;

  // Header elements
  private toggleBtn: HTMLButtonElement;
  private healthPill: HTMLElement;
  private roomType: HTMLElement;
  private coverageText: HTMLElement;
  private coverageBar: HTMLElement;
  private submapsEl: HTMLElement;
  private queryCountEl: HTMLElement;

  // Task/mission elements
  private activeTasksEl: HTMLElement;
  private missionsEl: HTMLElement;

  // Gallery elements
  private contextGallery: HTMLElement;
  private contextEmpty: HTMLElement;

  // Feed elements
  private activityFeed: HTMLElement;
  private filterBtns: NodeListOf<Element>;
  private pauseBtn: HTMLButtonElement;
  private clearBtn: HTMLButtonElement;
  private feedPausedEl: HTMLElement;

  // Chat elements
  private chatInput: HTMLInputElement;
  private chatSend: HTMLButtonElement;

  // State (sidebar is always visible)
  private open = true;
  private autonomyEnabled = true;
  private feedPaused = false;
  private pauseBuffer: HTMLElement[] = [];
  private filterState = new Map<string, boolean>();
  private filterCounts = new Map<string, number>();

  // Callbacks
  private onChatSendCallback?: (message: string) => void;
  private onAgentToggleCallback?: (enabled: boolean) => void;

  constructor() {
    this.panel = this.getEl('agent-panel');
    this.toggleBtn = this.getEl('agent-toggle-btn') as HTMLButtonElement;
    this.healthPill = this.getEl('agent-health-pill');
    this.roomType = this.getEl('agent-room-type');
    this.coverageText = this.getEl('agent-coverage-text');
    this.coverageBar = this.getEl('agent-coverage-bar');
    this.submapsEl = this.getEl('agent-submaps');
    this.queryCountEl = this.getEl('agent-query-count');
    this.activeTasksEl = this.getEl('agent-active-tasks');
    this.missionsEl = this.getEl('agent-missions');
    this.contextGallery = this.getEl('agent-context-gallery');
    this.contextEmpty = this.getEl('agent-context-empty');
    this.activityFeed = this.getEl('agent-activity-feed');
    this.filterBtns = document.querySelectorAll('.agent-feed-filter');
    this.pauseBtn = this.getEl('agent-feed-pause') as HTMLButtonElement;
    this.clearBtn = this.getEl('agent-feed-clear') as HTMLButtonElement;
    this.feedPausedEl = this.getEl('agent-feed-paused');
    this.chatInput = this.getEl('agent-chat-input') as HTMLInputElement;
    this.chatSend = this.getEl('agent-chat-send') as HTMLButtonElement;

    this.initFilters();
    this.setupListeners();
  }

  private getEl(id: string): HTMLElement {
    const el = document.getElementById(id);
    if (!el) throw new Error(`AgentPanel: #${id} not found`);
    return el;
  }

  private initFilters(): void {
    this.filterBtns.forEach((btn) => {
      const cat = (btn as HTMLElement).dataset.category ?? '';
      this.filterState.set(cat, true);
      this.filterCounts.set(cat, 0);
    });
  }

  private setupListeners(): void {
    // Autonomy toggle
    this.toggleBtn.addEventListener('click', () => {
      this.autonomyEnabled = !this.autonomyEnabled;
      this.toggleBtn.textContent = this.autonomyEnabled ? 'AUTONOMY ON' : 'AUTONOMY OFF';
      this.toggleBtn.classList.toggle('active', this.autonomyEnabled);
      this.onAgentToggleCallback?.(this.autonomyEnabled);
    });

    // Chat
    this.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.sendChat();
      }
    });
    this.chatSend.addEventListener('click', () => this.sendChat());

    // Feed filters
    this.filterBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const cat = (btn as HTMLElement).dataset.category ?? '';
        const nowActive = !btn.classList.contains('active');
        this.filterState.set(cat, nowActive);
        btn.classList.toggle('active', nowActive);
        this.activityFeed
          .querySelectorAll<HTMLElement>(`.agent-entry[data-category="${cat}"]`)
          .forEach((el) => el.classList.toggle('entry-hidden', !nowActive));
      });
    });

    // Pause / resume
    this.pauseBtn.addEventListener('click', () => {
      this.feedPaused = !this.feedPaused;
      this.pauseBtn.classList.toggle('active', this.feedPaused);
      this.pauseBtn.textContent = this.feedPaused ? 'Resume' : 'Pause';
      if (!this.feedPaused) {
        for (const entry of this.pauseBuffer) {
          this.insertFeedEntry(entry);
        }
        this.pauseBuffer = [];
        this.feedPausedEl.classList.add('is-hidden');
      }
    });

    // Clear
    this.clearBtn.addEventListener('click', () => {
      this.activityFeed.innerHTML = '';
      this.pauseBuffer = [];
      this.filterCounts.forEach((_, cat) => this.filterCounts.set(cat, 0));
      this.updateFilterCountBadges();
      this.feedPausedEl.classList.add('is-hidden');
    });
  }

  private sendChat(): void {
    const msg = this.chatInput.value.trim();
    if (!msg) return;
    this.chatInput.value = '';
    this.onChatSendCallback?.(msg);
  }

  // ── Public API ──────────────────────────────────────────────────

  toggle(): void {
    this.open = !this.open;
    this.panel.classList.toggle('open', this.open);
  }

  isOpen(): boolean {
    return this.open;
  }

  onChatSend(callback: (message: string) => void): void {
    this.onChatSendCallback = callback;
  }

  onAgentToggle(callback: (enabled: boolean) => void): void {
    this.onAgentToggleCallback = callback;
  }

  // ── Event handlers ──────────────────────────────────────────────

  handleThought(data: AgentThought): void {
    const ts = this.formatTime(data.timestamp);
    const label = data.subagent ? `Thought · ${data.subagent}` : `Thought · ${data.type}`;
    const extraClass = data.type === 'error' ? 'error' : data.type === 'thinking' ? 'thinking' : undefined;
    const entry = this.buildEntry('thought', label, data.content, ts, extraClass);
    this.addToFeed(entry, 'thought');

    if (data.keyframe_b64) {
      this.addObsCard(data.keyframe_b64, data.subagent ?? data.type);
    }
  }

  handleAction(data: AgentAction): void {
    const ts = this.formatTime(data.timestamp);
    const entry = this.buildEntry('action', `Action · ${data.action}`, data.details, ts);
    this.addToFeed(entry, 'action');
  }

  handleFinding(data: AgentFinding): void {
    const ts = this.formatTime(data.timestamp);
    const posStr = data.position
      ? ` @ (${data.position.map((v) => v.toFixed(2)).join(', ')})`
      : '';
    const entry = this.buildEntry('finding', `Found: ${data.query}`, data.description + posStr, ts);
    this.addToFeed(entry, 'finding');
  }

  handleState(data: AgentState): void {
    const pct = Math.round(data.coverage_estimate * 100);
    this.coverageBar.style.width = `${pct}%`;
    this.coverageText.textContent = `${pct}%`;
    this.roomType.textContent = data.room_type || 'Scanning...';
    this.submapsEl.textContent = String(data.submaps_processed ?? 0);
    this.queryCountEl.textContent = String(data.active_queries?.length ?? 0);

    // Health pill
    const health = data.health ?? (data.enabled ? 'ok' : 'disabled');
    this.healthPill.className = 'agent-health-pill';
    if (health === 'ok') {
      this.healthPill.classList.add('health-ok');
      this.healthPill.textContent = 'SYSTEM READY';
    } else if (health === 'degraded') {
      this.healthPill.classList.add('health-degraded');
      this.healthPill.textContent = 'DEGRADED';
    } else {
      this.healthPill.classList.add('health-disabled');
      this.healthPill.textContent = 'DISABLED';
    }

    // Autonomy toggle button
    this.autonomyEnabled = data.enabled;
    this.toggleBtn.textContent = data.enabled ? 'AUTONOMY ON' : 'AUTONOMY OFF';
    this.toggleBtn.classList.toggle('active', data.enabled);

    this.renderMissions(data.missions ?? []);
    this.renderActiveTasks(data.active_tasks ?? []);
  }

  handleToolEvent(data: AgentToolEvent): void {
    const ts = this.formatTime(Date.now() / 1000);
    const statusClass =
      data.status === 'succeeded' ? 'tool-succeeded' : data.status === 'failed' ? 'tool-failed' : undefined;
    const content = data.error ?? (data.result ? JSON.stringify(data.result).slice(0, 120) : '');
    const entry = this.buildEntry('tool', `Tool · ${data.tool} · ${data.status}`, content, ts, statusClass);
    if (data.args) {
      const details = document.createElement('details');
      details.className = 'agent-entry-details';
      details.innerHTML = `<summary>Args</summary><pre>${this.escapeHtml(JSON.stringify(data.args, null, 2))}</pre>`;
      entry.appendChild(details);
    }
    this.addToFeed(entry, 'tool');
  }

  handleTaskEvent(data: AgentTaskEvent): void {
    const ts = this.formatTime(data.timestamp);
    const statusClass =
      data.status === 'succeeded'
        ? 'task-succeeded'
        : data.status === 'failed' || data.status === 'timed_out' || data.status === 'stalled'
        ? `task-${data.status}`
        : undefined;
    const content = data.error ?? data.details ?? '';
    const entry = this.buildEntry('task', `Task · ${data.name} · ${data.status}`, content, ts, statusClass);
    this.addToFeed(entry, 'task');
  }

  handleUICommand(cmd: AgentUICommand): void {
    const ts = this.formatTime(Date.now() / 1000);
    const entry = this.buildEntry('ui', `UI Command · ${cmd.name}`, JSON.stringify(cmd.args).slice(0, 100), ts);
    this.addToFeed(entry, 'ui');
  }

  handleUIResult(result: AgentUIResult, cmdName: string): void {
    const ts = this.formatTime(Date.now() / 1000);
    const statusClass =
      result.status === 'ok' ? 'ui-ok' : result.status === 'error' ? 'ui-error' : undefined;
    const entry = this.buildEntry(
      'ui',
      `UI Result · ${cmdName} · ${result.status}`,
      result.error ?? '',
      ts,
      statusClass,
    );
    this.addToFeed(entry, 'ui');
  }

  // ── Private helpers ─────────────────────────────────────────────

  private renderMissions(missions: MissionState[]): void {
    if (missions.length === 0) {
      this.missionsEl.innerHTML = '<div class="agent-missions-empty">No active missions</div>';
      return;
    }
    this.missionsEl.innerHTML = '';
    for (const m of missions) {
      const pct = Math.round(m.confidence * 100);
      const card = document.createElement('div');
      card.className = `agent-mission-card mission-${m.status}`;
      card.innerHTML = `
        <div class="agent-mission-header">
          <span class="agent-mission-category">${this.escapeHtml(m.category)}</span>
          <span class="agent-mission-status">${m.status}</span>
        </div>
        <div class="agent-mission-goal">${this.escapeHtml(m.goal)}</div>
        ${m.stall_reason ? `<div class="agent-mission-stall-reason">${this.escapeHtml(m.stall_reason)}</div>` : ''}
        <div class="agent-mission-progress-row">
          <span>${m.findings_count} findings</span>
          <span>${pct}%</span>
        </div>
        <div class="agent-mission-bar-container">
          <div class="agent-mission-bar" style="width:${pct}%"></div>
        </div>
        <div class="agent-mission-queries">${m.queries
          .map(
            (q) =>
              `<span class="agent-mission-query${m.found.includes(q) ? ' found' : ''}">${this.escapeHtml(q)}</span>`,
          )
          .join('')}</div>`;
      this.missionsEl.appendChild(card);
    }
  }

  private renderActiveTasks(tasks: AgentTaskState[]): void {
    if (tasks.length === 0) {
      this.activeTasksEl.innerHTML = '<span class="agent-task-pill idle">No active tasks</span>';
      return;
    }
    this.activeTasksEl.innerHTML = '';
    for (const t of tasks) {
      const pill = document.createElement('span');
      pill.className = `agent-task-pill status-${t.status}`;
      pill.textContent = t.name;
      this.activeTasksEl.appendChild(pill);
    }
  }

  private addObsCard(imageB64: string, label: string): void {
    // Remove oldest OBS cards if over limit
    const existing = this.contextGallery.querySelectorAll<HTMLElement>('.agent-context-card--obs');
    if (existing.length >= MAX_OBS_CARDS) {
      existing[existing.length - 1].remove();
    }

    // Hide the empty placeholder once we have an OBS card
    this.contextEmpty.classList.add('is-hidden');

    const card = document.createElement('figure');
    card.className = 'agent-context-card agent-context-card--obs';
    card.innerHTML = `
      <span class="context-type-tag context-type-tag--obs">OBS</span>
      <img src="data:image/jpeg;base64,${imageB64}" alt="${this.escapeHtml(label)}" />
      <figcaption>${this.escapeHtml(label)}</figcaption>`;

    // Insert after #detectionResultsList so detection cards stay grouped at top
    const detList = document.getElementById('detectionResultsList');
    if (detList && detList.nextSibling) {
      this.contextGallery.insertBefore(card, detList.nextSibling);
    } else {
      this.contextGallery.appendChild(card);
    }
  }

  private buildEntry(
    category: string,
    label: string,
    content: string,
    timestamp: string,
    extraClass?: string,
  ): HTMLElement {
    const entry = document.createElement('div');
    entry.className = `agent-entry agent-entry-${category}${extraClass ? ` agent-entry-${extraClass}` : ''}`;
    entry.dataset.category = category;
    // Apply hidden state if the filter for this category is off
    if (!this.filterState.get(category)) {
      entry.classList.add('entry-hidden');
    }
    entry.innerHTML = `
      <div class="agent-entry-header">
        <span class="agent-entry-label">${this.escapeHtml(label)}</span>
        <span class="agent-entry-meta">${timestamp}</span>
      </div>
      ${content ? `<p class="agent-entry-content">${this.escapeHtml(content)}</p>` : ''}`;
    return entry;
  }

  private addToFeed(entry: HTMLElement, category: string): void {
    if (this.feedPaused) {
      this.pauseBuffer.unshift(entry);
      this.feedPausedEl.textContent = `${this.pauseBuffer.length} paused`;
      this.feedPausedEl.classList.remove('is-hidden');
      return;
    }
    this.insertFeedEntry(entry);
  }

  private insertFeedEntry(entry: HTMLElement): void {
    const cat = entry.dataset.category ?? '';
    if (!this.filterState.get(cat)) {
      entry.classList.add('entry-hidden');
    }
    this.activityFeed.prepend(entry);
    // Trim to max
    while (this.activityFeed.children.length > MAX_FEED_ENTRIES) {
      this.activityFeed.removeChild(this.activityFeed.lastChild!);
    }
    // Update count badge
    const count = (this.filterCounts.get(cat) ?? 0) + 1;
    this.filterCounts.set(cat, count);
    this.updateFilterCountBadges();
  }

  private updateFilterCountBadges(): void {
    this.filterBtns.forEach((btn) => {
      const cat = (btn as HTMLElement).dataset.category ?? '';
      const badge = btn.querySelector('.agent-feed-filter-count');
      if (badge) {
        badge.textContent = String(this.filterCounts.get(cat) ?? 0);
      }
    });
  }

  private formatTime(ts: number): string {
    // ts may be seconds or milliseconds
    const d = new Date(ts > 1e10 ? ts : ts * 1000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  private escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
