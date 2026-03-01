import type {
  AgentAction,
  AgentAttachment,
  AgentFinding,
  AgentState,
  AgentThought,
  AgentToolEvent,
  AgentUICommand,
  AgentUIResult,
  AgentTaskEvent,
  AgentTaskState,
  MissionState,
} from '../types';

type FeedCategory = 'thought' | 'action' | 'finding' | 'tool' | 'task' | 'ui';

interface FeedEntryOptions {
  id: string;
  className: string;
  label: string;
  content: string;
  category: FeedCategory;
  meta?: string;
  details?: unknown;
}

interface FeedEntryRecord {
  id: string;
  category: FeedCategory;
  el: HTMLElement;
}

interface ContextSnapshot {
  id: string;
  imageB64: string;
  label: string;
  timestamp: number;
}

/**
 * Spatial command-center panel.
 * Renders timeline, mission board, context gallery, and chat controls.
 */
export class AgentPanel {
  private panel: HTMLElement;
  private activityFeed: HTMLElement;
  private missionsContainer: HTMLElement;
  private contextGallery: HTMLElement;
  private chatInput: HTMLInputElement;
  private chatSendBtn: HTMLButtonElement;
  private toggleBtn: HTMLButtonElement;
  private roomTypeEl: HTMLElement;
  private coverageBar: HTMLElement;
  private coverageText: HTMLElement;
  private submapsEl: HTMLElement;
  private queryCountEl: HTMLElement;
  private healthPill: HTMLElement;
  private activeTasksEl: HTMLElement;

  private feedPauseBtn: HTMLButtonElement;
  private feedClearBtn: HTMLButtonElement;
  private pausedBadge: HTMLElement;
  private filterButtons = new Map<FeedCategory, HTMLButtonElement>();
  private filterCounts = new Map<FeedCategory, HTMLElement>();

  private agentEnabled = true;
  private feedPaused = false;

  // Callbacks
  private onSendChat?: (message: string) => void;
  private onToggle?: (enabled: boolean) => void;

  // Timeline state (keep last N)
  private entries: FeedEntryRecord[] = [];
  private seenEntryIds = new Set<string>();
  private pendingEntries: FeedEntryOptions[] = [];
  private pausedEntries: FeedEntryOptions[] = [];
  private renderScheduled = false;
  private maxEntries = 120;

  // Context image state (keep recent snapshots)
  private snapshots: ContextSnapshot[] = [];
  private maxSnapshots = 8;

  private enabledCategories = new Set<FeedCategory>([
    'thought',
    'action',
    'finding',
    'tool',
    'task',
    'ui',
  ]);

  private categoryTotals: Record<FeedCategory, number> = {
    thought: 0,
    action: 0,
    finding: 0,
    tool: 0,
    task: 0,
    ui: 0,
  };

  constructor() {
    this.panel = document.getElementById('agent-panel')!;
    this.activityFeed = document.getElementById('agent-activity-feed')!;
    this.missionsContainer = document.getElementById('agent-missions')!;
    this.contextGallery = document.getElementById('agent-context-gallery')!;
    this.chatInput = document.getElementById('agent-chat-input') as HTMLInputElement;
    this.chatSendBtn = document.getElementById('agent-chat-send') as HTMLButtonElement;
    this.toggleBtn = document.getElementById('agent-toggle-btn') as HTMLButtonElement;
    this.roomTypeEl = document.getElementById('agent-room-type')!;
    this.coverageBar = document.getElementById('agent-coverage-bar')!;
    this.coverageText = document.getElementById('agent-coverage-text')!;
    this.submapsEl = document.getElementById('agent-submaps')!;
    this.queryCountEl = document.getElementById('agent-query-count')!;
    this.healthPill = document.getElementById('agent-health-pill')!;
    this.activeTasksEl = document.getElementById('agent-active-tasks')!;

    this.feedPauseBtn = document.getElementById('agent-feed-pause') as HTMLButtonElement;
    this.feedClearBtn = document.getElementById('agent-feed-clear') as HTMLButtonElement;
    this.pausedBadge = document.getElementById('agent-feed-paused')!;

    this.initializeFeedFilters();
    this.setupListeners();
    this.renderFeedCounts();
    this.renderPausedBadge();
  }

  private initializeFeedFilters(): void {
    const filterButtons = this.panel.querySelectorAll<HTMLButtonElement>('.agent-feed-filter');
    for (const btn of filterButtons) {
      const category = btn.dataset.category as FeedCategory | undefined;
      if (!category || !(category in this.categoryTotals)) {
        continue;
      }
      this.filterButtons.set(category, btn);
      btn.classList.toggle('active', this.enabledCategories.has(category));

      const countEl = btn.querySelector<HTMLElement>('.agent-feed-filter-count');
      if (countEl) {
        this.filterCounts.set(category, countEl);
      }

      btn.addEventListener('click', () => {
        if (this.enabledCategories.has(category)) {
          this.enabledCategories.delete(category);
        } else {
          this.enabledCategories.add(category);
        }
        btn.classList.toggle('active', this.enabledCategories.has(category));
        this.applyFilters();
      });
    }
  }

  private setupListeners(): void {
    this.chatSendBtn.addEventListener('click', () => this.sendChat());
    this.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendChat();
      }
    });

    this.toggleBtn.addEventListener('click', () => {
      this.agentEnabled = !this.agentEnabled;
      this.renderToggleState();
      this.onToggle?.(this.agentEnabled);
    });

    this.feedPauseBtn.addEventListener('click', () => {
      this.feedPaused = !this.feedPaused;
      this.feedPauseBtn.textContent = this.feedPaused ? 'Resume' : 'Pause';
      this.feedPauseBtn.classList.toggle('active', this.feedPaused);

      if (!this.feedPaused && this.pausedEntries.length > 0) {
        this.pendingEntries.push(...this.pausedEntries);
        this.pausedEntries = [];
        this.scheduleRender();
      }
      this.renderPausedBadge();
    });

    this.feedClearBtn.addEventListener('click', () => {
      this.entries = [];
      this.pendingEntries = [];
      this.pausedEntries = [];
      this.seenEntryIds.clear();
      this.categoryTotals = {
        thought: 0,
        action: 0,
        finding: 0,
        tool: 0,
        task: 0,
        ui: 0,
      };
      this.activityFeed.innerHTML = '';
      this.renderFeedCounts();
      this.renderPausedBadge();
    });
  }

  private sendChat(): void {
    const msg = this.chatInput.value.trim();
    if (!msg) return;

    this.enqueueEntry({
      id: `user-${Date.now()}`,
      className: 'agent-entry-user',
      label: 'User',
      content: msg,
      category: 'ui',
      meta: 'chat',
    });

    this.onSendChat?.(msg);
    this.chatInput.value = '';
  }

  // ------------------------------------------------------------------
  // Event handlers (called from main.ts)
  // ------------------------------------------------------------------

  handleThought(data: AgentThought): void {
    const labelMap: Record<string, string> = {
      observation: 'Observe',
      thinking: 'Thinking',
      chat_response: 'Response',
      error: 'Error',
      action: 'Action',
    };

    const metaParts: string[] = [];
    if (data.subagent) metaParts.push(data.subagent);
    if (typeof data.confidence === 'number') {
      metaParts.push(`${Math.round(data.confidence * 100)}%`);
    }

    this.enqueueEntry({
      id: data.id,
      className: `agent-entry-${data.type}`,
      label: labelMap[data.type] || 'Thought',
      content: data.content,
      category: 'thought',
      meta: metaParts.join(' · '),
      details: data,
    });

    if (data.keyframe_b64) {
      this.addSnapshot(data.keyframe_b64, data.subagent || data.type);
    }

    if (data.attachments?.length) {
      this.addAttachments(data.attachments);
    }
  }

  handleAction(data: AgentAction): void {
    this.enqueueEntry({
      id: data.id,
      className: 'agent-entry-action',
      label: this.humanizeAction(data.action),
      content: data.details,
      category: 'action',
      details: data,
    });

    if (Array.isArray(data.queries)) {
      this.queryCountEl.textContent = String(data.queries.length);
    }
  }

  handleFinding(data: AgentFinding): void {
    const metaParts: string[] = [];
    metaParts.push(`confidence ${Math.round(data.confidence * 100)}%`);
    if (data.mission_id != null) metaParts.push(`mission ${data.mission_id}`);

    this.enqueueEntry({
      id: data.id,
      className: 'agent-entry-finding',
      label: 'Finding',
      content: data.description,
      category: 'finding',
      meta: metaParts.join(' · '),
      details: data,
    });
  }

  handleToolEvent(data: AgentToolEvent): void {
    const statusLabel = data.status.toUpperCase();
    const latency = typeof data.latency_ms === 'number' ? `${data.latency_ms}ms` : '';
    const meta = [statusLabel, latency].filter(Boolean).join(' · ');

    this.enqueueEntry({
      id: `tool-${data.id}-${data.status}`,
      className: `agent-entry-tool agent-entry-tool-${data.status}`,
      label: `Tool · ${data.tool}`,
      content: data.status === 'failed'
        ? data.error || 'Tool call failed'
        : `${data.tool} ${data.status}`,
      category: 'tool',
      meta,
      details: {
        args: data.args,
        result: data.result,
        error: data.error,
      },
    });
  }

  handleUICommand(data: AgentUICommand): void {
    const missionMeta = data.mission_id != null ? `mission ${data.mission_id}` : 'ui';
    this.enqueueEntry({
      id: `ui-cmd-${data.id}`,
      className: 'agent-entry-ui',
      label: 'UI Command',
      content: data.name,
      category: 'ui',
      meta: missionMeta,
      details: data.args,
    });
  }

  handleUIResult(result: AgentUIResult, commandName?: string): void {
    const label = commandName ? `UI Result · ${commandName}` : 'UI Result';
    const content = result.status === 'ok'
      ? 'Command applied'
      : result.error || 'Command did not apply';

    this.enqueueEntry({
      id: `ui-result-${result.id}-${result.status}`,
      className: `agent-entry-ui agent-entry-ui-${result.status}`,
      label,
      content,
      category: 'ui',
      meta: result.status,
      details: result.result ?? result.error,
    });
  }

  handleTaskEvent(data: AgentTaskEvent): void {
    const latency = typeof data.latency_ms === 'number' ? `${data.latency_ms}ms` : '';
    const metaBits = [data.task_type, data.status.toUpperCase(), latency].filter(Boolean);
    const content = data.error || data.details || data.name;
    this.enqueueEntry({
      id: `task-${data.id}-${data.status}`,
      className: `agent-entry-task agent-entry-task-${data.status}`,
      label: `Task · ${data.name}`,
      content,
      category: 'task',
      meta: metaBits.join(' · '),
      details: data,
    });
  }

  handleState(data: AgentState): void {
    this.agentEnabled = data.enabled;
    this.renderToggleState();

    this.roomTypeEl.textContent = data.room_type !== 'unknown'
      ? data.room_type.charAt(0).toUpperCase() + data.room_type.slice(1)
      : 'Scanning...';

    const pct = Math.round(data.coverage_estimate * 100);
    this.coverageBar.style.width = `${pct}%`;
    this.coverageText.textContent = `${pct}%`;
    this.submapsEl.textContent = String(data.submaps_processed ?? 0);
    this.queryCountEl.textContent = String(data.active_queries?.length ?? 0);

    this.renderHealthPill(data);
    this.renderActiveTasks(data.active_tasks || []);
    this.renderMissions(data.missions);
  }

  // ------------------------------------------------------------------
  // Timeline
  // ------------------------------------------------------------------

  private enqueueEntry(opts: FeedEntryOptions): void {
    if (!opts.id) {
      return;
    }
    if (this.seenEntryIds.has(opts.id)) {
      return;
    }

    this.seenEntryIds.add(opts.id);
    this.categoryTotals[opts.category] += 1;
    this.renderFeedCounts();

    if (this.feedPaused) {
      this.pausedEntries.push(opts);
      this.renderPausedBadge();
      return;
    }

    this.pendingEntries.push(opts);
    this.scheduleRender();
  }

  private scheduleRender(): void {
    if (this.renderScheduled) {
      return;
    }
    this.renderScheduled = true;

    requestAnimationFrame(() => {
      this.renderScheduled = false;
      this.flushPendingEntries();
    });
  }

  private flushPendingEntries(): void {
    if (!this.pendingEntries.length) {
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const entry of this.pendingEntries) {
      const el = this.createEntryElement(entry);
      fragment.appendChild(el);
      this.entries.push({
        id: entry.id,
        category: entry.category,
        el,
      });
    }

    this.pendingEntries = [];
    this.activityFeed.appendChild(fragment);

    while (this.entries.length > this.maxEntries) {
      const old = this.entries.shift();
      if (!old) {
        continue;
      }
      this.seenEntryIds.delete(old.id);
      old.el.remove();
    }

    this.applyFilters();
    this.activityFeed.scrollTop = this.activityFeed.scrollHeight;
    this.renderPausedBadge();
  }

  private createEntryElement(opts: FeedEntryOptions): HTMLElement {
    const el = document.createElement('article');
    el.className = `agent-entry ${opts.className}`;
    el.dataset.entryId = opts.id;

    const header = document.createElement('div');
    header.className = 'agent-entry-header';

    const label = document.createElement('span');
    label.className = 'agent-entry-label';
    label.textContent = opts.label;
    header.appendChild(label);

    const meta = document.createElement('span');
    meta.className = 'agent-entry-meta';
    meta.textContent = opts.meta || 'now';
    header.appendChild(meta);

    el.appendChild(header);

    const body = document.createElement('p');
    body.className = 'agent-entry-content';
    body.textContent = opts.content;
    el.appendChild(body);

    if (opts.details !== undefined) {
      const details = document.createElement('details');
      details.className = 'agent-entry-details';

      const summary = document.createElement('summary');
      summary.textContent = 'details';
      details.appendChild(summary);

      const pre = document.createElement('pre');
      pre.textContent = this.formatDetails(opts.details);
      details.appendChild(pre);

      el.appendChild(details);
    }

    return el;
  }

  private formatDetails(value: unknown): string {
    if (typeof value === 'string') {
      return value;
    }
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  private renderFeedCounts(): void {
    for (const [category, countEl] of this.filterCounts) {
      countEl.textContent = String(this.categoryTotals[category]);
    }
  }

  private renderPausedBadge(): void {
    const count = this.pausedEntries.length;
    if (!this.feedPaused || count === 0) {
      this.pausedBadge.classList.add('is-hidden');
      this.pausedBadge.textContent = '';
      return;
    }

    this.pausedBadge.classList.remove('is-hidden');
    this.pausedBadge.textContent = `Paused · ${count} buffered`;
  }

  private applyFilters(): void {
    for (const entry of this.entries) {
      const visible = this.enabledCategories.has(entry.category);
      entry.el.classList.toggle('entry-hidden', !visible);
    }
  }

  // ------------------------------------------------------------------
  // Mission board
  // ------------------------------------------------------------------

  private renderMissions(missions: MissionState[]): void {
    if (!missions.length) {
      this.missionsContainer.innerHTML = '<div class="agent-missions-empty">No active missions</div>';
      return;
    }

    this.missionsContainer.innerHTML = '';

    const sorted = [...missions].sort((a, b) => {
      const order: Record<MissionState['status'], number> = {
        active: 0,
        stalled: 1,
        completed: 2,
      };
      return order[a.status] - order[b.status];
    });

    for (const mission of sorted) {
      const card = document.createElement('div');
      card.className = `agent-mission-card mission-${mission.status}`;

      const found = mission.found.length;
      const total = mission.queries.length;
      const pct = total > 0 ? Math.round((found / total) * 100) : 0;
      const stallReason = mission.status === 'stalled' && mission.stall_reason
        ? `<div class="agent-mission-stall-reason">${this.escapeHtml(mission.stall_reason)}</div>`
        : '';
      const progressAgeText = typeof mission.last_progress_ts === 'number'
        ? `${Math.max(0, Math.floor((Date.now() / 1000) - mission.last_progress_ts))}s ago`
        : 'unknown';

      card.innerHTML = `
        <div class="agent-mission-header">
          <span class="agent-mission-category">${this.escapeHtml(mission.category)}</span>
          <span class="agent-mission-status">${this.escapeHtml(mission.status)}</span>
        </div>
        <div class="agent-mission-goal">${this.escapeHtml(mission.goal)}</div>
        <div class="agent-mission-last-progress">Last progress: ${this.escapeHtml(progressAgeText)}</div>
        ${stallReason}
        <div class="agent-mission-progress-row">
          <span>${found}/${total} resolved</span>
          <span>${pct}%</span>
        </div>
        <div class="agent-mission-bar-container">
          <div class="agent-mission-bar" style="width:${pct}%"></div>
        </div>
        <div class="agent-mission-queries">
          ${mission.queries.map((q) => {
            const isFound = mission.found.includes(q);
            return `<span class="agent-mission-query ${isFound ? 'found' : ''}">${this.escapeHtml(q)}</span>`;
          }).join('')}
        </div>
      `;

      this.missionsContainer.appendChild(card);
    }
  }

  // ------------------------------------------------------------------
  // Context gallery
  // ------------------------------------------------------------------

  private addAttachments(attachments: AgentAttachment[]): void {
    for (const attachment of attachments) {
      if (attachment.image_b64) {
        this.addSnapshot(attachment.image_b64, attachment.label || attachment.kind);
      }
    }
  }

  private addSnapshot(imageB64: string, label: string): void {
    if (!imageB64) return;
    const id = imageB64.slice(0, 24);
    if (this.snapshots.some((s) => s.id === id)) return;

    this.snapshots.unshift({
      id,
      imageB64,
      label,
      timestamp: Date.now(),
    });
    this.snapshots = this.snapshots.slice(0, this.maxSnapshots);
    this.renderContextGallery();
  }

  private renderContextGallery(): void {
    if (!this.snapshots.length) {
      this.contextGallery.innerHTML = '<div class="agent-context-empty">No snapshots yet</div>';
      return;
    }

    this.contextGallery.innerHTML = '';
    for (const snap of this.snapshots) {
      const card = document.createElement('figure');
      card.className = 'agent-context-card';
      card.innerHTML = `
        <img src="data:image/jpeg;base64,${snap.imageB64}" alt="${this.escapeHtml(snap.label)}" />
        <figcaption>${this.escapeHtml(snap.label)}</figcaption>
      `;
      this.contextGallery.appendChild(card);
    }
  }

  // ------------------------------------------------------------------
  // Panel visibility
  // ------------------------------------------------------------------

  toggle(): void {
    this.panel.classList.toggle('open');
  }

  isOpen(): boolean {
    return this.panel.classList.contains('open');
  }

  open(): void {
    this.panel.classList.add('open');
  }

  close(): void {
    this.panel.classList.remove('open');
  }

  // ------------------------------------------------------------------
  // Callback setters
  // ------------------------------------------------------------------

  onChatSend(callback: (message: string) => void): void {
    this.onSendChat = callback;
  }

  onAgentToggle(callback: (enabled: boolean) => void): void {
    this.onToggle = callback;
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  private renderToggleState(): void {
    this.toggleBtn.textContent = this.agentEnabled ? 'AUTONOMY ON' : 'AUTONOMY OFF';
    this.toggleBtn.classList.toggle('active', this.agentEnabled);
  }

  private renderHealthPill(data: AgentState): void {
    const degraded = data.degraded_mode || data.health === 'degraded';
    const disabled = !data.enabled || data.health === 'disabled';

    this.healthPill.classList.remove('health-ok', 'health-degraded', 'health-disabled');

    if (disabled) {
      this.healthPill.textContent = 'AUTONOMY OFF';
      this.healthPill.classList.add('health-disabled');
      return;
    }

    if (degraded) {
      this.healthPill.textContent = 'DEGRADED MODE';
      this.healthPill.classList.add('health-degraded');
      return;
    }

    this.healthPill.textContent = 'SYSTEM READY';
    this.healthPill.classList.add('health-ok');
  }

  private renderActiveTasks(tasks: AgentTaskState[]): void {
    if (!tasks.length) {
      this.activeTasksEl.innerHTML = '<span class="agent-task-pill idle">No active tasks</span>';
      return;
    }

    this.activeTasksEl.innerHTML = '';
    for (const task of tasks.slice(0, 6)) {
      const el = document.createElement('span');
      el.className = `agent-task-pill status-${task.status}`;
      el.textContent = `${task.name} · ${task.status}`;
      this.activeTasksEl.appendChild(el);
    }
  }

  private humanizeAction(action: string): string {
    return action
      .split('_')
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ');
  }

  private escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
