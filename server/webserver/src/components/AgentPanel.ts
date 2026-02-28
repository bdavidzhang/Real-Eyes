import type {
  AgentAction,
  AgentAttachment,
  AgentFinding,
  AgentState,
  AgentThought,
  MissionState,
} from '../types';

interface FeedEntryOptions {
  id: string;
  className: string;
  label: string;
  content: string;
  meta?: string;
  typewriter?: boolean;
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
  private agentEnabled = true;

  // Callbacks
  private onSendChat?: (message: string) => void;
  private onToggle?: (enabled: boolean) => void;

  // Timeline state (keep last N)
  private entries: { id: string; el: HTMLElement }[] = [];
  private maxEntries = 80;

  // Context image state (keep recent snapshots)
  private snapshots: ContextSnapshot[] = [];
  private maxSnapshots = 8;

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

    this.setupListeners();
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
  }

  private sendChat(): void {
    const msg = this.chatInput.value.trim();
    if (!msg) return;
    this.addEntry({
      id: 'user-' + Date.now(),
      className: 'agent-entry-user',
      label: 'User',
      content: msg,
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

    this.addEntry({
      id: data.id,
      className: `agent-entry-${data.type}`,
      label: labelMap[data.type] || 'Event',
      content: data.content,
      meta: metaParts.join(' · '),
      typewriter: data.type === 'chat_response',
    });

    if (data.keyframe_b64) {
      this.addSnapshot(data.keyframe_b64, data.subagent || data.type);
    }

    if (data.attachments?.length) {
      this.addAttachments(data.attachments);
    }
  }

  handleAction(data: AgentAction): void {
    this.addEntry({
      id: data.id,
      className: 'agent-entry-action',
      label: this.humanizeAction(data.action),
      content: data.details,
    });

    if (Array.isArray(data.queries)) {
      this.queryCountEl.textContent = String(data.queries.length);
    }
  }

  handleFinding(data: AgentFinding): void {
    const metaParts: string[] = [];
    metaParts.push(`confidence ${Math.round(data.confidence * 100)}%`);
    if (data.mission_id != null) metaParts.push(`mission ${data.mission_id}`);

    this.addEntry({
      id: data.id,
      className: 'agent-entry-finding',
      label: 'Finding',
      content: data.description,
      meta: metaParts.join(' · '),
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
    this.renderMissions(data.missions);
  }

  // ------------------------------------------------------------------
  // Timeline
  // ------------------------------------------------------------------

  private addEntry(opts: FeedEntryOptions): void {
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
    el.appendChild(body);

    this.activityFeed.appendChild(el);

    if (opts.typewriter && opts.content.length > 0) {
      this.typewrite(body, opts.content);
    } else {
      body.textContent = opts.content;
    }

    this.entries.push({ id: opts.id, el });
    while (this.entries.length > this.maxEntries) {
      const old = this.entries.shift();
      old?.el.remove();
    }

    this.activityFeed.scrollTop = this.activityFeed.scrollHeight;
  }

  private typewrite(el: HTMLElement, text: string): void {
    let i = 0;
    const interval = setInterval(() => {
      if (i < text.length) {
        el.textContent += text[i];
        i++;
        this.activityFeed.scrollTop = this.activityFeed.scrollHeight;
      } else {
        clearInterval(interval);
      }
    }, 12);
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

      card.innerHTML = `
        <div class="agent-mission-header">
          <span class="agent-mission-category">${this.escapeHtml(mission.category)}</span>
          <span class="agent-mission-status">${this.escapeHtml(mission.status)}</span>
        </div>
        <div class="agent-mission-goal">${this.escapeHtml(mission.goal)}</div>
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
