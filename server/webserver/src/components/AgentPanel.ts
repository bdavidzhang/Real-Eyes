import type { AgentThought, AgentAction, AgentFinding, AgentState, MissionState } from '../types';

/**
 * AgentPanel — Spatial Intelligence Agent UI component.
 *
 * Renders activity feed, mission cards, and chat input.
 * Replaces the detection panel when the agent is active.
 */
export class AgentPanel {
  private panel: HTMLElement;
  private activityFeed: HTMLElement;
  private missionsContainer: HTMLElement;
  private chatInput: HTMLInputElement;
  private chatSendBtn: HTMLButtonElement;
  private toggleBtn: HTMLButtonElement;
  private roomTypeEl: HTMLElement;
  private coverageBar: HTMLElement;
  private coverageText: HTMLElement;
  private agentEnabled = true;

  // Callbacks
  private onSendChat?: (message: string) => void;
  private onToggle?: (enabled: boolean) => void;

  // Activity feed entries (keep last 50)
  private entries: { id: string; el: HTMLElement }[] = [];
  private maxEntries = 50;

  constructor() {
    this.panel = document.getElementById('agent-panel')!;
    this.activityFeed = document.getElementById('agent-activity-feed')!;
    this.missionsContainer = document.getElementById('agent-missions')!;
    this.chatInput = document.getElementById('agent-chat-input') as HTMLInputElement;
    this.chatSendBtn = document.getElementById('agent-chat-send') as HTMLButtonElement;
    this.toggleBtn = document.getElementById('agent-toggle-btn') as HTMLButtonElement;
    this.roomTypeEl = document.getElementById('agent-room-type')!;
    this.coverageBar = document.getElementById('agent-coverage-bar')!;
    this.coverageText = document.getElementById('agent-coverage-text')!;

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
      this.toggleBtn.textContent = this.agentEnabled ? 'ON' : 'OFF';
      this.toggleBtn.classList.toggle('active', this.agentEnabled);
      this.onToggle?.(this.agentEnabled);
    });
  }

  private sendChat(): void {
    const msg = this.chatInput.value.trim();
    if (!msg) return;
    // Show user message in feed
    this.addEntry({
      id: 'user-' + Date.now(),
      icon: '> ',
      content: msg,
      className: 'agent-entry-user',
    });
    this.onSendChat?.(msg);
    this.chatInput.value = '';
  }

  // ------------------------------------------------------------------
  // Event handlers (called from main.ts)
  // ------------------------------------------------------------------

  handleThought(data: AgentThought): void {
    const iconMap: Record<string, string> = {
      observation: '~ ',
      thinking: '~ ',
      chat_response: '< ',
      error: '! ',
      action: '> ',
    };
    this.addEntry({
      id: data.id,
      icon: iconMap[data.type] || '~ ',
      content: data.content,
      className: `agent-entry-${data.type}`,
      typewriter: data.type !== 'error',
    });
  }

  handleAction(data: AgentAction): void {
    this.addEntry({
      id: data.id,
      icon: '> ',
      content: data.details,
      className: 'agent-entry-action',
    });
  }

  handleFinding(data: AgentFinding): void {
    this.addEntry({
      id: data.id,
      icon: '* ',
      content: data.description,
      className: 'agent-entry-finding',
    });
  }

  handleState(data: AgentState): void {
    this.agentEnabled = data.enabled;
    this.toggleBtn.textContent = data.enabled ? 'ON' : 'OFF';
    this.toggleBtn.classList.toggle('active', data.enabled);

    // Update room type
    this.roomTypeEl.textContent = data.room_type !== 'unknown'
      ? data.room_type.charAt(0).toUpperCase() + data.room_type.slice(1)
      : 'Scanning...';

    // Update coverage
    const pct = Math.round(data.coverage_estimate * 100);
    this.coverageBar.style.width = `${pct}%`;
    this.coverageText.textContent = `${pct}%`;

    // Update missions
    this.renderMissions(data.missions);
  }

  // ------------------------------------------------------------------
  // Activity feed
  // ------------------------------------------------------------------

  private addEntry(opts: {
    id: string;
    icon: string;
    content: string;
    className: string;
    typewriter?: boolean;
  }): void {
    const el = document.createElement('div');
    el.className = `agent-entry ${opts.className}`;

    const iconSpan = document.createElement('span');
    iconSpan.className = 'agent-entry-icon';
    iconSpan.textContent = opts.icon;
    el.appendChild(iconSpan);

    const contentSpan = document.createElement('span');
    contentSpan.className = 'agent-entry-content';
    el.appendChild(contentSpan);

    const timeSpan = document.createElement('span');
    timeSpan.className = 'agent-entry-time';
    timeSpan.textContent = 'now';
    el.appendChild(timeSpan);

    this.activityFeed.appendChild(el);

    // Typewriter effect
    if (opts.typewriter && opts.content.length > 0) {
      this.typewrite(contentSpan, opts.content);
    } else {
      contentSpan.textContent = opts.content;
    }

    // Trim old entries
    this.entries.push({ id: opts.id, el });
    while (this.entries.length > this.maxEntries) {
      const old = this.entries.shift();
      old?.el.remove();
    }

    // Auto-scroll
    this.activityFeed.scrollTop = this.activityFeed.scrollHeight;
  }

  private typewrite(el: HTMLElement, text: string): void {
    let i = 0;
    const interval = setInterval(() => {
      if (i < text.length) {
        el.textContent += text[i];
        i++;
        // Keep scrolled to bottom during typing
        this.activityFeed.scrollTop = this.activityFeed.scrollHeight;
      } else {
        clearInterval(interval);
      }
    }, 18);
  }

  // ------------------------------------------------------------------
  // Mission cards
  // ------------------------------------------------------------------

  private renderMissions(missions: MissionState[]): void {
    if (missions.length === 0) {
      this.missionsContainer.innerHTML = '<div class="agent-missions-empty">No active missions</div>';
      return;
    }

    this.missionsContainer.innerHTML = '';
    for (const m of missions) {
      const card = document.createElement('div');
      card.className = `agent-mission-card ${m.status === 'completed' ? 'completed' : ''}`;

      const found = m.found.length;
      const total = m.queries.length;
      const pct = total > 0 ? Math.round((found / total) * 100) : 0;

      card.innerHTML = `
        <div class="agent-mission-header">
          <span class="agent-mission-category">${this.escapeHtml(m.category)}</span>
          <span class="agent-mission-progress">${found}/${total} found</span>
        </div>
        <div class="agent-mission-queries">
          ${m.queries.map(q => {
            const isFound = m.found.includes(q);
            return `<span class="agent-mission-query ${isFound ? 'found' : ''}">${this.escapeHtml(q)} ${isFound ? '&#10003;' : '&#183;'}</span>`;
          }).join(' ')}
        </div>
        <div class="agent-mission-bar-container">
          <div class="agent-mission-bar" style="width: ${pct}%"></div>
        </div>
      `;

      this.missionsContainer.appendChild(card);
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

  private escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
