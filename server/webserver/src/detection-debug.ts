/**
 * Detection Debug Page
 *
 * Runs the EXACT same detection pipeline as production
 * (detect_all_in_submap → cross-submap dedup) via a debug socket event,
 * but returns rich per-frame diagnostics so you can see every intermediate
 * step: CLIP cosine similarities, SAM 3 masks, 3D bounding box validity,
 * and deduplication results.
 *
 * Uses the same ObjectDetector methods as server.py / rescan_all_submaps().
 */

import './detection-debug.css';
import { io, Socket } from 'socket.io-client';

// ---- Server URL (must match server.py) ----
const SERVER_URL = import.meta.env.VITE_SERVER_URL || `https://${window.location.hostname}:5000`;

// ---- DOM refs ----
const queryInput     = document.getElementById('queryInput')     as HTMLTextAreaElement;
const thresholdInput = document.getElementById('thresholdInput') as HTMLInputElement;
const samThresholdInput = document.getElementById('samThresholdInput') as HTMLInputElement;
const searchBtn      = document.getElementById('searchBtn')      as HTMLButtonElement;
const connectBtn     = document.getElementById('connectBtn')     as HTMLButtonElement;
const statusBar      = document.getElementById('statusBar')      as HTMLDivElement;
const emptyState     = document.getElementById('emptyState')     as HTMLDivElement;
const resultsWrap    = document.getElementById('resultsWrap')    as HTMLDivElement;
const summaryBar     = document.getElementById('summaryBar')     as HTMLDivElement;
const resultsGrid    = document.getElementById('resultsGrid')    as HTMLDivElement;
const finalSection   = document.getElementById('finalDetections') as HTMLDivElement;
const connDot        = document.getElementById('connectionDot')  as HTMLSpanElement;
const connText       = document.getElementById('connectionText') as HTMLSpanElement;
const mapStats       = document.getElementById('mapStats')       as HTMLSpanElement;
const filterBtns     = document.getElementById('filterBtns')     as HTMLDivElement;

// ---- Socket state ----
let socket: Socket | null = null;
let connected = false;

// ---- Types ----

interface SamMaskDiag {
  score: number;
  box_2d: number[];
  mask_image: string;      // base64 overlay
  above_sam_threshold: boolean;
  sam_threshold_used: number;
  has_3d_box: boolean;
  bbox_3d: any | null;
  dedup_kept?: boolean;    // present only if a detection was created from this mask
}

interface FrameDiag {
  submap_id: number;
  frame_idx: number;
  query: string;
  clip_similarity: number;
  above_threshold: boolean;
  clip_threshold_used: number;
  sam_threshold_used: number;
  thumbnail: string | null;
  resolution?: string;
  sam_masks: SamMaskDiag[];
  sam_error?: string;
  detections_before_dedup: any[];
}

interface DebugDetectResponse {
  queries: string[];
  clip_thresholds: Record<string, number>;
  sam_thresholds: Record<string, number>;
  frames: FrameDiag[];
  raw_detection_count: number;
  deduped_detection_count: number;
  detections: any[];
  total_frames_scanned: number;
  query_time_ms: number;
  error?: string;
}

// ---- Last response for filtering ----
let lastResponse: DebugDetectResponse | null = null;
type FilterType = 'all' | 'above' | 'sam' | 'detected';
let currentFilter: FilterType = 'all';

// =============================================
// Connection
// =============================================

function connect(): void {
  if (socket) socket.disconnect();
  setConnectionState('connecting');

  socket = io(SERVER_URL, {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5,
  });

  socket.on('connect', () => {
    setConnectionState('connected');
    connected = true;
    socket!.emit('get_global_map');
  });

  socket.on('disconnect', () => { setConnectionState('disconnected'); connected = false; });
  socket.on('connect_error', () => { setConnectionState('error'); connected = false; });

  socket.on('slam_update', handleMapStats);
  socket.on('global_map', handleMapStats);
  socket.on('debug_detect_results', handleDebugResults);
}

function handleMapStats(data: any): void {
  const ns = data.num_submaps ?? 0;
  const nf = data.frame_id ?? 0;
  const np = data.n_points ?? 0;
  mapStats.textContent = `${ns} submaps · ${nf} frames · ${np.toLocaleString()} pts`;
}

function setConnectionState(state: string): void {
  connDot.className = `conn-dot ${state}`;
  const labels: Record<string, string> = {
    disconnected: 'Disconnected', connecting: 'Connecting…',
    connected: 'Connected', error: 'Error',
  };
  connText.textContent = labels[state] ?? state;
  connectBtn.textContent = state === 'connected' ? 'Reconnect' : 'Connect';
  searchBtn.disabled = state !== 'connected';
}

// =============================================
// Run detection
// =============================================

function doSearch(): void {
  const raw = queryInput.value.trim();
  if (!raw || !socket || !connected) return;

  // Split on newlines and commas
  const queries = raw.split(/[\n,]+/).map(s => s.trim()).filter(Boolean);
  if (queries.length === 0) return;

  const clipThreshold = parseFloat(thresholdInput.value) || 0.2;
  const samThreshold = parseFloat(samThresholdInput.value) || 0.3;

  showStatus(`Running full detection pipeline for [${queries.join(', ')}] (CLIP≥${clipThreshold}, SAM≥${samThreshold})…`, 'info');
  searchBtn.disabled = true;

  socket.emit('debug_detect', {
    queries,
    clip_thresholds: { default: clipThreshold },
    sam_thresholds: { default: samThreshold },
  });
}

// =============================================
// Render results
// =============================================

function handleDebugResults(data: DebugDetectResponse): void {
  searchBtn.disabled = !connected;

  if (data.error) {
    showStatus(`Error: ${data.error}`, 'error');
    return;
  }

  lastResponse = data;
  currentFilter = 'all';

  const { queries, clip_thresholds, sam_thresholds, frames, raw_detection_count,
          deduped_detection_count, detections, total_frames_scanned, query_time_ms } = data;

  const clipDefault = clip_thresholds?.default ?? 0.2;
  const samDefault = sam_thresholds?.default ?? 0.3;
  const aboveCount = frames.filter(f => f.above_threshold).length;
  const samCount = frames.filter(f => f.sam_masks.length > 0).length;

  showStatus(
    `[${queries.join(', ')}] — ${total_frames_scanned} frame-query combos · ` +
    `${aboveCount} above CLIP (${clipDefault}) · ` +
    `${samCount} with SAM masks (SAM≥${samDefault}) · ` +
    `${raw_detection_count} raw → ${deduped_detection_count} deduped · ` +
    `${query_time_ms}ms`,
    deduped_detection_count > 0 ? 'success' : 'warn'
  );

  // Show filter buttons
  filterBtns.classList.remove('hidden');
  filterBtns.innerHTML = '';
  const filters: { key: FilterType; label: string; count: number }[] = [
    { key: 'all', label: 'All frames', count: frames.length },
    { key: 'above', label: 'Above threshold', count: aboveCount },
    { key: 'sam', label: 'SAM masks found', count: samCount },
    { key: 'detected', label: 'Valid detections', count: raw_detection_count },
  ];
  for (const f of filters) {
    const btn = document.createElement('button');
    btn.className = `filter-btn ${f.key === 'all' ? 'active' : ''}`;
    btn.textContent = `${f.label} (${f.count})`;
    btn.addEventListener('click', () => {
      currentFilter = f.key;
      filterBtns.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderFrameGrid(data.frames);
    });
    filterBtns.appendChild(btn);
  }

  emptyState.classList.add('hidden');
  resultsWrap.classList.remove('hidden');

  renderFrameGrid(frames);
  renderFinalDetections(detections, queries);
}

function renderFrameGrid(frames: FrameDiag[]): void {
  const filtered = frames.filter(f => {
    if (currentFilter === 'above') return f.above_threshold;
    if (currentFilter === 'sam') return f.sam_masks.length > 0;
    if (currentFilter === 'detected') return f.detections_before_dedup.length > 0;
    return true;
  });

  resultsGrid.innerHTML = '';

  if (filtered.length === 0) {
    resultsGrid.innerHTML = '<p class="no-results">No frames match this filter.</p>';
    return;
  }

  for (const f of filtered) {
    const card = document.createElement('div');
    const aboveCls = f.above_threshold ? 'above' : 'below';
    const hasMasks = f.sam_masks.length > 0;
    const hasDets = f.detections_before_dedup.length > 0;
    card.className = `frame-card ${aboveCls} ${hasMasks ? 'has-masks' : ''} ${hasDets ? 'has-dets' : ''}`;

    const simPct = Math.min(Math.max(f.clip_similarity * 100, 0), 100);
    const simColor = similarityColor(f.clip_similarity);

    card.innerHTML = `
      <div class="fc-img">
        ${f.thumbnail
          ? `<img src="data:image/jpeg;base64,${f.thumbnail}" />`
          : `<div class="fc-placeholder">No image</div>`}
        <span class="fc-sim" style="background:${simColor}">${simPct.toFixed(1)}%</span>
        ${hasMasks ? `<span class="fc-badge fc-badge-sam">${f.sam_masks.length} mask${f.sam_masks.length > 1 ? 's' : ''}</span>` : ''}
        ${!f.above_threshold ? '<span class="fc-badge fc-badge-below">below</span>' : ''}
      </div>
      <div class="fc-meta">
        <div class="fc-row"><span>Query</span><span class="mono">${esc(f.query)}</span></div>
        <div class="fc-row"><span>Submap</span><span>${f.submap_id}</span></div>
        <div class="fc-row"><span>Frame</span><span>${f.frame_idx}</span></div>
        <div class="fc-row"><span>CLIP sim</span><span class="mono" style="color:${simColor}">${f.clip_similarity.toFixed(4)}</span></div>
        <div class="fc-row"><span>CLIP thresh</span><span class="mono">${f.clip_threshold_used}</span></div>
        <div class="fc-row"><span>SAM thresh</span><span class="mono">${f.sam_threshold_used}</span></div>
        ${f.resolution ? `<div class="fc-row"><span>Res</span><span class="mono">${f.resolution}</span></div>` : ''}
        ${f.sam_error ? `<div class="fc-row fc-error"><span>SAM error</span><span>${esc(f.sam_error)}</span></div>` : ''}
        <div class="fc-bar"><div class="fc-bar-fill" style="width:${simPct}%;background:${simColor}"></div></div>
      </div>
      ${hasMasks ? renderSamMasksInline(f.sam_masks) : ''}
    `;

    resultsGrid.appendChild(card);
  }
}

function renderSamMasksInline(masks: SamMaskDiag[]): string {
  let html = '<div class="fc-masks">';
  for (let i = 0; i < masks.length; i++) {
    const m = masks[i];
    // Color: red if below SAM threshold, green if kept after dedup, yellow if discarded by dedup or no 3D box
    const maskColor = !m.above_sam_threshold ? '#f87171'
      : m.dedup_kept === true ? '#4ade80'
      : m.dedup_kept === false ? '#facc15'
      : m.has_3d_box ? '#4ade80' : '#facc15';
    html += `
      <div class="fc-mask-item">
        <img src="data:image/png;base64,${m.mask_image}" />
        <div class="fc-mask-info">
          <span class="fc-mask-score" style="color:${maskColor}">
            SAM ${(m.score * 100).toFixed(1)}%
          </span>
          <span class="fc-mask-detail">
            ${!m.above_sam_threshold
              ? `<span style="color:#f87171">below SAM threshold (${m.sam_threshold_used})</span>`
              : m.has_3d_box ? '3D box: valid' : '3D box: none (insufficient points)'}
          </span>
          ${m.bbox_3d ? `<span class="fc-mask-detail mono">ext: [${m.bbox_3d.extent.map((v: number) => v.toFixed(3)).join(', ')}]</span>` : ''}
          ${m.dedup_kept === true ? '<span class="fc-mask-detail" style="color:#4ade80">✓ kept after dedup</span>' : ''}
          ${m.dedup_kept === false ? '<span class="fc-mask-detail" style="color:#facc15">✗ discarded by dedup</span>' : ''}
        </div>
      </div>
    `;
  }
  html += '</div>';
  return html;
}

function renderFinalDetections(detections: any[], queries: string[]): void {
  finalSection.innerHTML = '';
  if (detections.length === 0) {
    finalSection.innerHTML = '<h3 class="section-title">Final Detections (after dedup)</h3><p class="no-results">No detections survived deduplication / validation.</p>';
    return;
  }

  let html = `<h3 class="section-title">Final Detections (after dedup): ${detections.length}</h3>`;
  html += '<div class="det-list">';
  for (const d of detections) {
    const bb = d.bounding_box;
    html += `
      <div class="det-card">
        <div class="det-header">
          <span class="det-query">${esc(d.query)}</span>
          <span class="det-conf">${((d.confidence ?? 0) * 100).toFixed(1)}%</span>
        </div>
        <div class="det-body">
          <div class="fc-row"><span>Submap</span><span>${d.matched_submap}</span></div>
          <div class="fc-row"><span>Frame</span><span>${d.matched_frame}</span></div>
          ${bb ? `
          <div class="fc-row"><span>Center</span><span class="mono">[${bb.center.map((v: number) => v.toFixed(3)).join(', ')}]</span></div>
          <div class="fc-row"><span>Extent</span><span class="mono">[${bb.extent.map((v: number) => v.toFixed(3)).join(', ')}]</span></div>
          ` : ''}
        </div>
      </div>
    `;
  }
  html += '</div>';
  finalSection.innerHTML = html;
}

function similarityColor(sim: number): string {
  if (sim >= 0.3) return '#4ade80';
  if (sim >= 0.2) return '#94d82d';
  if (sim >= 0.15) return '#facc15';
  if (sim >= 0.1) return '#ff922b';
  return '#f87171';
}

// =============================================
// Helpers
// =============================================

function showStatus(msg: string, kind: 'info' | 'success' | 'warn' | 'error'): void {
  statusBar.textContent = msg;
  statusBar.className = `status-bar status-${kind}`;
  statusBar.classList.remove('hidden');
}

function esc(str: string): string {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

// =============================================
// Event bindings
// =============================================

document.addEventListener('DOMContentLoaded', () => {
  connectBtn.addEventListener('click', connect);
  searchBtn.addEventListener('click', doSearch);
  queryInput.addEventListener('keydown', (e) => {
    // Ctrl/Cmd+Enter to search (allow normal Enter for multi-line)
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doSearch();
  });

  // Auto-connect
  connect();
});
