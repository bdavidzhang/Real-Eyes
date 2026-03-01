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
const SERVER_URL = import.meta.env.VITE_SERVER_URL || window.location.origin;

// ---- DOM refs ----
const queryInput        = document.getElementById('queryInput')        as HTMLTextAreaElement;
const thresholdInput    = document.getElementById('thresholdInput')    as HTMLInputElement;
const samThresholdInput = document.getElementById('samThresholdInput') as HTMLInputElement;
const topKInput         = document.getElementById('topKInput')         as HTMLInputElement;
const searchBtn         = document.getElementById('searchBtn')         as HTMLButtonElement;
const connectBtn        = document.getElementById('connectBtn')        as HTMLButtonElement;
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
  clip_score: number;
  combined_score: number;
  box_2d: number[];
  mask_image: string;
  above_sam_threshold: boolean;
  sam_threshold_used: number;
  has_3d_box: boolean;
  bbox_3d: any | null;
  dedup_kept?: boolean | null;
}

interface FrameDiag {
  submap_id: number;
  frame_idx: number;
  query: string;
  clip_similarity: number;
  clip_rank: number;
  above_threshold: boolean;
  in_top_k: boolean;
  sam_skipped: boolean;
  clip_threshold_used: number;
  sam_threshold_used: number;
  top_k_used: number;
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
  top_k: number;
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
type FilterType = 'all' | 'above' | 'topk' | 'sam' | 'detected';
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
  const topK = parseInt(topKInput.value, 10) || 3;

  showStatus(
    `Running detection for [${queries.join(', ')}] — CLIP≥${clipThreshold}, SAM≥${samThreshold}, top-K=${topK}…`,
    'info'
  );
  searchBtn.disabled = true;

  socket.emit('debug_detect', {
    queries,
    clip_thresholds: { default: clipThreshold },
    sam_thresholds: { default: samThreshold },
    top_k: topK,
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

  const { queries, clip_thresholds, sam_thresholds, top_k, frames, raw_detection_count,
          deduped_detection_count, detections, total_frames_scanned, query_time_ms } = data;

  const clipDefault = clip_thresholds?.default ?? 0.2;
  const samDefault = sam_thresholds?.default ?? 0.3;
  const aboveCount = frames.filter(f => f.above_threshold).length;
  const topKCount = frames.filter(f => f.in_top_k).length;
  const samCount = frames.filter(f => f.sam_masks.length > 0).length;
  const skippedCount = frames.filter(f => f.sam_skipped).length;

  showStatus(
    `[${queries.join(', ')}] — ${total_frames_scanned} frame-query combos · ` +
    `${aboveCount} above CLIP (≥${clipDefault}) · ` +
    `${topKCount} in top-${top_k ?? 3} · ` +
    `${skippedCount} skipped (above but not top-K) · ` +
    `${samCount} with SAM masks (SAM≥${samDefault}) · ` +
    `${raw_detection_count} raw → ${deduped_detection_count} deduped · ` +
    `${query_time_ms}ms`,
    deduped_detection_count > 0 ? 'success' : 'warn'
  );

  // Show filter buttons
  filterBtns.classList.remove('hidden');
  filterBtns.innerHTML = '';
  const filters: { key: FilterType; label: string; count: number }[] = [
    { key: 'all',      label: 'All frames',       count: frames.length },
    { key: 'above',    label: 'Above CLIP thresh', count: aboveCount },
    { key: 'topk',     label: `Top-${top_k ?? 3} selected`,   count: topKCount },
    { key: 'sam',      label: 'SAM masks found',  count: samCount },
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
    if (currentFilter === 'above')    return f.above_threshold;
    if (currentFilter === 'topk')     return f.in_top_k;
    if (currentFilter === 'sam')      return f.sam_masks.length > 0;
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
    const skipped = f.sam_skipped;
    card.className = [
      'frame-card',
      aboveCls,
      f.in_top_k  ? 'in-topk'    : '',
      hasMasks    ? 'has-masks'  : '',
      hasDets     ? 'has-dets'   : '',
      skipped     ? 'sam-skipped': '',
    ].filter(Boolean).join(' ');

    const simPct = Math.min(Math.max(f.clip_similarity * 100, 0), 100);
    const simColor = similarityColor(f.clip_similarity);

    // Status badge in top-left corner
    let statusBadge = '';
    if (!f.above_threshold) {
      statusBadge = '<span class="fc-badge fc-badge-below">below CLIP</span>';
    } else if (skipped) {
      statusBadge = '<span class="fc-badge fc-badge-skipped">skipped (not top-K)</span>';
    } else if (f.in_top_k) {
      statusBadge = `<span class="fc-badge fc-badge-topk">rank #${f.clip_rank}</span>`;
    }

    card.innerHTML = `
      <div class="fc-img">
        ${f.thumbnail
          ? `<img src="data:image/jpeg;base64,${f.thumbnail}" />`
          : `<div class="fc-placeholder">No image</div>`}
        <span class="fc-sim" style="background:${simColor}">${simPct.toFixed(1)}%</span>
        ${hasMasks ? `<span class="fc-badge fc-badge-sam">${f.sam_masks.length} mask${f.sam_masks.length > 1 ? 's' : ''}</span>` : ''}
        ${statusBadge}
      </div>
      <div class="fc-meta">
        <div class="fc-row"><span>Query</span><span class="mono">${esc(f.query)}</span></div>
        <div class="fc-row"><span>Submap / Frame</span><span>${f.submap_id} / ${f.frame_idx}</span></div>
        <div class="fc-row"><span>CLIP rank</span><span class="mono">#${f.clip_rank} of ${f.top_k_used} selected</span></div>
        <div class="fc-row"><span>CLIP sim</span><span class="mono" style="color:${simColor}">${f.clip_similarity.toFixed(4)}</span></div>
        <div class="fc-row"><span>CLIP thresh</span><span class="mono">${f.clip_threshold_used}</span></div>
        <div class="fc-row"><span>SAM thresh</span><span class="mono">${f.sam_threshold_used}</span></div>
        ${f.resolution ? `<div class="fc-row"><span>Res</span><span class="mono">${f.resolution}</span></div>` : ''}
        ${skipped ? `<div class="fc-row"><span style="color:#facc15">SAM</span><span style="color:#facc15">skipped — not in top-${f.top_k_used}</span></div>` : ''}
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
  for (const m of masks) {
    const maskColor = !m.above_sam_threshold ? '#f87171'
      : m.dedup_kept === true  ? '#4ade80'
      : m.dedup_kept === false ? '#facc15'
      : m.has_3d_box           ? '#4ade80' : '#facc15';
    html += `
      <div class="fc-mask-item">
        <img src="data:image/png;base64,${m.mask_image}" />
        <div class="fc-mask-info">
          <span class="fc-mask-score" style="color:${maskColor}">
            SAM ${(m.score * 100).toFixed(1)}%
          </span>
          <span class="fc-mask-detail mono" style="color:#a1a1aa">
            CLIP ${(m.clip_score * 100).toFixed(1)}%
            · combined ${(m.combined_score * 100).toFixed(2)}%
          </span>
          <span class="fc-mask-detail">
            ${!m.above_sam_threshold
              ? `<span style="color:#f87171">below SAM threshold (${m.sam_threshold_used})</span>`
              : m.has_3d_box ? '3D box: valid' : '3D box: none (insufficient points)'}
          </span>
          ${m.bbox_3d ? `<span class="fc-mask-detail mono">ext: [${m.bbox_3d.extent.map((v: number) => v.toFixed(3)).join(', ')}]</span>` : ''}
          ${m.dedup_kept === true  ? '<span class="fc-mask-detail" style="color:#4ade80">✓ kept after dedup</span>' : ''}
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
    const combinedPct = ((d.confidence ?? 0) * 100).toFixed(2);
    const clipPct  = d.clip_score  != null ? `${(d.clip_score  * 100).toFixed(1)}%` : '—';
    const samPct   = d.sam_score   != null ? `${(d.sam_score   * 100).toFixed(1)}%` : '—';
    html += `
      <div class="det-card">
        <div class="det-header">
          <span class="det-query">${esc(d.query)}</span>
          <span class="det-conf">${combinedPct}% combined</span>
        </div>
        <div class="det-body">
          <div class="fc-row"><span>Submap / Frame</span><span>${d.matched_submap} / ${d.matched_frame}</span></div>
          <div class="fc-row"><span>CLIP score</span><span class="mono">${clipPct}</span></div>
          <div class="fc-row"><span>SAM score</span><span class="mono">${samPct}</span></div>
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
