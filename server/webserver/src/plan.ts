import { DedalusAPI, TrackingPlan } from './services/DedalusAPI';
import { navigateToDashboard } from './utils/navigation';
import { initMeshBackground } from './components/MeshBackground';
import './styles/glass-ui.css';
import './styles/plan.css';

type TrackingSource = 'live' | 'demo';

interface DemoVideo {
  video_id: string;
  name: string;
  filename: string;
  thumbnail?: string | null;
  fps?: number | null;
  duration_sec?: number | null;
}

let currentPlan: TrackingPlan | null = null;
let selectedObjects: Set<string> = new Set();
let trackingSource: TrackingSource = 'live';
let selectedDemoVideoId: string | null = null;
let demoVideos: DemoVideo[] = [];

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Plan page loaded');

  // Initialize background mesh animation
  const bgCanvas = document.getElementById('meshBg') as HTMLCanvasElement;
  if (bgCanvas) initMeshBackground(bgCanvas);

  const params = new URLSearchParams(window.location.search);
  const prompt = params.get('prompt');

  if (!prompt) {
    showError();
    return;
  }

  try {
    // Show loading state
    showLoading();

    // Step 1: Extract objects
    const objects = await DedalusAPI.extractObjects(prompt);
    console.log('Extracted objects:', objects);

    // Step 2: Generate plan with justifications
    const plan = await DedalusAPI.generatePlan(prompt, objects);
    console.log('Generated plan:', plan);

    // if the word DEMO is in the prompt, hardcode the objects to be "chair", "brick", "table"
    if (prompt.toUpperCase().includes('DEMO')) {
      plan.objects = ['chair', 'brick', 'table', 'bottle'];
    }

    currentPlan = plan;
    selectedObjects = new Set(plan.objects);

    // Step 3: Display plan
    displayPlan(plan);
  } catch (error) {
    console.error('Failed to generate plan:', error);
    showError();
  }
});

function showLoading(): void {
  document.getElementById('loadingState')!.style.display = 'flex';
  document.getElementById('planContent')!.style.display = 'none';
  document.getElementById('errorState')!.style.display = 'none';
}

function showError(): void {
  document.getElementById('loadingState')!.style.display = 'none';
  document.getElementById('planContent')!.style.display = 'none';
  document.getElementById('errorState')!.style.display = 'flex';

  document.getElementById('retryBtn')?.addEventListener('click', () => {
    window.location.reload();
  });
}

function displayPlan(plan: TrackingPlan): void {
  document.getElementById('loadingState')!.style.display = 'none';
  document.getElementById('planContent')!.style.display = 'block';

  // Render objects with checkboxes
  renderObjects(plan.objects);

  // Render justifications
  document.getElementById('waypointsJustification')!.textContent = plan.waypoints.justification;
  document.getElementById('pathfindingJustification')!.textContent = plan.pathfinding.justification;

  // Setup event handlers
  setupAddObjectHandler();
  setupSourceSelector();
  void loadDemoVideos();
  setupConfirmHandler();
}

function renderObjects(objects: string[]): void {
  const container = document.getElementById('objectsList')!;
  container.innerHTML = '';

  objects.forEach(obj => {
    const chip = document.createElement('div');
    chip.className = 'object-chip';
    chip.innerHTML = `
      <span>${obj}</span>
      <button class="remove-btn" data-object="${obj}">×</button>
    `;

    chip.querySelector('.remove-btn')?.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const objectName = target.dataset.object!;
      selectedObjects.delete(objectName);
      chip.remove();
    });

    container.appendChild(chip);
  });
}

function setupAddObjectHandler(): void {
  const input = document.getElementById('customObjectInput') as HTMLInputElement;
  const btn = document.getElementById('addObjectBtn')!;

  const addObject = () => {
    const objectName = input.value.trim();
    if (objectName && !selectedObjects.has(objectName)) {
      selectedObjects.add(objectName);
      renderObjects(Array.from(selectedObjects));
      input.value = '';
    }
  };

  btn.addEventListener('click', addObject);
  input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addObject();
  });
}

function setupConfirmHandler(): void {
  const btn = document.getElementById('confirmBtn')!;

  btn.addEventListener('click', () => {
    if (selectedObjects.size === 0) {
      alert('Please select at least one object to track');
      return;
    }
    if (trackingSource === 'demo' && !selectedDemoVideoId) {
      alert('Please select a demo video');
      return;
    }

    navigateToDashboard({
      objects: Array.from(selectedObjects),
      waypoints: true,
      pathfinding: true,
      trackingSource,
      demoVideoId: trackingSource === 'demo' ? selectedDemoVideoId ?? undefined : undefined,
    });
  });
}

function setupSourceSelector(): void {
  const liveBtn = document.getElementById('liveSourceBtn') as HTMLButtonElement;
  const demoBtn = document.getElementById('demoSourceBtn') as HTMLButtonElement;

  liveBtn.addEventListener('click', () => setTrackingSource('live'));
  demoBtn.addEventListener('click', () => setTrackingSource('demo'));
  setTrackingSource('live');
}

function setTrackingSource(source: TrackingSource): void {
  trackingSource = source;
  const liveBtn = document.getElementById('liveSourceBtn') as HTMLButtonElement;
  const demoBtn = document.getElementById('demoSourceBtn') as HTMLButtonElement;
  const demoSection = document.getElementById('demoVideosSection') as HTMLDivElement;
  liveBtn.classList.toggle('selected', source === 'live');
  demoBtn.classList.toggle('selected', source === 'demo');
  demoSection.style.display = source === 'demo' ? 'block' : 'none';
}

async function loadDemoVideos(): Promise<void> {
  const loadingEl = document.getElementById('demoVideosLoading') as HTMLDivElement;
  const emptyEl = document.getElementById('demoVideosEmpty') as HTMLDivElement;

  loadingEl.style.display = 'block';
  emptyEl.style.display = 'none';
  try {
    const response = await fetch('/api/demo/videos');
    if (!response.ok) {
      throw new Error(`Failed to fetch demo videos (${response.status})`);
    }
    const data = await response.json();
    demoVideos = Array.isArray(data.videos) ? data.videos : [];
    renderDemoVideos();
    if (demoVideos.length === 0) {
      emptyEl.style.display = 'block';
    }
  } catch (error) {
    console.error('Failed to load demo videos:', error);
    emptyEl.textContent = 'Failed to load demo videos';
    emptyEl.style.display = 'block';
  } finally {
    loadingEl.style.display = 'none';
  }
}

function renderDemoVideos(): void {
  const listEl = document.getElementById('demoVideosList') as HTMLDivElement;
  listEl.innerHTML = '';

  for (const video of demoVideos) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'demo-video-card';
    if (video.video_id === selectedDemoVideoId) {
      card.classList.add('selected');
    }

    const thumb = video.thumbnail
      ? document.createElement('img')
      : document.createElement('div');
    if (thumb instanceof HTMLImageElement) {
      thumb.className = 'demo-video-thumb';
      thumb.src = video.thumbnail;
      thumb.alt = `${video.name} thumbnail`;
    } else {
      thumb.className = 'demo-video-thumb';
      thumb.textContent = 'No preview';
      thumb.style.display = 'grid';
      thumb.style.placeItems = 'center';
      thumb.style.fontSize = '12px';
      thumb.style.color = 'var(--text-secondary)';
    }

    const meta = document.createElement('div');
    meta.className = 'demo-video-meta';

    const name = document.createElement('div');
    name.className = 'demo-video-name';
    name.textContent = video.name || video.filename || video.video_id;
    meta.appendChild(name);

    const details = document.createElement('div');
    details.className = 'demo-video-details';
    details.textContent = formatDemoVideoDetails(video);
    meta.appendChild(details);

    card.appendChild(thumb);
    card.appendChild(meta);
    card.addEventListener('click', () => {
      selectedDemoVideoId = video.video_id;
      renderDemoVideos();
    });

    listEl.appendChild(card);
  }
}

function formatDemoVideoDetails(video: DemoVideo): string {
  const parts: string[] = [];
  if (video.duration_sec && Number.isFinite(video.duration_sec)) {
    parts.push(`${Math.round(video.duration_sec)}s`);
  }
  if (video.fps && Number.isFinite(video.fps)) {
    parts.push(`${Math.round(video.fps)} fps`);
  }
  if (parts.length === 0) {
    return video.filename || 'Demo video';
  }
  return parts.join(' | ');
}
