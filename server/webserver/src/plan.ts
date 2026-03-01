import { DedalusAPI, TrackingPlan } from './services/DedalusAPI';
import { navigateToDashboard } from './utils/navigation';
import { initMeshBackground } from './components/MeshBackground';
import './styles/glass-ui.css';
import './styles/plan.css';

type TrackingSource = 'live' | 'demo';

let selectedObjects: Set<string> = new Set();
let trackingSource: TrackingSource = 'live';
let selectedDemoVideoId: string | null = null;
let loadingLineTimer: number | null = null;

const LOADING_LINES = [
  'Calibrating scene intelligence...',
  'Stack-ranking likely targets...',
  'Tuning object filters for zero-noise lock...',
  'Preparing high-confidence pursuit profile...',
];

const AGENT_LINES = [
  'I chase signal, not noise. Give me targets and I will hunt.',
  'Mission feed is live. Every frame is evidence.',
  'Target confidence is dynamic. I adapt before the scene does.',
  'No blind spots. No drift. Just tracked reality.',
  'You choose the objective. I execute the sweep.',
];

const PLAN_FALLBACK_OBJECTS = ['person', 'chair', 'table', 'bottle', 'backpack', 'door'];

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Plan page loaded');

  // Initialize background mesh animation
  const bgCanvas = document.getElementById('meshBg') as HTMLCanvasElement;
  if (bgCanvas) initMeshBackground(bgCanvas);
  setupAgentVoice();

  const params = new URLSearchParams(window.location.search);
  const prompt = params.get('prompt');
  sessionStorage.setItem('userPrompt', prompt ?? '');
  trackingSource = params.get('mode') === 'demo' ? 'demo' : 'live';
  selectedDemoVideoId = params.get('video_id');
  setupCameraQr();

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

    // Step 2: Generate plan
    const plan = await DedalusAPI.generatePlan(prompt, objects);
    console.log('Generated plan:', plan);

    // if the word DEMO is in the prompt, hardcode the objects to be "chair", "brick", "table"
    if (prompt.toUpperCase().includes('DEMO')) {
      plan.objects = ['chair', 'brick', 'table', 'bottle', 'backpack', 'door'];
    }
    plan.objects = ensureMinimumObjects(plan.objects, 5);

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

  const loadingLineEl = document.getElementById('loadingLine');
  if (loadingLineEl && loadingLineTimer === null) {
    loadingLineTimer = rotateLines(loadingLineEl, LOADING_LINES, 1700);
  }
}

function showError(): void {
  stopLoadingLineRotation();
  document.getElementById('loadingState')!.style.display = 'none';
  document.getElementById('planContent')!.style.display = 'none';
  document.getElementById('errorState')!.style.display = 'flex';
  setAgentLine('Signal interrupted. Re-run mission compile and I will recover.');

  document.getElementById('retryBtn')?.addEventListener('click', () => {
    window.location.reload();
  });
}

function displayPlan(plan: TrackingPlan): void {
  stopLoadingLineRotation();
  document.getElementById('loadingState')!.style.display = 'none';
  document.getElementById('planContent')!.style.display = 'block';

  // Render selected objects
  renderObjects(plan.objects);
  setAgentLine(buildLaunchLine(plan.objects));

  // Setup event handlers
  setupAddObjectHandler();
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

function setupAgentVoice(): void {
  const agentLineEl = document.getElementById('agentLine');
  if (!agentLineEl) return;
  setAgentLine(AGENT_LINES[0]);
  rotateLines(agentLineEl, AGENT_LINES, 4200);
}

function rotateLines(el: HTMLElement, lines: string[], intervalMs: number): number {
  let index = Math.floor(Math.random() * lines.length);
  el.textContent = lines[index];
  return window.setInterval(() => {
    index = (index + 1) % lines.length;
    el.textContent = lines[index];
  }, intervalMs);
}

function stopLoadingLineRotation(): void {
  if (loadingLineTimer !== null) {
    window.clearInterval(loadingLineTimer);
    loadingLineTimer = null;
  }
}

function setAgentLine(message: string): void {
  const el = document.getElementById('agentLine');
  if (el) el.textContent = message;
}

function buildLaunchLine(objects: string[]): string {
  if (objects.length === 0) return 'No targets locked yet. Add objects and I will spin up the sweep.';
  if (objects.length === 1) return `Target locked: ${objects[0]}. I am ready to track.`;
  if (objects.length === 2) return `Targets locked: ${objects[0]} + ${objects[1]}. Ready to execute.`;
  return `Targets locked: ${objects[0]} + ${objects[1]}. ${objects.length - 2} more queued.`;
}

function setupCameraQr(): void {
  const qrImage = document.getElementById('cameraQrImage') as HTMLImageElement | null;
  const cameraLink = document.getElementById('cameraPageLink') as HTMLAnchorElement | null;
  const hint = document.getElementById('cameraQrHint');
  if (!qrImage || !cameraLink || !hint) return;

  const cameraUrl = buildCameraUrl();
  const cameraUrlText = cameraUrl.toString();
  const qrSrc = `https://quickchart.io/qr?size=300&margin=2&ecLevel=M&text=${encodeURIComponent(cameraUrlText)}`;

  qrImage.src = qrSrc;
  qrImage.loading = 'lazy';
  cameraLink.href = cameraUrlText;
  cameraLink.textContent = cameraUrlText;

  if (cameraUrl.hostname === 'localhost' || cameraUrl.hostname === '127.0.0.1' || cameraUrl.hostname === '::1') {
    hint.textContent = 'This link uses localhost. Replace it with your laptop LAN IP if your phone cannot reach it.';
    return;
  }

  hint.textContent = 'If scan fails, open the link directly on your phone browser.';
}

function buildCameraUrl(): URL {
  const url = new URL('/sender.html', window.location.origin);
  if (trackingSource === 'demo') {
    url.searchParams.set('mode', 'demo');
    if (selectedDemoVideoId) {
      url.searchParams.set('video_id', selectedDemoVideoId);
    }
  }
  return url;
}

function ensureMinimumObjects(objects: string[], minCount: number): string[] {
  const merged = new Set(
    objects
      .map((obj) => obj.trim().toLowerCase())
      .filter((obj) => obj.length > 0 && obj !== 'object' && obj !== 'objects' && obj !== 'item'),
  );

  for (const fallback of PLAN_FALLBACK_OBJECTS) {
    if (merged.size >= minCount) break;
    merged.add(fallback);
  }

  return Array.from(merged).slice(0, 8);
}
