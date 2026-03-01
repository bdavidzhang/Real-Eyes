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
let typewriterTimer: number | null = null;
let introTypewriterTimer: number | null = null;

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

const DEMO_OBJECTS_BY_VIDEO: Record<string, string[]> = {
  'office_loop.mp4': ['whiteboard', 'recycling bin', 'trash can', 'printer', 'refrigerator', 'coffee machine', 'monitor', 'door'],
  'house.MOV': ['couch', 'table', 'laptop', 'shoe', 'bottle', 'chair', 'plant', 'sink'],
  'house2.MOV': ['couch', 'laptop', 'plant', 'table', 'chair', 'bottle',  'cabinet'],
  'crime_scene.mov': ['evidence marker', 'footprint', 'broken glass', 'eyeglasses', 'bookshelf', 'laptop', 'dining table', 'chair', 'lamp', 'carpet stain'],
  'disaster.mov': ['table', 'yellow boots', 'bicycle', 'car', 'mattress', 'utility pole', 'bathtub', 'shopping cart', 'debris pile'],
  'disaster2.mov': ['barber chair', 'piano', 'streetlight', 'office chair', 'traffic light', 'monitor', 'concrete slab', 'puddle'],
  'hackathon_loop.MOV': ['staircase', 'bench', 'trash can', 'laptop', 'table', 'backpack', 'chair'],
  'our_workspace.MOV': ['laptop', 'chair', 'bottle', 'backpack', 'human', 'table'],
};

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Open Reality — mission brief loaded');

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
    showLoading();
    setPipelineStage(1);

    const objects = await DedalusAPI.extractObjects(prompt);
    console.log('Extracted objects:', objects);

    const plan = await DedalusAPI.generatePlan(prompt, objects);
    console.log('Generated plan:', plan);

    setPipelineStage(2);

    if (prompt.toUpperCase().includes('DEMO')) {
      const videoObjects = selectedDemoVideoId ? DEMO_OBJECTS_BY_VIDEO[selectedDemoVideoId] : null;
      plan.objects = videoObjects ?? ['chair', 'table', 'bottle', 'backpack', 'door'];
    }
    plan.objects = ensureMinimumObjects(plan.objects, 5);

    selectedObjects = new Set(plan.objects);

    displayPlan(plan);
    setPipelineStage(3);
  } catch (error) {
    console.error('Failed to generate plan:', error);
    showError();
  }
});

/* ── Pipeline ── */

function setPipelineStage(stage: number): void {
  for (let i = 1; i <= 3; i++) {
    const el = document.getElementById(`pipelineStep${i}`);
    if (!el) continue;
    el.classList.remove('active', 'completed');
    if (i < stage) el.classList.add('completed');
    if (i === stage) el.classList.add('active');
  }
}

/* ── Loading / Error ── */

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
  typewriteAgentLine('Signal interrupted. Re-run mission compile and I will recover.');

  document.getElementById('retryBtn')?.addEventListener('click', () => {
    window.location.reload();
  });
}

/* ── Display Plan ── */

function displayPlan(plan: TrackingPlan): void {
  stopLoadingLineRotation();
  document.getElementById('loadingState')!.style.display = 'none';
  document.getElementById('planContent')!.style.display = 'block';

  const introBox = document.getElementById('agentIntroBox');
  if (introBox) introBox.style.display = 'block';
  typewriteIntroLine(plan.agent_intro ?? buildDefaultIntro());

  renderObjects(plan.objects);
  typewriteAgentLine(buildLaunchLine(plan.objects));

  setupAddObjectHandler();
  setupConfirmHandler();
}

/* ── Object Chips with Staggered Animation ── */

function renderObjects(objects: string[]): void {
  const container = document.getElementById('objectsList')!;
  container.innerHTML = '';

  objects.forEach((obj, index) => {
    const chip = document.createElement('div');
    chip.className = 'object-chip';
    chip.style.animationDelay = `${index * 60}ms`;
    chip.innerHTML = `
      <span>${obj}</span>
      <button class="remove-btn" data-object="${obj}">\u00d7</button>
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

/* ── Agent Voice — Typewriter Effect ── */

function setupAgentVoice(): void {
  const agentLineEl = document.getElementById('agentLine');
  if (!agentLineEl) return;
  typewriteAgentLine(AGENT_LINES[0]);

  let lineIndex = 0;
  setInterval(() => {
    lineIndex = (lineIndex + 1) % AGENT_LINES.length;
    typewriteAgentLine(AGENT_LINES[lineIndex]);
  }, 6000);
}

function typewriteAgentLine(message: string): void {
  if (typewriterTimer !== null) {
    clearInterval(typewriterTimer);
    typewriterTimer = null;
  }

  const el = document.getElementById('agentLine');
  if (!el) return;

  el.classList.remove('typing-cursor');
  el.textContent = '';

  let charIndex = 0;
  el.classList.add('typing-cursor');

  typewriterTimer = window.setInterval(() => {
    if (charIndex < message.length) {
      el.textContent = message.slice(0, charIndex + 1);
      charIndex++;
    } else {
      if (typewriterTimer !== null) {
        clearInterval(typewriterTimer);
        typewriterTimer = null;
      }
    }
  }, 28);
}

function typewriteIntroLine(message: string): void {
  if (introTypewriterTimer !== null) {
    clearInterval(introTypewriterTimer);
    introTypewriterTimer = null;
  }

  const el = document.getElementById('agentIntroLine');
  if (!el) return;

  el.classList.remove('typing-cursor');
  el.textContent = '';

  let charIndex = 0;
  el.classList.add('typing-cursor');

  introTypewriterTimer = window.setInterval(() => {
    if (charIndex < message.length) {
      el.textContent = message.slice(0, charIndex + 1);
      charIndex++;
    } else {
      el.classList.remove('typing-cursor');
      if (introTypewriterTimer !== null) {
        clearInterval(introTypewriterTimer);
        introTypewriterTimer = null;
      }
    }
  }, 28);
}

function buildDefaultIntro(): string {
  return 'I will scan the scene and lock onto all high-value targets in the environment.';
}

/* ── Utilities ── */

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

function buildLaunchLine(objects: string[]): string {
  if (objects.length === 0) return 'No targets locked yet. Add objects and I will spin up the sweep.';
  if (objects.length === 1) return `Target locked: ${objects[0]}. I am ready to track.`;
  if (objects.length === 2) return `Targets locked: ${objects[0]} + ${objects[1]}. Ready to execute.`;
  return `Targets locked: ${objects[0]} + ${objects[1]}. ${objects.length - 2} more queued. Systems hot.`;
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
