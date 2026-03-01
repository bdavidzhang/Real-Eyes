import { DedalusAPI, TrackingPlan } from './services/DedalusAPI';
import { navigateToDashboard } from './utils/navigation';
import { initMeshBackground } from './components/MeshBackground';
import './styles/glass-ui.css';
import './styles/plan.css';

type TrackingSource = 'live' | 'demo';

let currentPlan: TrackingPlan | null = null;
let selectedObjects: Set<string> = new Set();
let trackingSource: TrackingSource = 'live';
let selectedDemoVideoId: string | null = null;

document.addEventListener('DOMContentLoaded', async () => {
  console.log('Plan page loaded');

  // Initialize background mesh animation
  const bgCanvas = document.getElementById('meshBg') as HTMLCanvasElement;
  if (bgCanvas) initMeshBackground(bgCanvas);

  const params = new URLSearchParams(window.location.search);
  const prompt = params.get('prompt');
  sessionStorage.setItem('userPrompt', prompt ?? '');
  trackingSource = params.get('mode') === 'demo' ? 'demo' : 'live';
  selectedDemoVideoId = params.get('video_id');

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

