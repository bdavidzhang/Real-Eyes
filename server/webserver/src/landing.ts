import { HeroScene } from './components/HeroScene';
import { initMeshBackground } from './components/MeshBackground';
import { navigateToPlan } from './utils/navigation';
import './styles/glass-ui.css';
import './styles/landing.css';

interface DemoVideo {
  video_id: string;
  name: string;
  thumbnail?: string | null;
  filename?: string;
}

const DEMO_PROMPT_BY_VIDEO: Record<string, string> = {
  'office_loop.mp4':
    'DEMO: Act as a facilities survey agent inside a university research building. ' +
    'Walk through the hallway and office areas assessing workspace setup, equipment placement, ' +
    'and safety infrastructure. Identify and track key items like whiteboards, waste bins, ' +
    'lab/kitchen appliances, workstations, access control devices, and emergency equipment. ' +
    'Also note transitions in environment (e.g., wall colors or layout changes) and explore ' +
    'any additional objects relevant to office operations or safety.',

  'house.MOV':
    'DEMO: Act as a residential environment surveyor exploring a home from entryway to kitchen. ' +
    'Assess how spaces are organized and how objects are distributed across living, dining, ' +
    'and kitchen areas. Track representative items such as seating, tables, storage areas, ' +
    'appliances, food-related objects, and safety equipment. Identify clutter, high-use surfaces, ' +
    'and anything notable about room function or layout. Explore and document other relevant objects you encounter.',

  'house2.MOV':
    'DEMO: Perform a secondary interior survey of a residential home focusing on layout, usage patterns, ' +
    'and object distribution. Track common furniture, electronics, storage units, and kitchen appliances. ' +
    'Pay attention to how personal items and consumables are arranged. Identify patterns of occupancy ' +
    'and explore additional objects that help characterize how the space is used.',

  'crime_scene.mov':
    'DEMO: Act as a forensic survey agent examining a staged indoor crime scene. ' +
    'Document spatial relationships between evidence markers and surrounding objects. ' +
    'Track signs of disturbance such as spills, broken items, footprints, displaced furniture, ' +
    'or unusual object placements. Explore beyond predefined evidence to identify anything ' +
    'potentially relevant to reconstructing events or understanding the scene context.',

  'disaster.mov':
    'DEMO: Act as a disaster response surveyor assessing damage in a flooded and earthquake-affected ' +
    'residential street. Evaluate structural damage, displaced household objects, infrastructure failure, ' +
    'and hazards in the roadway. Track representative debris, vehicles, utilities, and structural elements. ' +
    'Also explore additional signs of environmental instability or safety risks as you navigate the area.',

  'disaster2.mov':
    'DEMO: Act as an urban damage assessment agent surveying an earthquake aftermath scene. ' +
    'Assess building integrity, debris distribution, fallen infrastructure, and abandoned objects. ' +
    'Track major structural failures and unusual displaced items. Identify hazards, blocked pathways, ' +
    'and environmental conditions such as dust or water. Explore additional features relevant to recovery planning.',

  'hackathon_loop.MOV':
    'DEMO: Act as an event environment analysis agent surveying a university hackathon venue. ' +
    'Assess crowd activity, workspace setup, seating distribution, and structural layout. ' +
    'Track representative furniture, collaborative work areas, staircases, railings, and trash receptacles. ' +
    'Explore additional objects that indicate event logistics, attendee behavior, or space utilization.',

  'our_workspace.MOV':
    'DEMO: Act as a workspace activity observer analyzing a hackathon team area. ' +
    'Track laptops, seating, personal belongings, food items, and collaborative tools such as whiteboards. ' +
    'Assess workspace density, object clustering, and signs of active development. ' +
    'Explore additional objects that help characterize productivity, organization, or team dynamics.',
};

document.addEventListener('DOMContentLoaded', () => {
  console.log('Open Reality — landing page loaded');

  const meshCanvas = document.getElementById('meshBg') as HTMLCanvasElement;
  if (meshCanvas) initMeshBackground(meshCanvas);

  const canvas = document.getElementById('hero-3d-canvas') as HTMLDivElement;
  const heroScene = new HeroScene(canvas);

  const landingContent = document.querySelector('.landing-content') as HTMLElement | null;
  const landingOverlay = document.querySelector('.landing-overlay') as HTMLElement | null;
  let interacted = false;
  const onFirstInteract = () => {
    if (interacted) return;
    interacted = true;
    heroScene.startInteraction();
    if (landingContent) {
      landingContent.classList.add('active');
    }
    if (landingOverlay) {
      landingOverlay.classList.add('active');
    }
  };

  canvas.addEventListener('pointerdown', onFirstInteract, { once: true });
  canvas.addEventListener('touchstart', onFirstInteract, { once: true });

  const form = document.getElementById('promptForm') as HTMLFormElement;
  const input = document.getElementById('promptInput') as HTMLTextAreaElement;
  const demoGrid = document.getElementById('demoVideoGrid') as HTMLDivElement | null;
  const demoStartBtn = document.getElementById('demoStartBtn') as HTMLButtonElement | null;
  let demoVideos: DemoVideo[] = [];
  let selectedDemoVideoId: string | null = null;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const prompt = input.value.trim();

    if (prompt.length < 10) {
      alert('Please provide a more detailed description');
      return;
    }

    navigateToPlan(prompt);
  });

  const getDemoPrompt = (videoId: string, label: string): string => {
    return DEMO_PROMPT_BY_VIDEO[videoId]
      || `DEMO: Explore scene from ${label} and track furniture, containers, and navigation landmarks.`;
  };

  const loadDemoVideos = async () => {
    if (!demoGrid || !demoStartBtn) return;
    try {
      const response = await fetch('/api/demo/videos');
      if (!response.ok) {
        throw new Error(`Failed to fetch demo videos (${response.status})`);
      }
      const data = await response.json();
      demoVideos = Array.isArray(data.videos) ? data.videos : [];
      renderDemoCards();
      demoStartBtn.disabled = demoVideos.length === 0;
    } catch (error) {
      console.error('Failed to load demo videos:', error);
      demoStartBtn.disabled = true;
    }
  };

  const renderDemoCards = () => {
    if (!demoGrid) return;
    demoGrid.innerHTML = '';
    for (const video of demoVideos) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'demo-thumb-btn';
      if (video.video_id === selectedDemoVideoId) {
        btn.classList.add('selected');
      }

      const thumb = document.createElement('img');
      thumb.className = 'demo-thumb';
      thumb.alt = `${video.name} thumbnail`;
      thumb.src = video.thumbnail || '';
      btn.appendChild(thumb);

      const name = document.createElement('div');
      name.className = 'demo-thumb-name';
      name.textContent = video.name || video.filename || video.video_id;
      btn.appendChild(name);

      btn.addEventListener('click', () => {
        selectedDemoVideoId = video.video_id;
        renderDemoCards();
      });
      demoGrid.appendChild(btn);
    }
  };

  demoStartBtn?.addEventListener('click', () => {
    if (!selectedDemoVideoId) {
      alert('Please select a demo video first');
      return;
    }
    const selectedVideo = demoVideos.find((v) => v.video_id === selectedDemoVideoId);
    const videoId = selectedDemoVideoId;
    const label = selectedVideo?.name || selectedVideo?.filename || videoId;
    const demoPrompt = getDemoPrompt(videoId, label);
    navigateToPlan(demoPrompt, {
      trackingSource: 'demo',
      demoVideoId: videoId,
    });
  });

  void loadDemoVideos();
});
