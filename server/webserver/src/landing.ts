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

// Editable prompts per demo video id.
const DEMO_PROMPT_BY_VIDEO: Record<string, string> = {
  'office_loop.mp4': 'DEMO: Navigate an office hallway and track chair, table, bottle, and brick-like objects.',
};

document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 Landing page loaded');

  // Initialize mesh background
  const meshCanvas = document.getElementById('meshBg') as HTMLCanvasElement;
  if (meshCanvas) initMeshBackground(meshCanvas);

  // Initialize 3D background with auto-rotate
  const canvas = document.getElementById('hero-3d-canvas') as HTMLDivElement;
  const heroScene = new HeroScene(canvas);

  // Slide content and activate 3D interaction on first pointer/touch
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

  // Pointerdown covers mouse and pen; add touchstart for mobile.
  canvas.addEventListener('pointerdown', onFirstInteract, { once: true });
  canvas.addEventListener('touchstart', onFirstInteract, { once: true });

  // Enable auto-rotate (already exists in HeroScene.ts)
  // The scene automatically rotates slowly in the background

  // Handle prompt submission
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
