import { HeroScene } from './components/HeroScene';
import { initMeshBackground } from './components/MeshBackground';
import { navigateToPlan } from './utils/navigation';
import './styles/glass-ui.css';
import './styles/landing.css';

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

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const prompt = input.value.trim();

    if (prompt.length < 10) {
      alert('Please provide a more detailed description');
      return;
    }

    navigateToPlan(prompt);
  });
});
