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
    'DEMO: Survey a university research lab office. Walk down a long carpeted hallway lined with ' +
    'cubicle partitions with frosted glass panels. Detect and track: whiteboards with writing, ' +
    'blue recycling bins, black trash cans, metal coat rack stands, a printer/copier on a filing cabinet, ' +
    'a stainless steel refrigerator with magnets, a kitchen counter with sink and coffee machine, ' +
    'research poster boards, glass and wooden doors, computer monitors at desks, a wall-mounted card reader, ' +
    'and a fire alarm pull station. The hallway transitions from white walls to blue walls midway through.',

  'house.MOV':
    'DEMO: Explore a residential house from entryway through living room, dining room, and kitchen. ' +
    'Start at the front door with a shoe rack holding multiple pairs of sneakers and sandals on a rubber mat, ' +
    'a wooden bench, and a clear plastic storage bin. In the living room find a fabric couch with pillows, ' +
    'a wooden coffee table with a laptop, water bottle, baseball cap, and orange juice bottle. ' +
    'The dining room has a large glass-top wooden table with chairs, a purple water filter pitcher, ' +
    'a white water dispenser, and a bottle of hot sauce. A potted plant sits on a wooden dresser. ' +
    'The kitchen has a white farmhouse sink by a window, blue dishes in a dish rack, red rubber gloves, ' +
    'green recycling bins under the sink, an electric stove with a kettle and frying pan, ' +
    'a fire extinguisher on the wall, a microwave on a shelf, and a coffee maker.',

  'house2.MOV':
    'DEMO: Walk through a second angle of a residential house focusing on living room and kitchen. ' +
    'Start at a wall-mounted black file/mail organizer by a window with blinds. In the living room ' +
    'find a dark leather couch/recliner with a laptop, a blanket, and a phone. A wooden side table ' +
    'holds papers next to a decorative plant on a plant stand. Water bottle packs and beer bottles ' +
    'sit on the floor. Wall text decor reads "House". The dining area has the same glass-top table ' +
    'with a purple water pitcher, white dispenser, and wooden chairs. The kitchen has white cabinets, ' +
    'a black air fryer, a Keurig-style coffee maker, a paper towel holder on the wall, ' +
    'a metal storage shelf with white bins full of snacks and groceries, a wire dish rack, ' +
    'and pots and pans on the counter.',

  'crime_scene.mov':
    'DEMO: Investigate a staged indoor crime scene in a sunlit apartment. The scene has numbered ' +
    'yellow evidence markers (1 through 8) placed on the floor and carpet. Track and locate: ' +
    'muddy footprint trails on hardwood flooring, an overturned whiskey glass with spilled liquid, ' +
    'broken eyeglasses/sunglasses near marker 3, a plastic evidence bag, small round objects as evidence, ' +
    'a white carpet/rug with scattered items, a wooden bookshelf full of books, a laptop on a desk ' +
    'by the window with blinds, a dark wood dining table with upholstered chairs, a floor lamp, ' +
    'a metal rolling side table with a wooden tray and items underneath, and carpet stains near the evidence.',

  'disaster.mov':
    'DEMO: Navigate a flood and earthquake disaster zone in a Japanese residential neighborhood street. ' +
    'The narrow street is covered in thick mud and standing water. Detect and track: a displaced wooden ' +
    'dining table and chairs sitting in the middle of the flooded street, a pair of yellow rubber boots, ' +
    'a knocked-over red bicycle, a displaced mattress, a silver car half-buried under collapsed building debris, ' +
    'power lines and leaning utility poles, cracked and upheaved concrete pavement, scattered wood planks ' +
    'and cardboard, a white bathtub washed into the street, a metal shopping cart, collapsed residential ' +
    'building facades, metal roofing sheets, and miscellaneous household debris blocking the road.',

  'disaster2.mov':
    'DEMO: Survey an urban earthquake aftermath scene along a wide city boulevard flanked by tall ' +
    'damaged apartment buildings. The street is covered in gray dust and debris. Detect and track: ' +
    'a red barber/salon chair standing upright in the middle of the road, a grand piano partially buried ' +
    'in rubble, toppled metal streetlights and lamp posts, scattered office desk chairs, large puddles ' +
    'of standing water reflecting the sky, chunks of concrete and rebar, shattered glass panels on the ground, ' +
    'a red traffic light dangling from a wire in the distance, a TV/computer monitor in the debris, ' +
    'bent metal structural beams, thick dust haze between the buildings, and collapsed concrete slabs.',

  'hackathon_loop.MOV':
    'DEMO: Walk through a university hackathon venue with a multi-level atrium layout. The ground floor ' +
    'has a wide corridor with polished floors. Detect and track: a metal staircase with steel handrails, ' +
    'a metal wire mesh railing/fence overlooking a lower seating area, orange/red metal public benches, ' +
    'black hexagonal trash cans, large concrete support columns/pillars, people sitting at folding tables ' +
    'working on laptops, backpacks on the floor, white rolling chairs, overhead fluorescent lights, ' +
    'brick wall sections along the railing, and red/orange accent columns.',

  'our_workspace.MOV':
    'DEMO: Pan around a hackathon team workspace where multiple people are actively coding. ' +
    'Detect and track: multiple open laptops (MacBooks and others) showing code editors and 3D visualizations, ' +
    'orange ergonomic rolling chairs, light blue/mint rolling chairs, smartphones on tables, ' +
    'a black water bottle / hydro flask, bananas on the table, a fast food takeout bag, ' +
    'several backpacks on the floor, heavy winter puffer jackets draped on chairs, ' +
    'a rolling suitcase/luggage, white AirPods cases, charging cables, ' +
    'a whiteboard on the wall, a gray trash can with liner, and a jar of snacks/cereal.',
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
