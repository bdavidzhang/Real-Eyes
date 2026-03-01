/**
 * Lightweight canvas-based particle mesh background animation.
 * Renders drifting points connected by faint lines when close enough.
 */
export function initMeshBackground(canvas: HTMLCanvasElement): () => void {
  const ctx = canvas.getContext('2d')!;
  let animId: number;
  let width: number;
  let height: number;

  const POINT_COUNT = 60;
  const MAX_DIST = 160;
  const POINT_RADIUS = 1.5;
  const LINE_OPACITY = 0.12;
  const POINT_OPACITY = 0.35;
  const SPEED = 0.3;

  const TINT_COLORS = [
    '255, 255, 255',   // white
    '0, 240, 255',     // cyan
    '110, 168, 254',   // blue
    '168, 85, 247',    // violet
  ];

  interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    tint: string;
  }

  let particles: Particle[] = [];

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = [];
    for (let i = 0; i < POINT_COUNT; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * SPEED,
        vy: (Math.random() - 0.5) * SPEED,
        tint: TINT_COLORS[Math.random() < 0.6 ? 0 : Math.floor(Math.random() * TINT_COLORS.length)],
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    // Update positions
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;
    }

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < MAX_DIST) {
          const alpha = LINE_OPACITY * (1 - dist / MAX_DIST);
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${particles[i].tint}, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.tint}, ${POINT_OPACITY})`;
      ctx.fill();
    }

    animId = requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();

  window.addEventListener('resize', () => {
    resize();
    createParticles();
  });

  // Return a dispose function
  return () => {
    cancelAnimationFrame(animId);
  };
}
