export function navigateToPlan(prompt: string): void {
  const encoded = encodeURIComponent(prompt);
  window.location.href = `/plan.html?prompt=${encoded}`;
}

export function navigateToDashboard(plan: {
  objects: string[];
  waypoints: boolean;
  pathfinding: boolean;
  trackingSource: 'live' | 'demo';
  demoVideoId?: string;
}): void {
  // Store in sessionStorage for dashboard to read
  sessionStorage.setItem('trackedObjects', JSON.stringify(plan.objects));
  sessionStorage.setItem('waypointsEnabled', plan.waypoints.toString());
  sessionStorage.setItem('pathfindingEnabled', plan.pathfinding.toString());
  sessionStorage.setItem('trackingSource', plan.trackingSource);
  if (plan.demoVideoId) {
    sessionStorage.setItem('demoVideoId', plan.demoVideoId);
  } else {
    sessionStorage.removeItem('demoVideoId');
  }

  // Keep mode in URL so refreshes preserve demo UI state.
  const url = new URL('/viewer.html', window.location.origin);
  url.searchParams.set('source', 'plan');
  if (plan.trackingSource === 'demo') {
    url.searchParams.set('mode', 'demo');
    if (plan.demoVideoId) {
      url.searchParams.set('video_id', plan.demoVideoId);
    }
  }
  window.location.href = `${url.pathname}${url.search}`;
}
