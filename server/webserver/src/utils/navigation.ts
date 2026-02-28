export function navigateToPlan(prompt: string): void {
  const encoded = encodeURIComponent(prompt);
  window.location.href = `/plan.html?prompt=${encoded}`;
}

export function navigateToDashboard(plan: {
  objects: string[];
  waypoints: boolean;
  pathfinding: boolean;
}): void {
  // Store in sessionStorage for dashboard to read
  sessionStorage.setItem('trackedObjects', JSON.stringify(plan.objects));
  sessionStorage.setItem('waypointsEnabled', plan.waypoints.toString());
  sessionStorage.setItem('pathfindingEnabled', plan.pathfinding.toString());

  // Navigate with query param to signal data is available
  window.location.href = '/viewer.html?source=plan';
}
