export interface TrackingPlan {
  objects: string[];
  waypoints: {
    enabled: true;
    justification: string;
  };
  pathfinding: {
    enabled: true;
    justification: string;
  };
}

export class DedalusAPI {
  // Cache so extractObjects + generatePlan only hits the server once per prompt
  private static _cache = new Map<string, TrackingPlan>();

  private static async _fetchPlan(prompt: string): Promise<TrackingPlan> {
    if (this._cache.has(prompt)) {
      return this._cache.get(prompt)!;
    }

    const response = await fetch(`${window.location.origin}/api/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`Plan API error ${response.status}: ${errBody}`);
    }

    const plan = (await response.json()) as TrackingPlan;
    this._cache.set(prompt, plan);
    return plan;
  }

  static async extractObjects(prompt: string): Promise<string[]> {
    const plan = await this._fetchPlan(prompt);
    return plan.objects;
  }

  static async generatePlan(prompt: string, _objects: string[]): Promise<TrackingPlan> {
    return this._fetchPlan(prompt);
  }
}
