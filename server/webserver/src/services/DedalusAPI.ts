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
  private static apiKey = ((import.meta as any).env as any).VITE_DEDALUS_API_KEY;
  private static endpoint = 'https://api.dedaluslabs.ai/v1/chat/completions';

  private static ensureApiKey(): void {
    if (!this.apiKey) {
      throw new Error(
        'VITE_DEDALUS_API_KEY is not set. Create a .env.local file in the webserver directory with:\nVITE_DEDALUS_API_KEY=your-key-here'
      );
    }
  }

  /**
   * Extract trackable objects from user prompt
   */
  static async extractObjects(prompt: string): Promise<string[]> {
    this.ensureApiKey();
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: 'Extract trackable physical objects from the user prompt. Return JSON: {"objects": ["obj1", "obj2", ...]}. Focus on concrete, visible items.',
          },
          { role: 'user', content: prompt }
        ],
        temperature: 0.0,
        max_tokens: 300,
      }),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`API error ${response.status}: ${errBody}`);
    }

    const data = await response.json();
    if (!data.choices?.length) {
      throw new Error(`Unexpected API response: ${JSON.stringify(data)}`);
    }
    const content = data.choices[0].message.content;
    const parsed = JSON.parse(content.match(/\{[\s\S]*\}/)[0]);
    return parsed.objects || [];
  }

  /**
   * Generate personalized tracking plan with justifications
   * Note: Waypoints and pathfinding are ALWAYS enabled
   */
  static async generatePlan(prompt: string, objects: string[]): Promise<TrackingPlan> {
    this.ensureApiKey();
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: `Generate a spatial tracking plan. Return JSON:
{
  "objects": ["obj1", "obj2", ...],
  "waypoints": {
    "enabled": true,
    "justification": "1-2 sentence explanation specific to their use case"
  },
  "pathfinding": {
    "enabled": true,
    "justification": "1-2 sentence explanation specific to their use case"
  }
}

Waypoints help mark key locations. Pathfinding shows traversed paths on a minimap. Tailor justifications to the user's specific scenario.`,
          },
          {
            role: 'user',
            content: `User scenario: "${prompt}"\nDetected objects: ${objects.join(', ')}\n\nGenerate personalized justifications for why waypoints and pathfinding would help in this scenario.`
          }
        ],
        temperature: 0.3,
        max_tokens: 512,
      }),
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(`API error ${response.status}: ${errBody}`);
    }

    const data = await response.json();
    if (!data.choices?.length) {
      throw new Error(`Unexpected API response: ${JSON.stringify(data)}`);
    }
    const content = data.choices[0].message.content;
    const parsed = JSON.parse(content.match(/\{[\s\S]*\}/)[0]);

    return {
      objects: parsed.objects || objects,
      waypoints: parsed.waypoints || {
        enabled: true,
        justification: 'Waypoints help you mark and share key locations with your team.',
      },
      pathfinding: parsed.pathfinding || {
        enabled: true,
        justification: 'Pathfinding visualizes your traversed route on a minimap for navigation.',
      },
    };
  }
}
