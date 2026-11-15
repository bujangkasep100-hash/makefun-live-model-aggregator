import { readFileSync } from 'node:fs';

export type Json = Record<string, unknown>;
export type Wire = 'openai' | 'anthropic';
export const INKLING_MODEL = 'openrouter/thinkingmachines/inkling-2-small:free';

export function isObject(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export type Model = {
  id: string;
  display_name: string;
  rating: number;
  context_length: number | null;
  tools: boolean;
  json: boolean;
  vision: boolean;
};

export function loadModels(briefPath: string | undefined): Model[] {
  if (!briefPath) return [];
  const source = readFileSync(briefPath, 'utf8');
  // The kit writes the catalog as one JSON object on its own line in the brief.
  const line = source.split(/\r?\n/).find((text) =>
    /^\s*\{\s*"schema"\s*:\s*"free-model-aggregator-02\/models\/v1"/.test(text),
  );
  const catalog = line ? JSON.parse(line) : null;
  if (!isObject(catalog) || !Array.isArray(catalog.models)) {
    throw new Error('The brief is missing its free model catalog');
  }

  const models = new Map<string, Model>();
  for (const row of catalog.models) {
    if (!isObject(row) || row.free !== true || typeof row.id !== 'string' ||
        !/^openrouter\/[^/]+\/[^/]+:free$/.test(row.id) || models.has(row.id)) continue;
    models.set(row.id, {
      id: row.id,
      display_name: typeof row.display_name === 'string' ? row.display_name : row.id,
      rating: typeof row.rating_milli === 'number' ? row.rating_milli : 0,
      context_length: typeof row.context_length === 'number' ? row.context_length : null,
      tools: row.tools === true,
      json: row.json === true,
      vision: row.vision === true,
    });
  }
  return [...models.values()];
}

export function catalogOf(models: Model[]): Json {
  return {
    providers: [{
      id: 'openrouter',
      base_url: 'https://openrouter.ai/api/v1',
      key_env: 'OPENROUTER_API_KEY',
      wire: 'openai',
      models: models.map(({ id, context_length }) => ({ id, free: true, context_length })),
    }],
  };
}

export function listModels(models: Model[], wire: Wire): Json {
  if (wire === 'openai') {
    return {
      object: 'list',
      data: models.map(({ id }) => ({ id, object: 'model', created: 0, owned_by: 'openrouter' })),
    };
  }
  return {
    data: models.map(({ id, display_name }) => ({
      id, display_name, type: 'model', created_at: '1970-01-01T00:00:00Z',
    })),
    has_more: false,
    first_id: models.at(0)?.id ?? null,
    last_id: models.at(-1)?.id ?? null,
  };
}

export class Router {
  private ranked: Model[];
  private health = new Map<string, { sequence: number; until: number; failures: number }>();
  private sequence = 0;

  constructor(models: Model[]) {
    this.ranked = models.slice().sort((a, b) => {
      // The two live grades recorded completions on Nemotron and none on Inkling.
      const availability = Number(a.id === INKLING_MODEL) - Number(b.id === INKLING_MODEL);
      return availability || b.rating - a.rating || (a.id < b.id ? -1 : 1);
    });
  }

  plan(body: Json) {
    const sequence = ++this.sequence;
    const needsTools = Array.isArray(body.tools) && body.tools.length > 0;
    const needsJson = isObject(body.response_format) && body.response_format.type !== 'text';
    const needsVision = Array.isArray(body.messages) && body.messages.some((message) =>
      isObject(message) && Array.isArray(message.content) && message.content.some((part) =>
        isObject(part) && part.type === 'image_url',
      ),
    );
    const eligible = this.ranked.filter((model) =>
      (body.model === 'auto' || body.model === model.id) &&
      (!needsTools || model.tools) && (!needsJson || model.json) && (!needsVision || model.vision),
    );
    const ready: Model[] = [];
    const cooling: Model[] = [];
    for (const model of eligible) {
      const until = this.health.get(model.id)?.until ?? 0;
      (sequence < until ? cooling : ready).push(model);
    }
    return { sequence, candidates: ready.concat(cooling) };
  }

  refused(id: string, sequence: number, status: number): void {
    const previous = this.health.get(id);
    if (previous && previous.sequence > sequence) return;
    const failures = (previous?.failures ?? 0) + 1;
    // A single timeout or 5xx may clear immediately. Other refusals get a short cooldown.
    const recurring = (status < 500 && status !== 408) || failures > 1;
    const wait = recurring ? Math.min(failures, 3) : 0;
    this.health.set(id, { sequence, failures, until: sequence + wait + 1 });
  }

  succeeded(id: string, sequence: number): void {
    const previous = this.health.get(id);
    if (previous && previous.sequence > sequence) return;
    this.health.set(id, { sequence, until: 0, failures: 0 });
  }
}
