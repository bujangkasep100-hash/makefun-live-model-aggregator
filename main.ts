import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { catalogOf, isObject, listModels, loadModels, Router, type Json, type Wire } from './lib/catalog.ts';
import { chatStream, jsonCompletion, messagesStream, RESPONSE_BYTES, toChatRequest } from './lib/wire.ts';

const REQUEST_BYTES = 8 * 1024 * 1024;
const MAX_ATTEMPTS = 8;
const TERMINAL_CODES = new Set(['budget_exhausted', 'request_cap']);
const SSE_HEADERS = { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' };

const models = loadModels(process.env.MAKEFUN_PROMPT_FILE);
const catalog = catalogOf(models);
const router = new Router(models);
const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const gatewayRoot = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const gatewayKey = process.env.OPENAI_API_KEY;
const chatUrl = `${gatewayRoot}${gatewayRoot.endsWith('/v1') ? '' : '/v1'}/chat/completions`;

type Failure = { status: number; code: string; model: string };

function json(res: ServerResponse, status: number, body: Json): void {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

function problem(res: ServerResponse, wire: Wire, status: number, code: string, message: string): void {
  const type = status === 401 ? 'authentication_error'
    : status === 403 ? 'permission_error'
    : status === 404 ? 'not_found_error'
    : status === 429 ? 'rate_limit_error'
    : status === 529 ? 'overloaded_error'
    : status >= 500 ? 'api_error'
    : 'invalid_request_error';
  const error = { type, message };
  json(res, status, wire === 'anthropic' ? { type: 'error', error } : { error: { ...error, code, param: null } });
}

async function bodyOf(req: IncomingMessage): Promise<Json | null> {
  let bytes = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    bytes += chunk.byteLength;
    if (bytes <= REQUEST_BYTES) chunks.push(chunk);
  }
  if (bytes > REQUEST_BYTES) return null;
  const body: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (!isObject(body)) throw new Error('The request must be a JSON object');
  return body;
}

async function upstreamBody(response: Response): Promise<string> {
  if (!response.body) return '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return text + decoder.decode();
      bytes += value.byteLength;
      if (bytes > RESPONSE_BYTES) throw new Error('The upstream response is too large');
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function refusal(status: number, raw: string) {
  let code = `http_${status}`;
  let message = `The model gateway answered HTTP ${status}`;
  try {
    const body = JSON.parse(raw);
    const error = isObject(body.error) ? body.error : body;
    const tag = error.code ?? error.type;
    if ((typeof tag === 'string' || typeof tag === 'number') && /^[\w.-]{1,64}$/.test(String(tag))) {
      code = String(tag);
    }
    if (typeof error.message === 'string' && error.message) message = error.message.slice(0, 800);
  } catch {
    // Some upstream errors are HTML or plain text. The status still determines whether to retry.
  }
  const retry = !TERMINAL_CODES.has(code) && (status === 404 || status === 408 || status === 429 || status >= 500);
  return { status, code, message, retry };
}

function addTrail(res: ServerResponse, failures: Failure[]): void {
  if (failures.length) res.setHeader('x-makefun-upstream-failures', JSON.stringify(failures));
}

async function complete(res: ServerResponse, wire: Wire, body: Json): Promise<void> {
  const { sequence, candidates } = router.plan(body);
  if (candidates.length === 0) {
    problem(res, wire, 400, 'no_eligible_model', 'No free model supports this request');
    return;
  }

  const disconnect = new AbortController();
  const onClose = () => { if (!res.writableEnded) disconnect.abort(); };
  res.once('close', onClose);
  const signal = AbortSignal.any([disconnect.signal, AbortSignal.timeout(27_000)]);
  const failures: Failure[] = [];
  try {
    let last;
    for (const { id } of candidates.slice(0, MAX_ATTEMPTS)) {
      const upstream = await fetch(chatUrl, {
        method: 'POST',
        headers: { authorization: `Bearer ${gatewayKey}`, 'content-type': 'application/json' },
        body: JSON.stringify({ ...body, model: id }),
        redirect: 'error',
        signal,
      });
      if (upstream.ok) {
        // An admitted completion owns the response, including any failure during its stream.
        addTrail(res, failures);
        if (body.stream !== true) {
          json(res, 200, jsonCompletion(await upstreamBody(upstream), wire, body, id));
        } else if (wire === 'anthropic') {
          const events = await messagesStream(upstream, body, id);
          res.writeHead(200, SSE_HEADERS);
          res.end(events);
        } else {
          await chatStream(upstream, body, id, async (frame) => {
            if (!res.headersSent) res.writeHead(200, SSE_HEADERS);
            if (!res.write(frame)) await once(res, 'drain', { signal });
          });
          res.end();
        }
        router.succeeded(id);
        return;
      }

      last = refusal(upstream.status, await upstreamBody(upstream));
      failures.push({ status: last.status, code: last.code, model: id });
      if (!last.retry) break;
      router.refused(id, sequence, last.status);
    }
    if (last) {
      addTrail(res, failures);
      problem(res, wire, last.status, last.code, last.message);
    }
  } catch {
    if (disconnect.signal.aborted) return;
    const status = signal.aborted ? 504 : 502;
    const code = signal.aborted ? 'upstream_timeout' : 'invalid_upstream_response';
    const message = signal.aborted ? 'The model gateway timed out' : 'The model gateway did not return a complete answer';
    if (res.headersSent) {
      res.end(`data: ${JSON.stringify({ error: { code, message, type: 'api_error' } })}\n\n`);
    } else {
      addTrail(res, failures);
      problem(res, wire, status, code, message);
    }
  } finally {
    res.off('close', onClose);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = (req.url ?? '').split('?', 1)[0];
  const wire: Wire = path === '/v1/messages' ? 'anthropic' : 'openai';
  if (req.method === 'GET' && path === '/v1/models') {
    const anthropic = !/^Bearer\s/i.test(req.headers.authorization ?? '') && Boolean(req.headers['x-api-key']);
    json(res, 200, listModels(models, anthropic ? 'anthropic' : 'openai'));
    return;
  }
  if (req.method !== 'POST' || (path !== '/v1/chat/completions' && path !== '/v1/messages')) {
    problem(res, wire, 404, 'not_found', 'Unknown API route');
    return;
  }

  let chat;
  try {
    const body = await bodyOf(req);
    if (!body) {
      problem(res, wire, 413, 'request_too_large', 'The request exceeds 8 MiB');
      return;
    }
    chat = toChatRequest(body, wire);
  } catch (error) {
    problem(res, wire, 400, 'invalid_request', error instanceof Error ? error.message : 'Invalid request');
    return;
  }
  await complete(res, wire, chat);
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (!envUrl || !gatewayRoot || !gatewayKey || models.length === 0) {
    throw new Error('A Make.fun trial environment, its gateway and a model brief are required');
  }

  const server = createServer((req, res) => {
    void handle(req, res).catch(() => {
      if (res.headersSent) res.destroy();
      else if (!res.destroyed) problem(res, 'openai', 500, 'internal_error', 'The server failed to answer');
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', resolve);
  });
  try {
    const address = server.address();
    if (!address || typeof address === 'string' || address.port === 8080) throw new Error('Invalid listening port');
    const registered = await fetch(`${envUrl}/register`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }),
      signal: AbortSignal.timeout(5_000),
    });
    if (!registered.ok) throw new Error(`Registration failed with HTTP ${registered.status}`);
    await registered.arrayBuffer();

    for (;;) {
      try {
        const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(2_000) });
        if (response.ok && (await response.json()).done === true) break;
      } catch {
        // A polling failure is transient. The trial owns the overall wall time.
      }
      await delay(100);
    }
  } finally {
    server.close();
    server.closeAllConnections();
  }
}

try {
  await main();
} catch (error) {
  process.stderr.write(`aggregator: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
