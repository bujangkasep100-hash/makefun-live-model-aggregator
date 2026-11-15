import { isObject, type Json, type Wire } from './catalog.ts';

const CHAT_FIELDS = [
  'temperature', 'top_p', 'stop', 'seed', 'frequency_penalty', 'presence_penalty', 'logit_bias',
  'logprobs', 'top_logprobs', 'tools', 'tool_choice', 'parallel_tool_calls', 'response_format',
  'reasoning', 'reasoning_effort',
];
const STOP_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
export const RESPONSE_BYTES = 2 * 1024 * 1024;

function object(value: unknown): Json {
  if (!isObject(value)) throw new Error('Expected an object');
  return value;
}

function objects(value: unknown): Json[] {
  if (!Array.isArray(value)) throw new Error('Expected an array of objects');
  return value.map(object);
}

function copyFields(body: Json, fields: string[]): Json {
  return Object.fromEntries(fields.filter((field) => Object.hasOwn(body, field)).map((field) => [field, body[field]]));
}

function textOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(textOf).join('');
  if (isObject(value) && value.type === 'text' && typeof value.text === 'string') return value.text;
  return '';
}

function imageUrl(block: Json): string {
  const source = object(block.source);
  if (source.type === 'url' && typeof source.url === 'string') return source.url;
  if (source.type === 'base64' && typeof source.media_type === 'string' && typeof source.data === 'string') {
    return `data:${source.media_type};base64,${source.data}`;
  }
  throw new Error('An image must have a URL or a base64 source');
}

function anthropicMessages(body: Json): Json[] {
  const messages: Json[] = [];
  if (body.system !== undefined) messages.push({ role: 'system', content: textOf(body.system) });
  for (const { role, content } of objects(body.messages)) {
    if (role !== 'user' && role !== 'assistant') throw new Error('Invalid Messages role');
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }

    const blocks = objects(content);
    if (role === 'assistant') {
      const calls = blocks.filter((block) => block.type === 'tool_use').map((block) => ({
        id: block.id,
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(object(block.input)) },
      }));
      const text = textOf(blocks);
      messages.push(calls.length ? { role, content: text || null, tool_calls: calls } : { role, content: text });
      continue;
    }

    let parts: Json[] = [];
    for (const block of blocks) {
      if (block.type === 'text') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image') parts.push({ type: 'image_url', image_url: { url: imageUrl(block) } });
      else if (block.type === 'tool_result') {
        if (parts.length) messages.push({ role, content: parts });
        parts = [];
        messages.push({ role: 'tool', tool_call_id: block.tool_use_id, content: textOf(block.content) });
      } else throw new Error('Unsupported Messages content block');
    }
    if (parts.length) messages.push({ role, content: parts });
  }
  return messages;
}

function anthropicOptions(body: Json): Json {
  const options = copyFields(body, ['temperature', 'top_p']);
  if (body.stop_sequences !== undefined) options.stop = body.stop_sequences;
  if (body.tools !== undefined) {
    options.tools = objects(body.tools).map(({ name, description, input_schema }) => {
      if (typeof name !== 'string' || !isObject(input_schema)) throw new Error('Expected a named tool and input_schema');
      return { type: 'function', function: { name, description, parameters: input_schema } };
    });
  }
  if (isObject(body.tool_choice)) {
    const { type, name, disable_parallel_tool_use } = body.tool_choice;
    if (type === 'auto' || type === 'none') options.tool_choice = type;
    else if (type === 'any') options.tool_choice = 'required';
    else if (type === 'tool' && typeof name === 'string') {
      options.tool_choice = { type: 'function', function: { name } };
    } else throw new Error('Invalid Messages tool_choice');
    if (disable_parallel_tool_use === true) options.parallel_tool_calls = false;
  }
  if (isObject(body.thinking)) {
    const { type, budget_tokens } = body.thinking;
    options.reasoning = { enabled: type !== 'disabled', max_tokens: type === 'enabled' ? budget_tokens : undefined };
  }
  return options;
}

export function toChatRequest(body: Json, wire: Wire): Json {
  if (typeof body.model !== 'string' || body.model === '') throw new Error('A model id or auto is required');
  const capField = wire === 'openai' && body.max_completion_tokens !== undefined ? 'max_completion_tokens' : 'max_tokens';
  const cap = body[capField] ?? (wire === 'openai' ? 1024 : undefined);
  if (typeof cap !== 'number' || !Number.isInteger(cap) || cap < 1 || cap > 8192) {
    throw new Error('The output cap must be an integer between 1 and 8192');
  }
  if (body.stream !== undefined && typeof body.stream !== 'boolean') throw new Error('stream must be a boolean');
  if (wire === 'openai' && body.n !== undefined && body.n !== 1) throw new Error('Only one choice per request is supported');

  const options = wire === 'openai' ? copyFields(body, CHAT_FIELDS) : anthropicOptions(body);
  const messages = wire === 'openai' ? objects(body.messages) : anthropicMessages(body);
  if (!messages.length) throw new Error('At least one message is required');
  const request: Json = { model: body.model, messages, [capField]: cap, ...options };
  if (isObject(request.response_format) && request.response_format.type !== 'text') {
    request.provider = { require_parameters: true };
  }
  if (request.reasoning === undefined && request.reasoning_effort === undefined) {
    // Use the client's output budget for visible output unless they ask for reasoning.
    request.reasoning = { enabled: false };
  }
  if (body.stream === true) {
    request.stream = true;
    request.stream_options = { include_usage: true };
  }
  return request;
}

type Usage = { input_tokens: number; output_tokens: number };

function usageOf(value: unknown): Usage {
  const { prompt_tokens, completion_tokens } = object(value);
  if (typeof prompt_tokens !== 'number' || !Number.isInteger(prompt_tokens) || prompt_tokens < 0 ||
      typeof completion_tokens !== 'number' || !Number.isInteger(completion_tokens) || completion_tokens < 1) {
    throw new Error('Missing upstream token usage');
  }
  return { input_tokens: prompt_tokens, output_tokens: completion_tokens };
}

function checkStop(value: unknown): string {
  if (typeof value !== 'string' || !STOP_REASONS.has(value)) throw new Error('Invalid upstream finish reason');
  return value;
}

function messageStop(reason: string, sequence: unknown): Json {
  if (reason === 'stop' && typeof sequence === 'string' && sequence !== '') {
    return { stop_reason: 'stop_sequence', stop_sequence: sequence };
  }
  const stop_reason = reason === 'length' ? 'max_tokens'
    : reason === 'tool_calls' || reason === 'function_call' ? 'tool_use'
    : reason === 'content_filter' ? 'refusal' : 'end_turn';
  return { stop_reason, stop_sequence: null };
}

function messageHead(id: unknown, model: string): Json {
  if (typeof id !== 'string' || !id) throw new Error('Missing upstream completion id');
  return { id: `msg_${id}`, type: 'message', role: 'assistant', model };
}

type ToolCall = { id: string; name: string };

function toolBlock(tool: ToolCall, input: Json): Json {
  if (!tool.id || !tool.name) throw new Error('Incomplete upstream tool call');
  return { type: 'tool_use', id: tool.id, name: tool.name, input };
}

function completeTool(value: Json): Json {
  const fn = object(value.function);
  if (typeof value.id !== 'string' || typeof fn.name !== 'string' || typeof fn.arguments !== 'string') {
    throw new Error('Invalid upstream tool call');
  }
  return toolBlock({ id: value.id, name: fn.name }, object(JSON.parse(fn.arguments)));
}

function checkFormat(request: Json, text: string, reason: string): void {
  const format = request.response_format;
  if (!isObject(format) || reason === 'content_filter' || reason === 'length') return;
  if (format.type === 'json_object') object(JSON.parse(text));
  else if (format.type === 'json_schema') JSON.parse(text);
}

function asJson(raw: string, wire: Wire, request: Json, model: string): Json {
  const answer = object(JSON.parse(raw));
  const choice = objects(answer.choices)[0];
  const message = object(choice?.message);
  const usage = usageOf(answer.usage);
  const reason = checkStop(choice.finish_reason);
  if (message.role !== 'assistant') throw new Error('Invalid upstream message role');
  const text = textOf(message.content);
  checkFormat(request, text, reason);
  if (wire === 'openai') return { ...answer, model };

  const content: Json[] = [];
  if (text) content.push({ type: 'text', text });
  else if (typeof message.refusal === 'string' && message.refusal) content.push({ type: 'text', text: message.refusal });
  if (Array.isArray(message.tool_calls)) content.push(...objects(message.tool_calls).map(completeTool));
  if (isObject(message.function_call)) {
    content.push(completeTool({ id: `toolu_${answer.id}_0`, function: message.function_call }));
  }
  return { ...messageHead(answer.id, model), content, ...messageStop(reason, choice.stop_sequence), usage };
}

type StreamAnswer = { chunks: Json[]; usage: Usage; finish: Json; reason: string };

function frameData(frame: string): string {
  return frame.split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).replace(/^ /, ''))
    .join('\n').trim();
}

async function* streamData(body: ReadableStream<Uint8Array> | null): AsyncGenerator<string> {
  if (!body) throw new Error('Missing upstream stream');
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let bytes = 0;
  let trailingCR = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      bytes += value?.byteLength ?? 0;
      if (bytes > RESPONSE_BYTES) throw new Error('The upstream stream is too large');
      let chunk = (trailingCR ? '\r' : '') + decoder.decode(value, { stream: !done });
      trailingCR = !done && chunk.endsWith('\r');
      if (trailingCR) chunk = chunk.slice(0, -1);
      buffer += chunk.replace(/\r\n?/g, '\n');
      let end = buffer.indexOf('\n\n');
      while (end !== -1) {
        const data = frameData(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (data) yield data;
        end = buffer.indexOf('\n\n');
      }
      if (done) {
        const data = frameData(buffer);
        if (data) yield data;
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function readEvents(response: Response, request: Json, onChunk: (chunk: Json) => void | Promise<void>) {
  let latestUsage: unknown;
  let finish: Json | undefined;
  let reason: string | undefined;
  let text = '';
  let done = false;
  for await (const data of streamData(response.body)) {
    if (data === '[DONE]') {
      done = true;
      continue;
    }
    if (done) throw new Error('Upstream events followed the stream terminator');
    const chunk = object(JSON.parse(data));
    const [choice] = objects(chunk.choices);
    if (chunk.error || choice?.finish_reason === 'error') throw new Error('The upstream stream failed');
    if (chunk.usage != null) latestUsage = chunk.usage;
    if (isObject(choice?.delta)) {
      const part = textOf(choice.delta.content);
      if (finish && part) throw new Error('Upstream content followed the finish reason');
      if (isObject(request.response_format)) text += part;
    }
    if (choice?.finish_reason != null) {
      if (finish) throw new Error('The upstream sent multiple finish reasons');
      reason = checkStop(choice.finish_reason);
      finish = choice;
    }
    await onChunk(chunk);
  }
  if (!finish || !reason) throw new Error('Incomplete upstream stream');
  checkFormat(request, text, reason);
  return { usage: usageOf(latestUsage), finish, reason };
}

function event(type: string, fields: Json = {}): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...fields })}\n\n`;
}

type ContentBlock = { type: 'text'; parts: string[] } | { type: 'tool_use'; tool: ToolCall; parts: string[] };

function streamBlocks(chunks: Json[], id: unknown): ContentBlock[] {
  const blocks: ContentBlock[] = [];
  const tools = new Map<number, Extract<ContentBlock, { type: 'tool_use' }>>();
  for (const chunk of chunks) {
    const [choice] = objects(chunk.choices);
    if (!isObject(choice?.delta)) continue;
    const delta = choice.delta;
    const text = textOf(delta.content) || (typeof delta.refusal === 'string' ? delta.refusal : '');
    if (text) {
      const previous = blocks.at(-1);
      if (previous?.type === 'text') previous.parts.push(text);
      else blocks.push({ type: 'text', parts: [text] });
    }
    const calls = Array.isArray(delta.tool_calls) ? objects(delta.tool_calls) : [];
    if (isObject(delta.function_call)) calls.push({ id: `toolu_${id}_0`, index: 0, function: delta.function_call });
    for (const call of calls) {
      if (typeof call.index !== 'number' || !Number.isInteger(call.index)) throw new Error('Missing upstream tool index');
      let block = tools.get(call.index);
      if (!block) {
        block = { type: 'tool_use', tool: { id: '', name: '' }, parts: [] };
        tools.set(call.index, block);
        blocks.push(block);
      }
      if (typeof call.id === 'string') block.tool.id ||= call.id;
      if (!isObject(call.function)) continue;
      const { name, arguments: args } = call.function;
      if (typeof name === 'string' && name !== block.tool.name) block.tool.name += name;
      if (typeof args === 'string' && args) block.parts.push(args);
    }
  }
  return blocks;
}

function anthropicStream(answer: StreamAnswer, model: string): string {
  const { chunks, usage, reason, finish } = answer;
  const id = chunks.find((chunk) => typeof chunk.id === 'string')?.id;
  const message = {
    ...messageHead(id, model), content: [], stop_reason: null, stop_sequence: null,
    usage: { input_tokens: usage.input_tokens, output_tokens: 0 },
  };
  const events = [event('message_start', { message })];
  for (const [index, block] of streamBlocks(chunks, id).entries()) {
    const content_block = block.type === 'text' ? { type: 'text', text: '' } : toolBlock(block.tool, {});
    events.push(event('content_block_start', { index, content_block }));
    for (const part of block.parts) {
      const delta = block.type === 'text' ? { type: 'text_delta', text: part } : { type: 'input_json_delta', partial_json: part };
      events.push(event('content_block_delta', { index, delta }));
    }
    events.push(event('content_block_stop', { index }));
  }
  events.push(event('message_delta', { delta: messageStop(reason, finish.stop_sequence), usage: { output_tokens: usage.output_tokens } }));
  events.push(event('message_stop'));
  return events.join('');
}

export function jsonCompletion(raw: string, wire: Wire, request: Json, model: string): Json {
  return asJson(raw, wire, request, model);
}

export async function messagesStream(upstream: Response, request: Json, model: string): Promise<string> {
  // OpenRouter sends usage at the end. Hold it for the Messages message_start and message_delta.
  const chunks: Json[] = [];
  const answer = await readEvents(upstream, request, (chunk) => { chunks.push(chunk); });
  return anthropicStream({ ...answer, chunks }, model);
}

export async function chatStream(
  upstream: Response, request: Json, model: string, send: (frame: string) => Promise<void>,
): Promise<void> {
  await readEvents(upstream, request, (chunk) => send(`data: ${JSON.stringify({ ...chunk, model })}\n\n`));
  await send('data: [DONE]\n\n');
}
