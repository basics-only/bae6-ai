import type { ContentPart, FinishReason, Message, Usage } from "@bae6/ai-core";
import { HttpError } from "./http.js";

const MAX_MESSAGES = 256;
const MAX_PARTS = 64;
const MAX_IMAGES = 20;
const MAX_EMBED_INPUTS = 16;
const MAX_EMBED_CHARS = 32_000;
const IMAGE_TOKENS = 1600;
const DATA_URL =
  /^data:(image\/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/]+={0,2})$/;

const invalid = (message: string) =>
  new HttpError(400, "invalid_request", message);
const unsupported = (message: string) =>
  new HttpError(400, "unsupported_parameter", message);

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

function onlyKeys(b: Record<string, unknown>, allowed: string[]) {
  for (const k of Object.keys(b)) {
    if (!allowed.includes(k)) throw unsupported(`${k} is not supported.`);
  }
}

export interface ChatInput {
  model: string;
  messages: Message[];
  maxTokens: number | undefined;
  temperature: number | undefined;
  stream: boolean;
  includeUsage: boolean;
  /** Conservative upper estimate of input tokens, for the budget reservation. */
  inputTokens: number;
}

function part(p: unknown, images: { n: number }): ContentPart {
  if (!isObject(p)) throw invalid("Each content part must be an object.");
  if (p.type === "text" && typeof p.text === "string") {
    return { type: "text", text: p.text };
  }
  if (p.type === "image_url" && isObject(p.image_url)) {
    // Only inline images: fetching caller-named URLs would let callers reach our network.
    const m = DATA_URL.exec(String(p.image_url.url));
    if (!m) throw invalid("image_url.url must be a base64 data: URL.");
    if (++images.n > MAX_IMAGES) {
      throw invalid(`At most ${MAX_IMAGES} images per request.`);
    }
    return {
      type: "image",
      mediaType: m[1] as "image/png",
      data: m[2] as string,
    };
  }
  throw unsupported(`Content part type ${String(p.type)} is not supported.`);
}

export function parseChat(body: unknown): ChatInput {
  if (!isObject(body)) throw invalid("Body must be an object.");
  onlyKeys(body, [
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "stream",
    "stream_options",
  ]);
  if (typeof body.model !== "string") throw invalid("model is required.");
  const raw = body.messages;
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > MAX_MESSAGES) {
    throw invalid(`messages must hold 1 to ${MAX_MESSAGES} messages.`);
  }
  const images = { n: 0 };
  let textBytes = 0;
  const messages = raw.map((m): Message => {
    if (!isObject(m)) throw invalid("Each message must be an object.");
    onlyKeys(m, ["role", "content"]);
    const role =
      m.role === "developer"
        ? "system"
        : m.role === "system" || m.role === "user" || m.role === "assistant"
          ? m.role
          : undefined;
    if (!role) throw unsupported(`Role ${String(m.role)} is not supported.`);
    let content: ContentPart[];
    if (typeof m.content === "string") {
      content = [{ type: "text", text: m.content }];
    } else if (
      Array.isArray(m.content) &&
      m.content.length > 0 &&
      m.content.length <= MAX_PARTS
    ) {
      content = m.content.map((p) => part(p, images));
    } else {
      throw invalid(`content must be a string or 1 to ${MAX_PARTS} parts.`);
    }
    for (const p of content) {
      if (p.type === "text") textBytes += Buffer.byteLength(p.text);
    }
    return { role, content };
  });

  const max = body.max_completion_tokens ?? body.max_tokens;
  if (max !== undefined && (!Number.isInteger(max) || (max as number) < 1)) {
    throw invalid("max_tokens must be a positive integer.");
  }
  const t = body.temperature;
  if (t !== undefined && (typeof t !== "number" || t < 0 || t > 2)) {
    throw invalid("temperature must be a number from 0 to 2.");
  }
  if (body.stream !== undefined && typeof body.stream !== "boolean") {
    throw invalid("stream must be a boolean.");
  }
  const so = body.stream_options;
  if (so !== undefined) {
    if (
      !isObject(so) ||
      (so.include_usage !== undefined && typeof so.include_usage !== "boolean")
    ) {
      throw invalid("stream_options must be {include_usage}.");
    }
    onlyKeys(so, ["include_usage"]);
  }
  return {
    model: body.model,
    messages,
    maxTokens: max as number | undefined,
    temperature: t as number | undefined,
    stream: body.stream === true,
    includeUsage: isObject(so) && so.include_usage === true,
    // About 4 bytes per token in practice; 3 over-reserves rather than under.
    inputTokens:
      Math.ceil(textBytes / 3) + images.n * IMAGE_TOKENS + messages.length * 4,
  };
}

export function parseEmbeddings(body: unknown) {
  if (!isObject(body)) throw invalid("Body must be an object.");
  onlyKeys(body, ["model", "input", "encoding_format"]);
  if (typeof body.model !== "string") throw invalid("model is required.");
  if (body.encoding_format !== undefined && body.encoding_format !== "float") {
    throw unsupported("Only encoding_format float is supported.");
  }
  const input = typeof body.input === "string" ? [body.input] : body.input;
  if (
    !Array.isArray(input) ||
    input.length === 0 ||
    input.length > MAX_EMBED_INPUTS ||
    !input.every(
      (s) =>
        typeof s === "string" && s.length > 0 && s.length <= MAX_EMBED_CHARS,
    )
  ) {
    throw invalid(
      `input must be 1 to ${MAX_EMBED_INPUTS} non-empty strings of at most ${MAX_EMBED_CHARS} characters.`,
    );
  }
  const strings = input as string[];
  return {
    model: body.model,
    input: strings,
    inputTokens: strings.reduce(
      (n, s) => n + Math.ceil(Buffer.byteLength(s) / 3),
      0,
    ),
  };
}

const FINISH: Record<FinishReason, string> = {
  stop: "stop",
  length: "length",
  other: "stop",
};

const usageJson = (u: Usage) => ({
  prompt_tokens: u.inputTokens,
  completion_tokens: u.outputTokens,
  total_tokens: u.inputTokens + u.outputTokens,
});

export function completion(
  id: string,
  model: string,
  text: string,
  finishReason: FinishReason,
  usage: Usage,
) {
  return {
    id: `chatcmpl-${id}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: FINISH[finishReason],
      },
    ],
    usage: usageJson(usage),
  };
}

export function chunk(
  id: string,
  model: string,
  delta: Record<string, string>,
  finishReason: FinishReason | null = null,
) {
  return {
    id: `chatcmpl-${id}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason && FINISH[finishReason],
      },
    ],
  };
}

export function usageChunk(id: string, model: string, usage: Usage) {
  return {
    id: `chatcmpl-${id}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [],
    usage: usageJson(usage),
  };
}
