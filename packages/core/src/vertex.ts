import { chunks, count, json, post } from "./http.js";
import { prepare, requireOperation } from "./request.js";
import {
  type ChatRequest,
  type FinishReason,
  type ModelDescriptor,
  type ModelProvider,
  ProviderError,
  type StreamEvent,
  type Usage,
} from "./types.js";

export interface VertexModel {
  descriptor: ModelDescriptor;
  /** Publisher model, e.g. `gemini-2.5-flash` or `text-embedding-005`. */
  model: string;
  project: string;
  region: string;
  /** An OAuth access token for the Vertex AI API. */
  token: () => Promise<string>;
  /** Overrides `https://<region>-aiplatform.googleapis.com`, for tests. */
  endpoint?: string;
}

const MAX_SSE_LINE = 1024 * 1024;

const finish = (reason: unknown): FinishReason =>
  reason === "STOP" ? "stop" : reason === "MAX_TOKENS" ? "length" : "other";

type Chunk = {
  candidates?: {
    content?: { parts?: { text?: unknown }[] };
    finishReason?: unknown;
  }[];
  usageMetadata?: Record<string, unknown>;
};

const textOf = (c: Chunk) =>
  (c.candidates?.[0]?.content?.parts ?? [])
    .map((p) => (typeof p.text === "string" ? p.text : ""))
    .join("");

function usageOf(u: Record<string, unknown> | undefined): Usage {
  const inputTokens = count(u?.promptTokenCount);
  // The total includes thinking tokens, which candidatesTokenCount leaves out but are billed as output.
  const outputTokens =
    u?.totalTokenCount === undefined
      ? count(u?.candidatesTokenCount) +
        (u?.thoughtsTokenCount === undefined ? 0 : count(u.thoughtsTokenCount))
      : count(u.totalTokenCount) - inputTokens;
  return { inputTokens, outputTokens: count(outputTokens) };
}

/** The body's lines without line endings, including a last unterminated one. */
async function* sseLines(res: Response): AsyncGenerator<string> {
  const tooLong = () =>
    new ProviderError("provider stream line too long", "unknown");
  let pending = "";
  const decoder = new TextDecoder();
  for await (const c of chunks(res)) {
    pending += decoder.decode(c, { stream: true });
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    if (pending.length > MAX_SSE_LINE) throw tooLong();
    for (const line of lines) {
      if (line.length > MAX_SSE_LINE) throw tooLong();
      yield line.replace(/\r$/, "");
    }
  }
  pending += decoder.decode();
  if (pending) yield pending.replace(/\r$/, "");
}

/** Vertex AI generateContent for chat, predict for embeddings. */
export function vertexProvider(m: VertexModel): ModelProvider {
  const { descriptor } = m;
  const base = m.endpoint ?? `https://${m.region}-aiplatform.googleapis.com`;
  const path = `/v1/projects/${encodeURIComponent(m.project)}/locations/${encodeURIComponent(m.region)}/publishers/google/models/${encodeURIComponent(m.model)}`;

  async function send(method: string, body: unknown, signal: AbortSignal) {
    let token: string;
    try {
      token = await m.token();
    } catch {
      throw new ProviderError("GCP access token unavailable", "no");
    }
    return post(`${base}${path}:${method}`, {
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    });
  }

  function generate(req: ChatRequest) {
    const { system, turns } = prepare(descriptor, req);
    return {
      contents: turns.map((t) => ({
        role: t.role === "assistant" ? "model" : "user",
        parts: t.content.map((p) =>
          p.type === "text"
            ? { text: p.text }
            : { inlineData: { mimeType: p.mediaType, data: p.data } },
        ),
      })),
      ...(system.length
        ? { systemInstruction: { parts: system.map((text) => ({ text })) } }
        : {}),
      generationConfig: {
        maxOutputTokens: req.maxTokens,
        ...(req.temperature === undefined
          ? {}
          : { temperature: req.temperature }),
      },
    };
  }

  return {
    descriptor,

    async chat(req) {
      requireOperation(descriptor, "chat");
      const body = (await json(
        await send("generateContent", generate(req), req.signal),
      )) as Chunk;
      return {
        text: textOf(body),
        finishReason: finish(body.candidates?.[0]?.finishReason),
        usage: usageOf(body.usageMetadata),
      };
    },

    async *stream(req): AsyncGenerator<StreamEvent> {
      requireOperation(descriptor, "chat_stream");
      const res = await send(
        "streamGenerateContent?alt=sse",
        generate(req),
        req.signal,
      );
      let finishReason: FinishReason | undefined;
      let usage: Record<string, unknown> | undefined;
      for await (const line of sseLines(res)) {
        if (!line.startsWith("data:")) continue;
        let event: Chunk;
        try {
          event = JSON.parse(line.slice(5));
        } catch {
          throw new ProviderError("provider stream malformed", "unknown");
        }
        const text = textOf(event);
        if (text) yield { type: "delta", text };
        const reason = event.candidates?.[0]?.finishReason;
        if (reason !== undefined) finishReason = finish(reason);
        if (event.usageMetadata) usage = event.usageMetadata;
      }
      // A stream cut before its final chunk may still have been billed.
      if (finishReason === undefined) {
        throw new ProviderError("provider stream ended early", "unknown");
      }
      yield { type: "done", finishReason, usage: usageOf(usage) };
    },

    async embed(input, signal) {
      requireOperation(descriptor, "embed");
      const body = await json(
        await send(
          "predict",
          { instances: input.map((content) => ({ content })) },
          signal,
        ),
      );
      const predictions = Array.isArray(body.predictions)
        ? body.predictions
        : [];
      if (predictions.length !== input.length) {
        throw new ProviderError(
          "provider sent the wrong number of embeddings",
          "unknown",
        );
      }
      let inputTokens = 0;
      const vectors = predictions.map((p) => {
        const values = p?.embeddings?.values;
        if (
          !Array.isArray(values) ||
          !values.every((x) => typeof x === "number")
        ) {
          throw new ProviderError("provider sent no embedding", "unknown");
        }
        inputTokens += count(p.embeddings.statistics?.token_count);
        return values as number[];
      });
      return { vectors, usage: { inputTokens } };
    },
  };
}
