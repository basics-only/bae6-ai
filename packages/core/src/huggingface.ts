import { count, json, post, sseLines } from "./http.js";
import { prepare, requireOperation } from "./request.js";
import {
  type ChatRequest,
  type FinishReason,
  type Message,
  type ModelDescriptor,
  type ModelProvider,
  ProviderError,
  type StreamEvent,
  UnsupportedOperation,
  type Usage,
} from "./types.js";

export interface HuggingFaceModel {
  descriptor: ModelDescriptor;
  /** The model repository, e.g. `meta-llama/Llama-3.1-8B-Instruct`. */
  model: string;
  /** Origin from `huggingfaceOrigin`; defaults to Inference Providers. */
  endpoint?: string;
  /** A fine-grained access token. Sent only to the allow-listed endpoint. */
  token: string;
  /** For tests against a local server. The gateway never sets it. */
  allowAnyHost?: boolean;
}

const ROUTER = "https://router.huggingface.co";
const DEDICATED = /^[a-z0-9-]{1,63}\.endpoints\.huggingface\.cloud$/;

/** The origin if it is Hugging Face's router or a dedicated endpoint, else null. */
export function huggingfaceOrigin(v: unknown): string | null {
  if (typeof v !== "string") return null;
  try {
    const u = new URL(v);
    const ok =
      u.protocol === "https:" &&
      u.username === "" &&
      u.password === "" &&
      u.port === "" &&
      u.search === "" &&
      u.hash === "" &&
      u.pathname === "/" &&
      (u.hostname === "router.huggingface.co" || DEDICATED.test(u.hostname));
    return ok ? u.origin : null;
  } catch {
    return null;
  }
}

const finish = (reason: unknown): FinishReason =>
  reason === "stop" ? "stop" : reason === "length" ? "length" : "other";

type Chunk = {
  choices?: {
    message?: { content?: unknown };
    delta?: { content?: unknown };
    finish_reason?: unknown;
  }[];
  usage?: { prompt_tokens?: unknown; completion_tokens?: unknown };
};

const usageOf = (u: Chunk["usage"]): Usage => ({
  inputTokens: count(u?.prompt_tokens),
  outputTokens: count(u?.completion_tokens),
});

const wire = (m: Message) => {
  const parts = m.content.map((p) =>
    p.type === "text"
      ? { type: "text", text: p.text }
      : {
          type: "image_url",
          image_url: { url: `data:${p.mediaType};base64,${p.data}` },
        },
  );
  const first = parts[0];
  return {
    role: m.role,
    content: parts.length === 1 && first?.type === "text" ? first.text : parts,
  };
};

/** The OpenAI-compatible chat API of Inference Providers and Inference Endpoints. */
export function huggingfaceProvider(m: HuggingFaceModel): ModelProvider {
  const { descriptor } = m;
  const origin =
    m.endpoint === undefined
      ? ROUTER
      : m.allowAnyHost
        ? m.endpoint.replace(/\/$/, "")
        : huggingfaceOrigin(m.endpoint);
  if (origin === null) {
    throw new Error(`${descriptor.id}: endpoint is not a Hugging Face host`);
  }

  function send(req: ChatRequest, stream: boolean) {
    const { system, turns } = prepare(descriptor, req);
    return post(`${origin}/v1/chat/completions`, {
      headers: {
        authorization: `Bearer ${m.token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: m.model,
        messages: [
          ...system.map((content) => ({ role: "system", content })),
          ...turns.map(wire),
        ],
        max_tokens: req.maxTokens,
        ...(req.temperature === undefined
          ? {}
          : { temperature: req.temperature }),
        ...(stream
          ? { stream: true, stream_options: { include_usage: true } }
          : {}),
      }),
      signal: req.signal,
    });
  }

  return {
    descriptor,

    async chat(req) {
      requireOperation(descriptor, "chat");
      const body = (await json(await send(req, false))) as Chunk;
      const choice = body.choices?.[0];
      if (typeof choice?.message?.content !== "string") {
        throw new ProviderError("provider sent no answer", "unknown");
      }
      return {
        text: choice.message.content,
        finishReason: finish(choice.finish_reason),
        usage: usageOf(body.usage),
      };
    },

    async *stream(req): AsyncGenerator<StreamEvent> {
      requireOperation(descriptor, "chat_stream");
      const res = await send(req, true);
      let finishReason: FinishReason | undefined;
      let usage: Chunk["usage"];
      for await (const line of sseLines(res)) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") break;
        let event: Chunk;
        try {
          event = JSON.parse(data);
        } catch {
          throw new ProviderError("provider stream malformed", "unknown");
        }
        const choice = event.choices?.[0];
        const text = choice?.delta?.content;
        if (typeof text === "string" && text) yield { type: "delta", text };
        if (choice?.finish_reason != null) {
          finishReason = finish(choice.finish_reason);
        }
        if (event.usage) usage = event.usage;
      }
      // A stream cut before its final chunk may still have been billed.
      if (finishReason === undefined) {
        throw new ProviderError("provider stream ended early", "unknown");
      }
      yield { type: "done", finishReason, usage: usageOf(usage) };
    },

    async embed() {
      throw new UnsupportedOperation(`${descriptor.id} does not support embed`);
    },
  };
}
