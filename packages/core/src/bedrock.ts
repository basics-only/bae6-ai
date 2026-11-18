import { frames } from "./eventstream.js";
import { chunks, count, json, post } from "./http.js";
import { prepare, requireOperation } from "./request.js";
import { type AwsCredentials, signV4 } from "./sigv4.js";
import {
  type ChatRequest,
  type FinishReason,
  type ModelDescriptor,
  type ModelProvider,
  ProviderError,
  type StreamEvent,
  type Usage,
} from "./types.js";

export interface BedrockModel {
  descriptor: ModelDescriptor;
  /** Bedrock model or inference-profile ID. */
  modelId: string;
  region: string;
  credentials: () => Promise<AwsCredentials>;
  /** Overrides `https://bedrock-runtime.<region>.amazonaws.com`, for tests. */
  endpoint?: string;
}

const finish = (stopReason: unknown): FinishReason =>
  stopReason === "end_turn" || stopReason === "stop_sequence"
    ? "stop"
    : stopReason === "max_tokens"
      ? "length"
      : "other";

/** Bedrock Converse for chat, InvokeModel (Titan embeddings) for embed. */
export function bedrockProvider(m: BedrockModel): ModelProvider {
  const { descriptor } = m;
  const base =
    m.endpoint ?? `https://bedrock-runtime.${m.region}.amazonaws.com`;

  async function send(
    action: string,
    body: unknown,
    signal: AbortSignal,
    accept = "application/json",
  ) {
    let credentials: AwsCredentials;
    try {
      credentials = await m.credentials();
    } catch {
      throw new ProviderError("AWS credentials unavailable", "no");
    }
    const url = new URL(
      `${base}/model/${encodeURIComponent(m.modelId)}/${action}`,
    );
    const text = JSON.stringify(body);
    const { host: _, ...headers } = signV4({
      method: "POST",
      url,
      headers: { "content-type": "application/json", accept },
      body: text,
      region: m.region,
      service: "bedrock",
      credentials,
    });
    return post(url.href, { headers, body: text, signal });
  }

  function converse(req: ChatRequest) {
    const { system, turns } = prepare(descriptor, req);
    return {
      messages: turns.map((t) => ({
        role: t.role,
        content: t.content.map((p) =>
          p.type === "text"
            ? { text: p.text }
            : {
                image: {
                  format: p.mediaType.slice("image/".length),
                  source: { bytes: p.data },
                },
              },
        ),
      })),
      ...(system.length ? { system: system.map((text) => ({ text })) } : {}),
      inferenceConfig: {
        maxTokens: req.maxTokens,
        ...(req.temperature === undefined
          ? {}
          : { temperature: req.temperature }),
      },
    };
  }

  const usageOf = (u: unknown): Usage => {
    const v = (u ?? {}) as Record<string, unknown>;
    return {
      inputTokens: count(v.inputTokens),
      outputTokens: count(v.outputTokens),
    };
  };

  return {
    descriptor,

    async chat(req) {
      requireOperation(descriptor, "chat");
      const body = await json(
        await send("converse", converse(req), req.signal),
      );
      const message = (body.output as { message?: { content?: unknown } })
        ?.message;
      const content = Array.isArray(message?.content) ? message.content : [];
      return {
        text: content
          .map((c) => (typeof c?.text === "string" ? c.text : ""))
          .join(""),
        finishReason: finish(body.stopReason),
        usage: usageOf(body.usage),
      };
    },

    async *stream(req): AsyncGenerator<StreamEvent> {
      requireOperation(descriptor, "chat_stream");
      const res = await send(
        "converse-stream",
        converse(req),
        req.signal,
        "application/vnd.amazon.eventstream",
      );
      let finishReason: FinishReason = "other";
      let usage: Usage | undefined;
      try {
        for await (const f of frames(chunks(res))) {
          if (f.headers[":message-type"] !== "event") {
            throw new ProviderError(
              `provider stream error (${f.headers[":exception-type"] ?? "unknown"})`,
              "unknown",
            );
          }
          const event = JSON.parse(f.payload.toString("utf8"));
          switch (f.headers[":event-type"]) {
            case "contentBlockDelta":
              if (typeof event.delta?.text === "string") {
                yield { type: "delta", text: event.delta.text };
              }
              break;
            case "messageStop":
              finishReason = finish(event.stopReason);
              break;
            case "metadata":
              usage = usageOf(event.usage);
              break;
          }
        }
      } catch (err) {
        if (err instanceof ProviderError) throw err;
        throw new ProviderError("provider stream malformed", "unknown");
      }
      if (!usage)
        throw new ProviderError("provider reported no usage", "unknown");
      yield { type: "done", finishReason, usage };
    },

    async embed(input, signal) {
      requireOperation(descriptor, "embed");
      const vectors: number[][] = [];
      let inputTokens = 0;
      // Titan embeds one text per call; the gateway bounds how many inputs there are.
      for (const inputText of input) {
        const body = await json(await send("invoke", { inputText }, signal));
        const { embedding } = body;
        if (
          !Array.isArray(embedding) ||
          !embedding.every((x) => typeof x === "number")
        ) {
          throw new ProviderError("provider sent no embedding", "unknown");
        }
        vectors.push(embedding);
        inputTokens += count(body.inputTextTokenCount);
      }
      return { vectors, usage: { inputTokens } };
    },
  };
}
