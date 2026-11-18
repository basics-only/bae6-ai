import assert from "node:assert/strict";
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { crc32 } from "node:zlib";
import { bedrockProvider } from "./bedrock.js";
import { encodeFrame, frames } from "./eventstream.js";
import { signV4 } from "./sigv4.js";
import {
  type ChatRequest,
  type ModelDescriptor,
  ProviderError,
  type StreamEvent,
  UnsupportedOperation,
} from "./types.js";
import { vertexProvider } from "./vertex.js";

type Seen = { path: string; headers: IncomingMessage["headers"]; body: any };

async function mock(
  handler: (seen: Seen, res: ServerResponse) => void,
): Promise<{ url: string; seen: Seen[]; close: () => void }> {
  const seen: Seen[] = [];
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const c of req) raw += c;
    const s = {
      path: req.url ?? "",
      headers: req.headers,
      body: raw ? JSON.parse(raw) : undefined,
    };
    seen.push(s);
    handler(s, res);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return {
    url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    seen,
    close: () => {
      server.closeAllConnections();
      server.close();
    },
  };
}

const sendJson = (res: ServerResponse, status: number, body: unknown) => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
};

const descriptor = (
  id: string,
  provider: "bedrock" | "vertex",
  inputModalities: ("text" | "image")[] = ["text", "image"],
): ModelDescriptor => ({
  kind: "model",
  id,
  operations: ["chat", "chat_stream", "embed"],
  limits: { maxInputTokens: 100_000, maxOutputTokens: 4096 },
  attributes: { provider, inputModalities },
});

const credentials = async () => ({
  accessKeyId: "AKIDEXAMPLE",
  secretAccessKey: "secret",
});

const request = (extra: Partial<ChatRequest> = {}): ChatRequest => ({
  messages: [
    { role: "system", content: [{ type: "text", text: "Be brief." }] },
    {
      role: "user",
      content: [
        { type: "text", text: "What is this?" },
        { type: "image", mediaType: "image/png", data: "aGVsbG8=" },
      ],
    },
  ],
  maxTokens: 64,
  signal: AbortSignal.timeout(2000),
  ...extra,
});

async function collect(it: AsyncIterable<StreamEvent>) {
  const out: StreamEvent[] = [];
  for await (const e of it) out.push(e);
  return out;
}

test("SigV4 matches the AWS get-vanilla test vector", () => {
  const headers = signV4({
    method: "GET",
    url: new URL("https://example.amazonaws.com/"),
    headers: {},
    body: "",
    region: "us-east-1",
    service: "service",
    credentials: {
      accessKeyId: "AKIDEXAMPLE",
      secretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
    },
    now: new Date("2015-08-30T12:36:00Z"),
  });
  assert.equal(
    headers.authorization,
    "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
  );
});

test("event-stream frames round-trip and a corrupted frame is rejected", async () => {
  const frame = encodeFrame(
    { ":message-type": "event", ":event-type": "x" },
    Buffer.from('{"a":1}'),
  );
  // Split mid-frame to exercise buffering.
  const parsed = [];
  for await (const f of frames(
    (async function* () {
      yield frame.subarray(0, 5);
      yield frame.subarray(5);
    })(),
  )) {
    parsed.push(f);
  }
  assert.deepEqual(parsed[0]?.headers, {
    ":message-type": "event",
    ":event-type": "x",
  });
  assert.equal(parsed[0]?.payload.toString(), '{"a":1}');

  const bad = Buffer.from(frame);
  bad.writeUInt8(bad.readUInt8(bad.length - 6) ^ 0xff, bad.length - 6);
  await assert.rejects(async () => {
    for await (const _ of frames(
      (async function* () {
        yield bad;
      })(),
    )) {
    }
  }, /CRC/);
});

test("Bedrock chat signs a Converse call and maps text, images and usage", async () => {
  const server = await mock((_, res) =>
    sendJson(res, 200, {
      output: { message: { content: [{ text: "A " }, { text: "logo." }] } },
      stopReason: "end_turn",
      usage: { inputTokens: 12, outputTokens: 3 },
    }),
  );
  try {
    const p = bedrockProvider({
      descriptor: descriptor("claude", "bedrock"),
      modelId: "anthropic.claude-v1:0",
      region: "us-east-1",
      credentials,
      endpoint: server.url,
    });
    const result = await p.chat(request({ temperature: 0.2 }));
    assert.deepEqual(result, {
      text: "A logo.",
      finishReason: "stop",
      usage: { inputTokens: 12, outputTokens: 3 },
    });
    const [s] = server.seen;
    assert.equal(s?.path, "/model/anthropic.claude-v1%3A0/converse");
    assert.match(String(s?.headers.authorization), /^AWS4-HMAC-SHA256 /);
    assert.deepEqual(s?.body, {
      messages: [
        {
          role: "user",
          content: [
            { text: "What is this?" },
            { image: { format: "png", source: { bytes: "aGVsbG8=" } } },
          ],
        },
      ],
      system: [{ text: "Be brief." }],
      inferenceConfig: { maxTokens: 64, temperature: 0.2 },
    });
  } finally {
    server.close();
  }
});

test("Bedrock streams deltas and ends with measured usage", async () => {
  const event = (type: string, body: unknown) =>
    encodeFrame(
      { ":message-type": "event", ":event-type": type },
      Buffer.from(JSON.stringify(body)),
    );
  const server = await mock((_, res) => {
    res.writeHead(200, {
      "content-type": "application/vnd.amazon.eventstream",
    });
    res.write(event("contentBlockDelta", { delta: { text: "Hi" } }));
    res.write(event("contentBlockDelta", { delta: { text: " there" } }));
    res.write(event("messageStop", { stopReason: "max_tokens" }));
    res.end(event("metadata", { usage: { inputTokens: 5, outputTokens: 2 } }));
  });
  try {
    const p = bedrockProvider({
      descriptor: descriptor("claude", "bedrock"),
      modelId: "m",
      region: "us-east-1",
      credentials,
      endpoint: server.url,
    });
    assert.deepEqual(await collect(p.stream(request())), [
      { type: "delta", text: "Hi" },
      { type: "delta", text: " there" },
      {
        type: "done",
        finishReason: "length",
        usage: { inputTokens: 5, outputTokens: 2 },
      },
    ]);
    assert.equal(server.seen[0]?.path, "/model/m/converse-stream");
  } finally {
    server.close();
  }
});

test("a provider error answer is not billed; missing usage, an exception frame or a timeout may be", async () => {
  let mode = "error";
  const server = await mock((_, res) => {
    if (mode === "error")
      return sendJson(res, 400, { message: "echoed prompt" });
    if (mode === "nousage") {
      return sendJson(res, 200, { output: { message: { content: [] } } });
    }
    if (mode === "exception") {
      res.writeHead(200);
      return res.end(
        encodeFrame(
          {
            ":message-type": "exception",
            ":exception-type": "throttlingException",
          },
          Buffer.from("{}"),
        ),
      );
    }
    // hang
  });
  try {
    const p = bedrockProvider({
      descriptor: descriptor("claude", "bedrock"),
      modelId: "m",
      region: "us-east-1",
      credentials,
      endpoint: server.url,
    });
    const billed = async (fn: () => Promise<unknown>) => {
      try {
        await fn();
      } catch (err) {
        assert.ok(err instanceof ProviderError, String(err));
        assert.doesNotMatch(err.message, /echoed prompt/);
        return err.billed;
      }
      assert.fail("expected a ProviderError");
    };
    assert.equal(await billed(() => p.chat(request())), "no");
    mode = "nousage";
    assert.equal(await billed(() => p.chat(request())), "unknown");
    mode = "exception";
    assert.equal(await billed(() => collect(p.stream(request()))), "unknown");
    mode = "hang";
    assert.equal(
      await billed(() => p.chat(request({ signal: AbortSignal.timeout(100) }))),
      "unknown",
    );
  } finally {
    server.close();
  }
});

test("an unsupported modality or operation fails before any provider call", async () => {
  const server = await mock((_, res) => sendJson(res, 500, {}));
  try {
    const p = bedrockProvider({
      descriptor: {
        ...descriptor("text-only", "bedrock", ["text"]),
        operations: ["chat"],
      },
      modelId: "m",
      region: "us-east-1",
      credentials,
      endpoint: server.url,
    });
    await assert.rejects(p.chat(request()), UnsupportedOperation);
    await assert.rejects(collect(p.stream(request())), UnsupportedOperation);
    await assert.rejects(
      p.embed(["x"], AbortSignal.timeout(1000)),
      UnsupportedOperation,
    );
    assert.equal(server.seen.length, 0);
  } finally {
    server.close();
  }
});

test("Vertex chat, SSE streaming and embeddings map to the common shapes", async () => {
  const server = await mock((s, res) => {
    if (s.path.endsWith(":generateContent")) {
      return sendJson(res, 200, {
        candidates: [
          { content: { parts: [{ text: "A logo." }] }, finishReason: "STOP" },
        ],
        usageMetadata: { promptTokenCount: 9, candidatesTokenCount: 2 },
      });
    }
    if (s.path.endsWith(":streamGenerateContent?alt=sse")) {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(
        'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\r\n\r\n',
      );
      return res.end(
        'data: {"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":6}}\n\n',
      );
    }
    sendJson(res, 200, {
      predictions: [
        { embeddings: { values: [0.1, 0.2], statistics: { token_count: 3 } } },
      ],
    });
  });
  try {
    const p = vertexProvider({
      descriptor: descriptor("gemini", "vertex"),
      model: "gemini-2.5-flash",
      project: "proj",
      region: "europe-west4",
      token: async () => "ya29.test",
      endpoint: server.url,
    });
    assert.deepEqual(await p.chat(request()), {
      text: "A logo.",
      finishReason: "stop",
      usage: { inputTokens: 9, outputTokens: 2 },
    });
    const [s] = server.seen;
    assert.equal(
      s?.path,
      "/v1/projects/proj/locations/europe-west4/publishers/google/models/gemini-2.5-flash:generateContent",
    );
    assert.equal(s?.headers.authorization, "Bearer ya29.test");
    assert.deepEqual(s?.body.systemInstruction, {
      parts: [{ text: "Be brief." }],
    });
    assert.deepEqual(s?.body.contents[0].parts[1], {
      inlineData: { mimeType: "image/png", data: "aGVsbG8=" },
    });

    assert.deepEqual(await collect(p.stream(request())), [
      { type: "delta", text: "Hi" },
      { type: "delta", text: "!" },
      {
        type: "done",
        finishReason: "length",
        usage: { inputTokens: 4, outputTokens: 2 },
      },
    ]);
    assert.deepEqual(await p.embed(["hello"], AbortSignal.timeout(1000)), {
      vectors: [[0.1, 0.2]],
      usage: { inputTokens: 3 },
    });
  } finally {
    server.close();
  }
});

test("event-stream headers that overrun their frame are rejected", async () => {
  const frame = encodeFrame({ ":message-type": "event" }, Buffer.from("{}"));
  frame.writeUInt32BE(frame.length, 4);
  frame.writeUInt32BE(crc32(frame.subarray(0, 8)), 8);
  frame.writeUInt32BE(
    crc32(frame.subarray(0, frame.length - 4)),
    frame.length - 4,
  );
  await assert.rejects(async () => {
    for await (const _ of frames(
      (async function* () {
        yield frame;
      })(),
    )) {
    }
  }, /overrun/);
});

test("a Bedrock embedding that fails after earlier inputs were billed may have been billed", async () => {
  let calls = 0;
  const server = await mock((_, res) =>
    ++calls === 1
      ? sendJson(res, 200, { embedding: [0.1], inputTextTokenCount: 1 })
      : sendJson(res, 429, {}),
  );
  try {
    const p = bedrockProvider({
      descriptor: descriptor("titan", "bedrock"),
      modelId: "m",
      region: "us-east-1",
      credentials,
      endpoint: server.url,
    });
    await assert.rejects(
      p.embed(["a", "b"], AbortSignal.timeout(1000)),
      (err) => err instanceof ProviderError && err.billed === "unknown",
    );
  } finally {
    server.close();
  }
});

test("Vertex bills thinking tokens, reads an unterminated last line, and a stream with no finish may be billed", async () => {
  let finished = true;
  const server = await mock((s, res) => {
    if (s.path.endsWith(":generateContent")) {
      return sendJson(res, 200, {
        candidates: [
          { content: { parts: [{ text: "x" }] }, finishReason: "STOP" },
        ],
        usageMetadata: {
          promptTokenCount: 9,
          candidatesTokenCount: 2,
          thoughtsTokenCount: 5,
          totalTokenCount: 16,
        },
      });
    }
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(
      finished
        ? 'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":6}}'
        : 'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":6}}\n\n',
    );
  });
  try {
    const p = vertexProvider({
      descriptor: descriptor("gemini", "vertex"),
      model: "gemini-2.5-flash",
      project: "proj",
      region: "europe-west4",
      token: async () => "ya29.test",
      endpoint: server.url,
    });
    assert.deepEqual((await p.chat(request())).usage, {
      inputTokens: 9,
      outputTokens: 7,
    });
    assert.deepEqual(await collect(p.stream(request())), [
      { type: "delta", text: "Hi" },
      {
        type: "done",
        finishReason: "stop",
        usage: { inputTokens: 4, outputTokens: 2 },
      },
    ]);
    finished = false;
    await assert.rejects(
      collect(p.stream(request())),
      (err) => err instanceof ProviderError && err.billed === "unknown",
    );
  } finally {
    server.close();
  }
});
