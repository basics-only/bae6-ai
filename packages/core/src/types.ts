export type Modality = "text" | "image";
export type Operation = "chat" | "chat_stream" | "embed";

/** A bae6-contracts CapabilityDescriptor for one model. */
export interface ModelDescriptor {
  kind: "model";
  id: string;
  operations: Operation[];
  limits: { maxInputTokens: number; maxOutputTokens: number };
  attributes: {
    provider: "bedrock" | "vertex" | "huggingface";
    inputModalities: Modality[];
    hosting?: "inference-providers" | "dedicated";
  };
}

export type ContentPart =
  | { type: "text"; text: string }
  | {
      type: "image";
      mediaType: "image/png" | "image/jpeg" | "image/gif" | "image/webp";
      /** base64 */
      data: string;
    };

export interface Message {
  role: "system" | "user" | "assistant";
  content: ContentPart[];
}

export interface ChatRequest {
  messages: Message[];
  maxTokens: number;
  temperature?: number;
  signal: AbortSignal;
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
}

export type FinishReason = "stop" | "length" | "other";

export interface ChatResult {
  text: string;
  finishReason: FinishReason;
  usage: Usage;
}

export type StreamEvent =
  | { type: "delta"; text: string }
  | { type: "done"; finishReason: FinishReason; usage: Usage };

export interface ModelProvider {
  readonly descriptor: ModelDescriptor;
  chat(req: ChatRequest): Promise<ChatResult>;
  stream(req: ChatRequest): AsyncIterable<StreamEvent>;
  embed(
    input: string[],
    signal: AbortSignal,
  ): Promise<{ vectors: number[][]; usage: { inputTokens: number } }>;
}

/** The model does not offer this operation or modality; never a silent fallback. */
export class UnsupportedOperation extends Error {
  readonly code = "unsupported_operation";
}

/**
 * The provider call failed. `billed: "no"` only when the provider answered with an error;
 * anything else (timeout, dropped connection, mid-stream failure) may have been billed.
 */
export class ProviderError extends Error {
  constructor(
    message: string,
    readonly billed: "no" | "unknown",
    readonly status?: number,
  ) {
    super(message);
  }
}
