import {
  type ChatRequest,
  type Message,
  type ModelDescriptor,
  type Operation,
  UnsupportedOperation,
} from "./types.js";

export function requireOperation(d: ModelDescriptor, op: Operation) {
  if (!d.operations.includes(op)) {
    throw new UnsupportedOperation(`${d.id} does not support ${op}`);
  }
}

/** Checks the modalities, then splits system text from the conversation. */
export function prepare(
  d: ModelDescriptor,
  req: ChatRequest,
): { system: string[]; turns: Message[] } {
  const system: string[] = [];
  const turns: Message[] = [];
  for (const m of req.messages) {
    for (const p of m.content) {
      if (!d.attributes.inputModalities.includes(p.type)) {
        throw new UnsupportedOperation(`${d.id} does not accept ${p.type}`);
      }
    }
    if (m.role !== "system") {
      turns.push(m);
      continue;
    }
    for (const p of m.content) {
      if (p.type !== "text") {
        throw new UnsupportedOperation("system messages are text only");
      }
      system.push(p.text);
    }
  }
  return { system, turns };
}
