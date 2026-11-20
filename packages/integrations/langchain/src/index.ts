import type { Tool } from "@bae6/ai-agents";

export interface LangChainTool {
  name: string;
  description: string;
  schema?: { parse(input: unknown): unknown };
  invoke(input: unknown, config?: { signal?: AbortSignal }): Promise<unknown>;
}

export interface AdaptOptions {
  requiresApproval?: boolean;
}

export function fromLangChainTool(
  tool: LangChainTool,
  options: AdaptOptions = {},
): Tool {
  return {
    name: tool.name,
    description: tool.description,
    ...(options.requiresApproval !== undefined && {
      requiresApproval: options.requiresApproval,
    }),
    parse: (input) => (tool.schema ? tool.schema.parse(input) : input),
    async run(input, signal) {
      const output = await tool.invoke(input, { signal });
      return typeof output === "string" ? output : JSON.stringify(output);
    },
  };
}
