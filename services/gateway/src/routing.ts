import type { RouteConfig, RouteRule } from "./config.js";

export interface RequestFacts {
  inputTokens: number;
  hasImages: boolean;
}

export interface Choice {
  modelId: string;
  /** `rule:<index>`, `default`, or `default:skipped` when a matching rule's model was unusable. */
  decision: string;
}

const matches = (rule: RouteRule, f: RequestFacts) =>
  f.inputTokens >= (rule.minInputTokens ?? 0) &&
  f.inputTokens <= (rule.maxInputTokens ?? Infinity) &&
  (rule.images === undefined || rule.images === f.hasImages);

/**
 * First matching rule whose model passes the hard filters (`usable`), else the default,
 * else null. A rule never overrides a filter: an unusable model is skipped, not forced.
 */
export function pick(
  route: RouteConfig,
  facts: RequestFacts,
  usable: (modelId: string) => boolean,
): Choice | null {
  let skipped = false;
  for (const [i, rule] of route.rules.entries()) {
    if (!matches(rule, facts)) continue;
    if (usable(rule.model))
      return { modelId: rule.model, decision: `rule:${i}` };
    skipped = true;
  }
  if (!usable(route.default)) return null;
  return {
    modelId: route.default,
    decision: skipped ? "default:skipped" : "default",
  };
}
