import type { ClassifierConfig, RouteConfig, RouteRule } from "./config.js";

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
 * Rules decide when one matches; otherwise the classifier's tier does. A classifier that
 * gives no usable tier leaves the default, so it can only ever add a choice, never remove one.
 */
export async function pickRoute(
  route: RouteConfig,
  facts: RequestFacts,
  usable: (modelId: string) => boolean,
  classify: (c: ClassifierConfig) => Promise<string | null>,
): Promise<Choice | null> {
  const byRules = pick(route, facts, usable);
  if (!route.classifier || byRules?.decision.startsWith("rule:"))
    return byRules;
  const tier = await classify(route.classifier);
  const model = tier === null ? undefined : route.classifier.tiers[tier];
  if (tier !== null && model !== undefined && usable(model))
    return { modelId: model, decision: `classifier:${tier}` };
  return byRules && { ...byRules, decision: "default:classifier" };
}

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
