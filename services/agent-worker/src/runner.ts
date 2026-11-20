import type { RunResult } from "@bae6/ai-agents";
import type { Claimed, Store } from "./store.js";

export interface RunnerOptions {
  store: Store;
  /** Builds the agent for one run, so tool state and guard budgets never outlive it. */
  execute: (run: Claimed, signal: AbortSignal) => Promise<RunResult>;
  concurrency: number;
  leaseMs: number;
  heartbeatMs: number;
  pollMs: number;
  maxAttempts: number;
  maxResultChars: number;
  log?: (event: Record<string, unknown>) => void;
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const done = () => {
      clearTimeout(t);
      signal.removeEventListener("abort", done);
      resolve();
    };
    const t = setTimeout(done, ms);
    signal.addEventListener("abort", done);
  });

export function createRunner(o: RunnerOptions) {
  const { store, log = () => {} } = o;
  const shutdown = new AbortController();
  let slots: Promise<void>[] = [];
  let reaper: NodeJS.Timeout | undefined;

  async function runOne(run: Claimed): Promise<void> {
    const ctl = new AbortController();
    let cancelled = false;
    let leaseLost = false;
    let renewedAt = Date.now();
    const beat = setInterval(() => {
      store
        .heartbeat(run.id, run.attempt, o.leaseMs)
        .then((h) => {
          if (!h) {
            leaseLost = true;
            ctl.abort();
            return;
          }
          renewedAt = Date.now();
          if (h.cancelRequested) {
            cancelled = true;
            ctl.abort();
          }
        })
        .catch((err: Error) => {
          log({ msg: "heartbeat_failed", runId: run.id, error: err.name });
          if (Date.now() - renewedAt >= o.leaseMs) {
            leaseLost = true;
            ctl.abort();
          }
        });
    }, o.heartbeatMs);
    const stopOnShutdown = () => ctl.abort();
    shutdown.signal.addEventListener("abort", stopOnShutdown, { once: true });
    if (shutdown.signal.aborted) ctl.abort();
    try {
      let result: RunResult;
      try {
        result = await o.execute(run, ctl.signal);
      } catch (err) {
        log({
          msg: "execute_failed",
          runId: run.id,
          error: (err as Error).name,
        });
        result = { stop: "model_error", tokens: 0, receipts: [] };
      }
      // Interrupted by shutdown, not by the caller: leave the lease to expire so another replica retries it.
      if (
        (shutdown.signal.aborted || leaseLost) &&
        result.stop === "aborted" &&
        !cancelled
      ) {
        return;
      }
      const kept = await store.finish(run.id, run.attempt, {
        stop: result.stop,
        text: result.text?.slice(0, o.maxResultChars),
        tokens: result.tokens,
        receipts: result.receipts,
      });
      if (!kept) log({ msg: "result_discarded", runId: run.id });
    } catch (err) {
      // The lease expires and the reaper requeues the run, within the attempt bound.
      log({ msg: "finish_failed", runId: run.id, error: (err as Error).name });
    } finally {
      clearInterval(beat);
      shutdown.signal.removeEventListener("abort", stopOnShutdown);
    }
  }

  async function slot(): Promise<void> {
    while (!shutdown.signal.aborted) {
      let run: Claimed | null = null;
      try {
        run = await store.claim(o.leaseMs);
      } catch (err) {
        log({ msg: "claim_failed", error: (err as Error).name });
      }
      if (run) await runOne(run);
      else await sleep(o.pollMs, shutdown.signal);
    }
  }

  return {
    start() {
      slots = Array.from({ length: o.concurrency }, slot);
      reaper = setInterval(() => {
        store
          .reap(o.maxAttempts)
          .then((n) => n > 0 && log({ msg: "reaped", runs: n }))
          .catch((err: Error) => log({ msg: "reap_failed", error: err.name }));
      }, o.leaseMs / 2);
      reaper.unref();
    },
    /** Stops claiming and interrupts running work; resolves once every slot has returned. */
    async stop() {
      clearInterval(reaper);
      shutdown.abort();
      await Promise.all(slots);
    },
  };
}
