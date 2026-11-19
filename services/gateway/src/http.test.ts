import assert from "node:assert/strict";
import { createServer } from "node:http";
import { type AddressInfo, connect } from "node:net";
import { test } from "node:test";
import { HttpError, readBody } from "./http.js";

async function sendBody(parts: string[], gapMs: number, deadlineMs: number) {
  const done = Promise.withResolvers<unknown>();
  const server = createServer((req) => {
    readBody(req, 1024, deadlineMs).then(done.resolve, done.resolve);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const socket = connect((server.address() as AddressInfo).port, "127.0.0.1");
  socket.on("error", () => {});
  const length = parts.join("").length + (parts.length > 1 ? 0 : 100);
  socket.write(
    `POST / HTTP/1.1\r\nhost: x\r\ncontent-length: ${length}\r\n\r\n`,
  );
  for (const part of parts) {
    socket.write(part);
    await new Promise((r) => setTimeout(r, gapMs));
  }
  const result = await Promise.race([
    done.promise,
    new Promise((r) => setTimeout(() => r("hung"), 2000)),
  ]);
  socket.destroy();
  server.close();
  return result;
}

test("a client that stalls mid-body is cut off with a 408", async () => {
  const result = await sendBody(["{"], 0, 100);
  assert.ok(result instanceof HttpError);
  assert.equal(result.status, 408);
});

test("a slow but steady body longer than the deadline is still read", async () => {
  const result = await sendBody(['{"a"', ":1", "}"], 70, 100);
  assert.deepEqual((result as { json: unknown }).json, { a: 1 });
});
