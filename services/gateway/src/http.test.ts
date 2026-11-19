import assert from "node:assert/strict";
import { createServer } from "node:http";
import { type AddressInfo, connect } from "node:net";
import { test } from "node:test";
import { readBody } from "./http.js";

test("a client that stalls mid-body is cut off at the deadline", async () => {
  const done = Promise.withResolvers<string>();
  const server = createServer((req) => {
    readBody(req, 1024, 100).then(
      () => done.resolve("read"),
      () => done.resolve("aborted"),
    );
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const socket = connect((server.address() as AddressInfo).port, "127.0.0.1");
  socket.on("error", () => {});
  socket.write("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 100\r\n\r\n{");
  const result = await Promise.race([
    done.promise,
    new Promise<string>((r) => setTimeout(() => r("hung"), 2000)),
  ]);
  socket.destroy();
  server.close();
  assert.equal(result, "aborted");
});
