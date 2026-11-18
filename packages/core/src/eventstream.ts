import { crc32 } from "node:zlib";

const MAX_FRAME_BYTES = 1024 * 1024;
const STRING_HEADER = 7;

export interface Frame {
  headers: Record<string, string>;
  payload: Buffer;
}

/** Decodes the AWS event-stream framing Bedrock streams in, checking both CRCs. */
export async function* frames(
  body: AsyncIterable<Uint8Array>,
): AsyncGenerator<Frame> {
  let buf = Buffer.alloc(0);
  for await (const chunk of body) {
    buf = Buffer.concat([buf, chunk]);
    while (buf.length >= 12) {
      const total = buf.readUInt32BE(0);
      if (total < 16 || total > MAX_FRAME_BYTES) {
        throw new Error("event-stream frame length out of bounds");
      }
      if (crc32(buf.subarray(0, 8)) !== buf.readUInt32BE(8)) {
        throw new Error("event-stream prelude CRC mismatch");
      }
      if (buf.length < total) break;
      if (crc32(buf.subarray(0, total - 4)) !== buf.readUInt32BE(total - 4)) {
        throw new Error("event-stream message CRC mismatch");
      }
      const headersEnd = 12 + buf.readUInt32BE(4);
      if (headersEnd > total - 4) {
        throw new Error("event-stream headers overrun the frame");
      }
      yield {
        headers: parseHeaders(buf.subarray(12, headersEnd)),
        payload: Buffer.from(buf.subarray(headersEnd, total - 4)),
      };
      buf = buf.subarray(total);
    }
  }
  if (buf.length > 0) throw new Error("event stream ended mid-frame");
}

function parseHeaders(b: Buffer): Record<string, string> {
  const out: Record<string, string> = {};
  let i = 0;
  while (i < b.length) {
    const nameLength = b.readUInt8(i);
    const name = b.toString("utf8", i + 1, i + 1 + nameLength);
    i += 1 + nameLength;
    // readUInt* throw past the end; toString would silently truncate the value.
    if (b.readUInt8(i) !== STRING_HEADER) {
      throw new Error("unsupported event-stream header type");
    }
    const length = b.readUInt16BE(i + 1);
    if (i + 3 + length > b.length) {
      throw new Error("event-stream header overruns the headers");
    }
    out[name] = b.toString("utf8", i + 3, i + 3 + length);
    i += 3 + length;
  }
  return out;
}

/** The inverse of `frames`, for tests and fixtures. */
export function encodeFrame(
  headers: Record<string, string>,
  payload: Buffer,
): Buffer {
  const encoded = Buffer.concat(
    Object.entries(headers).map(([name, value]) => {
      const n = Buffer.from(name);
      const v = Buffer.from(value);
      const head = Buffer.alloc(1 + n.length + 3);
      head.writeUInt8(n.length, 0);
      n.copy(head, 1);
      head.writeUInt8(STRING_HEADER, 1 + n.length);
      head.writeUInt16BE(v.length, 2 + n.length);
      return Buffer.concat([head, v]);
    }),
  );
  const total = 16 + encoded.length + payload.length;
  const prelude = Buffer.alloc(12);
  prelude.writeUInt32BE(total, 0);
  prelude.writeUInt32BE(encoded.length, 4);
  prelude.writeUInt32BE(crc32(prelude.subarray(0, 8)), 8);
  const body = Buffer.concat([prelude, encoded, payload]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([body, crc]);
}
