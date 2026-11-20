import { readFileSync } from "node:fs";
import { nonAwsEndpointModels, parseConfig } from "./config.js";

const file = process.argv[2];
if (!file) {
  console.error("usage: check-config <config.json>");
  process.exit(2);
}

try {
  const config = parseConfig(JSON.parse(readFileSync(file, "utf8")));
  const foreign = nonAwsEndpointModels(config)[0];
  if (foreign !== undefined) {
    throw new Error(`model ${foreign}: endpoint is not an AWS host`);
  }
  console.log(
    `ok: ${config.models.length} models, price book ${config.priceBook.version}`,
  );
} catch (e) {
  console.error(e instanceof Error ? e.message : String(e));
  process.exit(1);
}
