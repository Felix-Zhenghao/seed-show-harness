/** DeepSeek Harness adapter: read-only history tools and the Python model gateway. */
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { resolve, relative, isAbsolute } from "node:path";
import { pathToFileURL } from "node:url";

// Resolve the active dsh installation, not a second global dependency tree.
const require = createRequire(process.argv[1]);
const { LlmAdapter } = await import(pathToFileURL(require.resolve("@deepseek-ai/dsh-llm")));

export const name = "seed-robot-planner";
export const inject = ["llm", "tools"];

const objectSchema = properties => ({ type: "object", properties,
  required: Object.keys(properties), additionalProperties: false });

function imageReferences(value) {
  if (Array.isArray(value)) return value.map(imageReferences);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, imageReferences(item)]));
  }
  if (typeof value === "string") {
    const match = /^data:image\/(png|jpeg|jpg|webp|gif);base64,(.*)$/s.exec(value);
    if (match) {
      const digest = createHash("sha256").update(Buffer.from(match[2], "base64")).digest("hex");
      return { asset_path: `images/${digest}.${match[1]}`, data_url_prefix: `data:image/${match[1]};base64,` };
    }
  }
  return value;
}

export function apply(ctx, config) {
  const root = resolve(config.root);
  const canonicalRoot = realpath(root);
  const imageCalls = new Map();

  async function checkedPath(path, folder) {
    if (typeof path !== "string" || !path.trim() || isAbsolute(path)) throw new Error("Expected a relative history path");
    const target = await realpath(resolve(root, path));
    const rel = relative(await canonicalRoot, target);
    if (rel.startsWith("..") || isAbsolute(rel) || !rel.startsWith(folder + "/")) {
      throw new Error("Path is outside the permitted history folder");
    }
    return target;
  }

  function register(name, description, parameters, schema, execute) {
    ctx.tools.register({ name, description, parameters, execute,
      output: { schema, render: (_args, data) => [{ type: "text", text: JSON.stringify(data) }] } });
  }

  register("read_history", "Read a window of the frozen episode event history, including executed actions, observations, and model-record paths.",
    objectSchema({ start_line: { type: "integer", minimum: 1 }, max_lines: { type: "integer", minimum: 1, maximum: 100 } }),
    objectSchema({ text: { type: "string" }, total_lines: { type: "integer" } }),
    async ({ start_line, max_lines }) => {
      const lines = (await readFile(config.history, "utf8")).trimEnd().split("\n");
      return { text: lines.slice(start_line - 1, start_line - 1 + max_lines).join("\n"), total_lines: lines.length };
    });
  register("read_model_record", "Read a saved JSON request or response referenced by the event history. Paths must be under calls/. Image data is represented by image paths; use view_observation to inspect those images.",
    objectSchema({ path: { type: "string" } }), objectSchema({ text: { type: "string" } }),
    async ({ path }) => {
      const record = JSON.parse(await readFile(await checkedPath(path, "calls"), "utf8"));
      return { text: JSON.stringify(imageReferences(record)).slice(0, 80000) };
    });
  register("view_observation", "View a saved PNG observation or annotation referenced by history. Paths must be under images/.",
    objectSchema({ path: { type: "string" } }), objectSchema({ image_path: { type: "string" } }),
    async ({ path }) => {
      const target = await checkedPath(path, "images");
      if (!target.endsWith(".png")) throw new Error("Only saved PNG observations are supported");
      return { image_path: path };
    });

  class StrictAdapter extends LlmAdapter {
    async resolveModel(provider, model) {
      return { provider, id: model, name: model, inputModalities: ["text", "image"], context: { contextWindow: 1000000 } };
    }

    async *stream(options) {
      const input = [];
      for (const message of options.messages) {
        if (message.role === "system" || message.role === "developer") continue;
        if (message.role === "tool") {
          const content = message.content.filter(b => b.type === "text").map(b => b.text).join("\n");
          const callId = message.source.callId;
          let output = content;
          if (imageCalls.get(callId) === "view_observation") {
            let data;
            // dsh renders failed tools as plain text, not the success schema.
            // Preserve that native result so the agent can correct its path.
            try { data = JSON.parse(content); } catch { data = undefined; }
            try {
              if (typeof data?.image_path === "string") {
                const bytes = await readFile(await checkedPath(data.image_path, "images"));
                output = [{ type: "input_text", text: content },
                  { type: "input_image", image_url: `data:image/png;base64,${bytes.toString("base64")}`, detail: "original" }];
              }
            } catch (error) {
              output = JSON.stringify({ error: `Cannot load historical image: ${error.message}` });
            }
          }
          input.push({ type: "function_call_output", call_id: callId, output });
          continue;
        }
        const text = message.content.filter(b => b.type === "text").map(b => b.text).join("\n");
        if (text) input.push({ role: message.role, content: text });
        for (const block of message.content) {
          if (block.type === "tool-call") {
            imageCalls.set(block.id, block.name);
            input.push({ type: "function_call", call_id: block.id, name: block.name, arguments: block.arguments });
          }
          if (block.type === "reasoning") input.push({ type: "reasoning", content: [{ type: "reasoning_text", text: block.text }] });
        }
      }
      const tools = (options.tools ?? []).map(tool => ({ type: "function", name: tool.name,
        description: tool.description, parameters: tool.parameters, strict: true }));
      const response = await fetch(config.gateway, { method: "POST", signal: options.signal,
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${process.env.SEED_GATEWAY_TOKEN}` },
        body: JSON.stringify({ input, tools }) });
      const envelope = await response.json();
      if (!response.ok) throw new Error(envelope.error ?? `Gateway HTTP ${response.status}`);
      const result = envelope.response;
      let index = 0, toolCalls = false;
      for (const item of result.output) {
        const blocks = [];
        if (item.type === "function_call") {
          toolCalls = true;
          imageCalls.set(item.call_id, item.name);
          blocks.push({ type: "tool-call", id: item.call_id, name: item.name, arguments: item.arguments });
        } else if (item.type === "message" || item.type === "reasoning") {
          for (const part of item.content ?? []) {
            if (part.type === "output_text") blocks.push({ type: "text", text: part.text });
            if (part.type === "reasoning_text") blocks.push({ type: "reasoning", text: part.text });
          }
        }
        for (const block of blocks) {
          yield { type: "block-start", index, blockType: block.type };
          if (block.type === "tool-call") yield { type: "tool-call-delta", index, id: block.id, name: block.name, argumentsDelta: block.arguments };
          else yield { type: block.type === "text" ? "text-delta" : "reasoning-delta", index, text: block.text };
          yield { type: "block-end", index, block };
          index++;
        }
      }
      const usage = result.usage ?? {}, cached = usage.input_tokens_details?.cached_tokens ?? 0;
      yield { type: "usage", usage: { inputTokens: (usage.input_tokens ?? 0) - cached,
        cacheReadTokens: cached, outputTokens: usage.output_tokens ?? 0,
        totalTokens: usage.total_tokens ?? ((usage.input_tokens ?? 0) + (usage.output_tokens ?? 0)) } };
      yield { type: "finish", reason: { kind: toolCalls ? "tool-calls" : "stop" } };
    }
  }
  ctx.llm.registerAdapter(["seed-strict"], new StrictAdapter());
}
