/** Package-local JSONL RPC over the unmodified Pi finance tools. */
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(process.argv[2] ?? "");
if (!process.argv[2]) throw new Error("native_tools requires the absolute pi-finance repository path");
const { loadFinanceTools } = await import(pathToFileURL(resolve(root, "tools/loader.ts")).href);
const { decodeToolText } = await import(pathToFileURL(resolve(root, "pifinance/safactory_transport.ts")).href);
// A missing model must never silently turn helper-model failures into evidence.
const { validateToolArguments } = await import(pathToFileURL(resolve(root, "node_modules/@earendil-works/pi-ai/dist/index.js")).href);
const tools = await loadFinanceTools(() => ({}));
const context = { messages: [], tools };
const emit = (value: unknown) => process.stdout.write(JSON.stringify(value) + "\n");
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
	try {
		const request = JSON.parse(line);
		if (request.op === "describe") {
			emit({ ok: true, tools: tools.map(({ name, description, parameters }: any) => ({ name, description, parameters })) });
			continue;
		}
		if (!["execute", "execute_calls"].includes(request.op) ||
			(request.op === "execute" && typeof request.text !== "string") || typeof request.id !== "string" ||
			!Number.isInteger(request.remaining_tool_calls) || request.remaining_tool_calls < 0 ||
			!Number.isFinite(request.timeout_ms) || request.timeout_ms <= 0) {
			throw new Error("invalid native tools RPC request");
		}
		const blocks = request.op === "execute" ? decodeToolText(request.text, context, request.id, false) :
			[...request.calls.map((call: any) => ({ type: "toolCall", ...call })),
			 { type: "text", text: request.visible_text ?? "" }];
		const calls = blocks.filter((block: any) => block.type === "toolCall");
		const visible = blocks.filter((block: any) => block.type === "text").map((block: any) => block.text).join("").trim();
		const identities = new Set();
		for (const call of calls) {
			if (typeof call.id !== "string" || identities.has(call.id)) throw new Error("Invalid native tool identity");
			identities.add(call.id);
			const tool = tools.find((item: any) => item.name === call.name);
			if (!tool) throw new Error("Unknown native tool");
			validateToolArguments(tool, call);
		}
		// A normal tool-free model response ends this teacher branch. Requiring
		// FINISH at character zero made a report with a preamble trigger a second
		// generation after an assistant message, which violates the token cache.
		// Completion alone does not establish new evidence or continuation value.
		const finish = calls.length === 0 && visible.length > 0;
		const finishKind = finish ? (/^FINISH:\s*\S/i.test(visible) ? "explicit_finish" : "unmarked_no_tool_stop") : null;
		if (calls.length === 0 && !finish) {
			emit({ ok: false, reason: "empty_no_tool_response", tool_calls: 0, observations: [], finish: false });
			continue;
		}
		if (calls.length > request.remaining_tool_calls) {
			emit({ ok: false, reason: "tool_budget", tool_calls: 0, observations: [], finish: false });
			continue;
		}
		const observations: any[] = [];
		let used = 0;
		let fatal: string | undefined;
		const deadline = Date.now() + request.timeout_ms;
		for (const call of calls) {
			if (call.name === "retrieve_information" || (call.name === "fetch_content" && call.arguments?.mode === "answer")) { fatal = `unsupported_llm_helper: ${call.name}`; break; }
			const remaining = deadline - Date.now();
			if (remaining <= 0) { fatal = "tool_timeout"; break; }
			const controller = new AbortController();
			const timer = setTimeout(() => controller.abort(), remaining);
			let content: string;
			let isError = false;
			try {
				const tool = tools.find((item: any) => item.name === call.name)!;
				used++;
				const result = await tool.execute(call.id, call.arguments, controller.signal);
				content = (result.content ?? []).map((block: any) => {
					if (block.type !== "text" || typeof block.text !== "string") throw new Error("non-text native tool result");
					return block.text;
				}).join("\n");
				isError = result.isError === true;
			} catch (error) {
				isError = true;
				content = errorText(error);
				if (controller.signal.aborted) fatal = "tool_timeout";
				else if (/Tool call budget exhausted/.test(content)) fatal = "tool_budget";
			} finally { clearTimeout(timer); }
			if (request.op === "execute_calls") observations.push({ role: "tool", tool_call_id: call.id,
				name: call.name, arguments: call.arguments, is_error: isError, content });
			else observations.push({ role: "user", content: JSON.stringify({ tool_result: {
				id: call.id, name: call.name, is_error: isError, content,
			} }) });
			if (fatal) break;
		}
		emit({ ok: !fatal, reason: fatal ?? null, finish, finish_kind: finishKind, observations, tool_calls: used,
			visible_text: visible, calls: calls.map(({ id, name, arguments: args }: any) => ({ id, name, arguments: args })) });
	} catch (error) {
		emit({ ok: false, reason: "invalid_action", error: errorText(error), observations: [], tool_calls: 0, finish: false });
	}
}
