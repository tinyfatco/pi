import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";

const realtimeMock = vi.hoisted(() => {
	type Listener = (...args: unknown[]) => void;

	class FakeSocket {
		readyState = 0;
		private listeners = new Map<string, Set<Listener>>();

		on(type: string, listener: Listener): void {
			const listeners = this.listeners.get(type) ?? new Set<Listener>();
			listeners.add(listener);
			this.listeners.set(type, listeners);
		}

		off(type: string, listener: Listener): void {
			this.listeners.get(type)?.delete(listener);
		}

		open(): void {
			this.readyState = 1;
			this.emit("open");
		}

		close(code = 1000, reason = "closed"): void {
			if (this.readyState === 3) return;
			this.readyState = 3;
			this.emit("close", code, reason);
		}

		emit(type: string, ...args: unknown[]): void {
			for (const listener of this.listeners.get(type) ?? []) {
				listener(...args);
			}
		}
	}

	class MockOpenAIRealtimeWS {
		static instances: MockOpenAIRealtimeWS[] = [];
		socket = new FakeSocket();
		sent: unknown[] = [];
		model: string;
		options: unknown;
		private listeners = new Map<string, Set<Listener>>();

		constructor(props: { model: string; options?: unknown }) {
			this.model = props.model;
			this.options = props.options;
		}

		static async create(
			_client: unknown,
			props: { model: string; options?: unknown },
		): Promise<MockOpenAIRealtimeWS> {
			const instance = new MockOpenAIRealtimeWS(props);
			MockOpenAIRealtimeWS.instances.push(instance);
			queueMicrotask(() => instance.socket.open());
			return instance;
		}

		on(type: string, listener: Listener): void {
			const listeners = this.listeners.get(type) ?? new Set<Listener>();
			listeners.add(listener);
			this.listeners.set(type, listeners);
		}

		off(type: string, listener: Listener): void {
			this.listeners.get(type)?.delete(listener);
		}

		send(event: unknown): void {
			this.sent.push(event);
		}

		close(props?: { code: number; reason: string }): void {
			this.socket.close(props?.code, props?.reason);
		}

		emitEvent(event: unknown): void {
			this.emit("event", event);
			if (event && typeof event === "object" && "type" in event && event.type === "error") {
				const error = "error" in event ? event.error : undefined;
				const message =
					error && typeof error === "object" && "message" in error && typeof error.message === "string"
						? error.message
						: "Realtime error";
				this.emit("error", new Error(message));
			}
		}

		emitError(error: Error): void {
			this.emit("error", error);
		}

		private emit(type: string, ...args: unknown[]): void {
			for (const listener of this.listeners.get(type) ?? []) {
				listener(...args);
			}
		}
	}

	return { MockOpenAIRealtimeWS };
});

vi.mock("openai/realtime/ws.js", () => ({
	OpenAIRealtimeWS: realtimeMock.MockOpenAIRealtimeWS,
}));

import { getApiProvider } from "../src/api-registry.ts";
import { getModel, getSupportedThinkingLevels } from "../src/models.ts";
import {
	buildResponseCreateEvent,
	streamOpenAIRealtime,
	streamSimpleOpenAIRealtime,
} from "../src/providers/openai-realtime.ts";
import "../src/providers/register-builtins.ts";
import type { AssistantMessage, Context, Model, ToolResultMessage, Usage } from "../src/types.ts";

const usage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function createModel(): Model<"openai-realtime"> {
	return {
		id: "gpt-realtime-2",
		name: "GPT Realtime 2",
		api: "openai-realtime",
		provider: "openai",
		baseUrl: "https://api.openai.com/v1",
		reasoning: true,
		thinkingLevelMap: { off: null, xhigh: "xhigh" },
		input: ["text", "image"],
		cost: { input: 4, output: 24, cacheRead: 0.4, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 32000,
	};
}

async function waitFor(predicate: () => boolean): Promise<void> {
	for (let attempt = 0; attempt < 50; attempt++) {
		if (predicate()) return;
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
	throw new Error("Timed out waiting for predicate");
}

function latestRealtimeInstance(): InstanceType<typeof realtimeMock.MockOpenAIRealtimeWS> {
	const instance = realtimeMock.MockOpenAIRealtimeWS.instances.at(-1);
	if (!instance) throw new Error("No realtime instance");
	return instance;
}

afterEach(() => {
	realtimeMock.MockOpenAIRealtimeWS.instances.length = 0;
	vi.restoreAllMocks();
});

describe("openai realtime provider", () => {
	it("registers gpt-realtime-2 as an openai-realtime model", () => {
		const provider = getApiProvider("openai-realtime");
		expect(provider).toBeDefined();

		const model = getModel("openai", "gpt-realtime-2");
		expect(model.api).toBe("openai-realtime");
		expect(model.contextWindow).toBe(128000);
		expect(model.maxTokens).toBe(32000);
		expect(model.input).toEqual(["text", "image"]);
		expect(getSupportedThinkingLevels(model)).toEqual(["minimal", "low", "medium", "high", "xhigh"]);
	});

	it("sends an out-of-band text response request and streams text/usage", async () => {
		const model = createModel();
		const context: Context = {
			systemPrompt: "Be concise.",
			messages: [
				{
					role: "user",
					content: [
						{ type: "text", text: "Describe this image" },
						{ type: "image", mimeType: "image/png", data: "aW1hZ2U=" },
					],
					timestamp: Date.now(),
				},
			],
			tools: [
				{
					name: "lookup",
					description: "Look up a value",
					parameters: Type.Object({ key: Type.String() }),
				},
			],
		};
		const events: string[] = [];
		const resultStream = streamOpenAIRealtime(model, context, {
			apiKey: "sk-test",
			reasoningEffort: "xhigh",
			maxTokens: 123,
			headers: { "x-test": "yes" },
		});
		const consume = (async () => {
			for await (const event of resultStream) {
				events.push(event.type === "text_delta" ? `text_delta:${event.delta}` : event.type);
			}
		})();

		await waitFor(() => latestRealtimeInstance().sent.length === 1);
		const instance = latestRealtimeInstance();
		expect(instance.model).toBe("gpt-realtime-2");
		expect(instance.options).toEqual({ headers: { "x-test": "yes" } });

		const payload = instance.sent[0] as {
			type: string;
			response: {
				conversation: string;
				instructions: string;
				output_modalities: string[];
				max_output_tokens: number;
				reasoning: { effort: string };
				parallel_tool_calls: boolean;
				input: Array<{ role?: string; content?: Array<Record<string, unknown>> }>;
				tools: Array<{ name?: string; parameters?: unknown }>;
			};
		};
		expect(payload.type).toBe("response.create");
		expect(payload.response.conversation).toBe("none");
		expect(payload.response.instructions).toBe("Be concise.");
		expect(payload.response.output_modalities).toEqual(["text"]);
		expect(payload.response.max_output_tokens).toBe(123);
		expect(payload.response.reasoning).toEqual({ effort: "xhigh" });
		expect(payload.response.parallel_tool_calls).toBe(true);
		expect(payload.response.tools[0]?.name).toBe("lookup");
		expect(payload.response.input[0]?.content?.[0]).toEqual({ type: "input_text", text: "Describe this image" });
		expect(payload.response.input[0]?.content?.[1]).toEqual({
			type: "input_image",
			detail: "auto",
			image_url: "data:image/png;base64,aW1hZ2U=",
		});

		instance.emitEvent({ type: "response.created", event_id: "evt_1", response: { id: "resp_1" } });
		instance.emitEvent({
			type: "response.output_item.added",
			event_id: "evt_2",
			response_id: "resp_1",
			output_index: 0,
			item: { type: "message", id: "msg_1", role: "assistant", status: "in_progress", content: [] },
		});
		instance.emitEvent({
			type: "response.output_text.delta",
			event_id: "evt_3",
			response_id: "resp_1",
			item_id: "msg_1",
			output_index: 0,
			content_index: 0,
			delta: "Hel",
		});
		instance.emitEvent({
			type: "response.output_text.delta",
			event_id: "evt_4",
			response_id: "resp_1",
			item_id: "msg_1",
			output_index: 0,
			content_index: 0,
			delta: "lo",
		});
		instance.emitEvent({
			type: "response.output_item.done",
			event_id: "evt_5",
			response_id: "resp_1",
			output_index: 0,
			item: {
				type: "message",
				id: "msg_1",
				role: "assistant",
				status: "completed",
				content: [{ type: "output_text", text: "Hello" }],
			},
		});
		instance.emitEvent({
			type: "response.done",
			event_id: "evt_6",
			response: {
				id: "resp_1",
				status: "completed",
				usage: {
					input_tokens: 12,
					output_tokens: 3,
					total_tokens: 15,
					input_token_details: { cached_tokens: 2 },
				},
			},
		});

		const result = await resultStream.result();
		await consume;
		expect(result.responseId).toBe("resp_1");
		expect(result.stopReason).toBe("stop");
		expect(result.content).toEqual([{ type: "text", text: "Hello", textSignature: "msg_1" }]);
		expect(result.usage.input).toBe(10);
		expect(result.usage.cacheRead).toBe(2);
		expect(result.usage.output).toBe(3);
		expect(result.usage.cost.total).toBeCloseTo(0.0001128);
		expect(events).toContain("text_delta:Hel");
		expect(events).toContain("text_delta:lo");
		expect(events.at(-1)).toBe("done");
	});

	it("converts tool history and streams tool calls", async () => {
		const model = createModel();
		const assistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "toolCall", id: "call_prev|fc_prev", name: "lookup", arguments: { key: "alpha" } }],
			api: "openai-realtime",
			provider: "openai",
			model: "gpt-realtime-2",
			usage,
			stopReason: "toolUse",
			timestamp: Date.now() - 2000,
		};
		const toolResult: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "call_prev|fc_prev",
			toolName: "lookup",
			content: [{ type: "text", text: "alpha=42" }],
			isError: false,
			timestamp: Date.now() - 1000,
		};
		const context: Context = {
			messages: [
				{ role: "user", content: "Use lookup", timestamp: Date.now() - 3000 },
				assistant,
				toolResult,
				{ role: "user", content: "Now continue", timestamp: Date.now() },
			],
			tools: [{ name: "lookup", description: "Look up", parameters: Type.Object({ key: Type.String() }) }],
		};

		const payload = buildResponseCreateEvent(model, context);
		expect(payload.response?.input?.[1]).toMatchObject({
			type: "function_call",
			id: "fc_prev",
			call_id: "call_prev",
			name: "lookup",
			arguments: JSON.stringify({ key: "alpha" }),
		});
		expect(payload.response?.input?.[2]).toMatchObject({
			type: "function_call_output",
			call_id: "call_prev",
			output: "alpha=42",
		});

		const resultStream = streamOpenAIRealtime(model, context, { apiKey: "sk-test" });
		await waitFor(() => latestRealtimeInstance().sent.length === 1);
		const instance = latestRealtimeInstance();

		instance.emitEvent({
			type: "response.output_item.added",
			event_id: "evt_1",
			response_id: "resp_1",
			output_index: 0,
			item: {
				type: "function_call",
				id: "fc_1",
				call_id: "call_1",
				name: "lookup",
				arguments: "",
				status: "in_progress",
			},
		});
		instance.emitEvent({
			type: "response.function_call_arguments.delta",
			event_id: "evt_2",
			response_id: "resp_1",
			item_id: "fc_1",
			call_id: "call_1",
			output_index: 0,
			delta: '{"key":"',
		});
		instance.emitEvent({
			type: "response.function_call_arguments.done",
			event_id: "evt_3",
			response_id: "resp_1",
			item_id: "fc_1",
			call_id: "call_1",
			name: "lookup",
			output_index: 0,
			arguments: '{"key":"beta"}',
		});
		instance.emitEvent({
			type: "response.output_item.done",
			event_id: "evt_4",
			response_id: "resp_1",
			output_index: 0,
			item: {
				type: "function_call",
				id: "fc_1",
				call_id: "call_1",
				name: "lookup",
				arguments: '{"key":"beta"}',
				status: "completed",
			},
		});
		instance.emitEvent({ type: "response.done", event_id: "evt_5", response: { id: "resp_1", status: "completed" } });

		const result = await resultStream.result();
		expect(result.stopReason).toBe("toolUse");
		expect(result.content).toEqual([
			{ type: "toolCall", id: "call_1|fc_1", name: "lookup", arguments: { key: "beta" } },
		]);
		expect("partialJson" in result.content[0]!).toBe(false);
	});

	it("maps realtime errors and aborts into final assistant messages", async () => {
		const model = createModel();
		const context: Context = { messages: [{ role: "user", content: "Hi", timestamp: Date.now() }] };
		const errored = streamOpenAIRealtime(model, context, { apiKey: "sk-test" });
		await waitFor(() => latestRealtimeInstance().sent.length === 1);
		latestRealtimeInstance().emitEvent({
			type: "error",
			event_id: "evt_err",
			error: {
				message: "bad request",
				type: "invalid_request_error",
				code: "bad",
				param: null,
				event_id: null,
			},
		});
		const errorResult = await errored.result();
		expect(errorResult.stopReason).toBe("error");
		expect(errorResult.errorMessage).toContain("bad request");

		realtimeMock.MockOpenAIRealtimeWS.instances.length = 0;
		const controller = new AbortController();
		const aborted = streamOpenAIRealtime(model, context, { apiKey: "sk-test", signal: controller.signal });
		await waitFor(() => latestRealtimeInstance().sent.length === 1);
		controller.abort();
		const abortResult = await aborted.result();
		expect(abortResult.stopReason).toBe("aborted");
		expect(abortResult.errorMessage).toBe("Request was aborted");
	});

	it("maps simple xhigh reasoning to the realtime response payload", async () => {
		const model = createModel();
		const context: Context = { messages: [{ role: "user", content: "Hi", timestamp: Date.now() }] };

		const resultStream = streamSimpleOpenAIRealtime(model, context, {
			apiKey: "sk-test",
			reasoning: "xhigh",
		});
		await waitFor(() => latestRealtimeInstance().sent.length === 1);

		const instance = latestRealtimeInstance();
		const payload = instance.sent[0] as { response?: { reasoning?: { effort?: string } } };
		expect(payload.response?.reasoning).toEqual({ effort: "xhigh" });
		instance.emitEvent({ type: "response.done", event_id: "evt_1", response: { id: "resp_1", status: "completed" } });
		await resultStream.result();
	});
});
