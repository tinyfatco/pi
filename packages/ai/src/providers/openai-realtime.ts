import OpenAI from "openai";
import { OpenAIRealtimeWS } from "openai/realtime/ws.js";
import type {
	ConversationItem,
	RealtimeClientEvent,
	RealtimeFunctionTool,
	RealtimeResponse,
	RealtimeResponseCreateParams,
	RealtimeServerEvent,
} from "openai/resources/realtime/realtime.js";
import { calculateCost, clampThinkingLevel } from "../models.ts";
import type {
	Api,
	AssistantMessage,
	Context,
	ImageContent,
	Model,
	SimpleStreamOptions,
	StopReason,
	StreamFunction,
	StreamOptions,
	TextContent,
	TextSignatureV1,
	Tool,
	ToolCall,
} from "../types.ts";
import { AssistantMessageEventStream } from "../utils/event-stream.ts";
import { parseStreamingJson } from "../utils/json-parse.ts";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.ts";
import { transformMessages } from "./transform-messages.ts";

const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS = 15_000;

type RealtimeReasoningEffort = "minimal" | "low" | "medium" | "high" | "xhigh";

export interface OpenAIRealtimeOptions extends StreamOptions {
	reasoningEffort?: RealtimeReasoningEffort;
}

type RealtimeResponseCreateParamsWithCurrentDocs = RealtimeResponseCreateParams & {
	parallel_tool_calls?: boolean;
	reasoning?: { effort?: RealtimeReasoningEffort | string };
};

type RealtimeResponseCreateEventWithCurrentDocs = {
	type: "response.create";
	event_id?: string;
	response?: RealtimeResponseCreateParamsWithCurrentDocs;
};

type TextBlockState = {
	kind: "text";
	contentIndex: number;
	block: TextContent;
};

type ToolCallBlock = ToolCall & { partialJson?: string };

type ToolBlockState = {
	kind: "tool";
	contentIndex: number;
	block: ToolCallBlock;
};

type BlockState = TextBlockState | ToolBlockState;

type RealtimeQueueItem =
	| { type: "event"; event: RealtimeServerEvent }
	| { type: "error"; error: Error }
	| { type: "close"; error: Error };

interface SocketLike {
	readyState?: number;
	on(type: "open" | "error" | "close", listener: (...args: unknown[]) => void): void;
	off?(type: "open" | "error" | "close", listener: (...args: unknown[]) => void): void;
	removeListener?(type: "open" | "error" | "close", listener: (...args: unknown[]) => void): void;
}

interface RealtimeConnectionLike {
	socket: SocketLike;
	send(event: RealtimeClientEvent): void;
	close(props?: { code: number; reason: string }): void;
	on(type: "event", listener: (event: RealtimeServerEvent) => void): void;
	on(type: "error", listener: (error: Error) => void): void;
	off?(type: "event", listener: (event: RealtimeServerEvent) => void): void;
	off?(type: "error", listener: (error: Error) => void): void;
	removeListener?(type: "event", listener: (event: RealtimeServerEvent) => void): void;
	removeListener?(type: "error", listener: (error: Error) => void): void;
}

export const streamOpenAIRealtime: StreamFunction<"openai-realtime", OpenAIRealtimeOptions> = (
	model: Model<"openai-realtime">,
	context: Context,
	options?: OpenAIRealtimeOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api as Api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};
		let realtime: RealtimeConnectionLike | undefined;

		try {
			const apiKey = options?.apiKey;
			if (!apiKey) {
				throw new Error(`No API key for provider: ${model.provider}`);
			}

			const client = new OpenAI({
				apiKey,
				baseURL: model.baseUrl,
				dangerouslyAllowBrowser: true,
			});
			const headers = buildRealtimeHeaders(model.headers, options?.headers);
			realtime = await OpenAIRealtimeWS.create(client, {
				model: model.id,
				options: { headers },
			});
			await waitForRealtimeOpen(
				realtime,
				options?.signal,
				normalizeTimeoutMs(options?.websocketConnectTimeoutMs) ?? DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS,
			);
			await options?.onResponse?.({ status: 101, headers: {} }, model);

			stream.push({ type: "start", partial: output });

			let payload = buildResponseCreateEvent(model, context, options);
			const nextPayload = await options?.onPayload?.(payload, model);
			if (nextPayload !== undefined) {
				payload = nextPayload as RealtimeResponseCreateEventWithCurrentDocs;
			}

			realtime.send(payload as RealtimeClientEvent);
			await processRealtimeStream(
				readRealtimeEvents(realtime, options?.signal, normalizeTimeoutMs(options?.timeoutMs)),
				output,
				stream,
				model,
			);

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}
			if (output.stopReason === "aborted" || output.stopReason === "error") {
				throw new Error(output.errorMessage || "An unknown error occurred");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			cleanupStreamingScratch(output);
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = formatOpenAIRealtimeError(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		} finally {
			closeRealtimeSilently(realtime, 1000, "done");
		}
	})();

	return stream;
};

export const streamSimpleOpenAIRealtime: StreamFunction<"openai-realtime", SimpleStreamOptions> = (
	model: Model<"openai-realtime">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	const apiKey = options?.apiKey;
	if (!apiKey) {
		throw new Error(`No API key for provider: ${model.provider}`);
	}

	const clampedReasoning = options?.reasoning ? clampThinkingLevel(model, options.reasoning) : undefined;
	const reasoningEffort = clampedReasoning === "off" ? undefined : clampedReasoning;

	return streamOpenAIRealtime(model, context, {
		temperature: options?.temperature,
		maxTokens: options?.maxTokens,
		signal: options?.signal,
		apiKey,
		transport: options?.transport,
		cacheRetention: options?.cacheRetention,
		sessionId: options?.sessionId,
		headers: options?.headers,
		onPayload: options?.onPayload,
		onResponse: options?.onResponse,
		timeoutMs: options?.timeoutMs,
		websocketConnectTimeoutMs: options?.websocketConnectTimeoutMs,
		maxRetries: options?.maxRetries,
		maxRetryDelayMs: options?.maxRetryDelayMs,
		metadata: options?.metadata,
		reasoningEffort: reasoningEffort as RealtimeReasoningEffort | undefined,
	} satisfies OpenAIRealtimeOptions);
};

export function convertRealtimeMessages(model: Model<"openai-realtime">, context: Context): ConversationItem[] {
	const transformedMessages = transformMessages(context.messages, model, normalizeRealtimeToolCallId);
	const input: ConversationItem[] = [];
	let messageIndex = 0;

	for (const message of transformedMessages) {
		if (message.role === "user") {
			if (typeof message.content === "string") {
				input.push({
					type: "message",
					role: "user",
					content: [{ type: "input_text", text: sanitizeSurrogates(message.content) }],
				});
			} else {
				const content = message.content.map((item) => {
					if (item.type === "text") {
						return { type: "input_text" as const, text: sanitizeSurrogates(item.text) };
					}
					return {
						type: "input_image" as const,
						detail: "auto" as const,
						image_url: `data:${item.mimeType};base64,${item.data}`,
					};
				});
				if (content.length > 0) {
					input.push({ type: "message", role: "user", content });
				}
			}
		} else if (message.role === "assistant") {
			let textBlockIndex = 0;
			for (const item of message.content) {
				if (item.type === "text") {
					input.push({
						type: "message",
						role: "assistant",
						id: getAssistantMessageItemId(item, messageIndex, textBlockIndex),
						status: "completed",
						content: [{ type: "output_text", text: sanitizeSurrogates(item.text) }],
					});
					textBlockIndex++;
				} else if (item.type === "toolCall") {
					const { callId, itemId } = splitRealtimeToolCallId(item.id);
					input.push({
						type: "function_call",
						id: itemId,
						call_id: callId,
						name: item.name,
						arguments: JSON.stringify(item.arguments),
						status: "completed",
					});
				}
			}
		} else if (message.role === "toolResult") {
			const textResult = message.content
				.filter((item): item is TextContent => item.type === "text")
				.map((item) => item.text)
				.join("\n");
			const hasImages = message.content.some((item): item is ImageContent => item.type === "image");
			const { callId } = splitRealtimeToolCallId(message.toolCallId);
			input.push({
				type: "function_call_output",
				call_id: callId,
				output: sanitizeSurrogates(textResult.length > 0 ? textResult : hasImages ? "(see attached image)" : ""),
			});
		}
		messageIndex++;
	}

	return input;
}

export function convertRealtimeTools(tools: Tool[]): RealtimeFunctionTool[] {
	return tools.map((tool) => ({
		type: "function",
		name: tool.name,
		description: tool.description,
		parameters: tool.parameters,
	}));
}

export function buildResponseCreateEvent(
	model: Model<"openai-realtime">,
	context: Context,
	options?: OpenAIRealtimeOptions,
): RealtimeResponseCreateEventWithCurrentDocs {
	const response: RealtimeResponseCreateParamsWithCurrentDocs = {
		conversation: "none",
		input: convertRealtimeMessages(model, context),
		instructions: context.systemPrompt ? sanitizeSurrogates(context.systemPrompt) : undefined,
		output_modalities: ["text"],
		tool_choice: context.tools && context.tools.length > 0 ? "auto" : undefined,
		tools: context.tools && context.tools.length > 0 ? convertRealtimeTools(context.tools) : undefined,
		parallel_tool_calls: context.tools && context.tools.length > 0 ? true : undefined,
	};

	if (options?.maxTokens !== undefined) {
		response.max_output_tokens = options.maxTokens;
	}
	if (options?.reasoningEffort !== undefined) {
		response.reasoning = { effort: model.thinkingLevelMap?.[options.reasoningEffort] ?? options.reasoningEffort };
	}

	return {
		type: "response.create",
		response,
	};
}

export async function processRealtimeStream(
	events: AsyncIterable<RealtimeServerEvent>,
	output: AssistantMessage,
	stream: AssistantMessageEventStream,
	model: Model<"openai-realtime">,
): Promise<void> {
	const blockStates = new Map<string, BlockState>();
	const outputIndexToKey = new Map<number, string>();
	const blocks = output.content;
	const blockIndex = () => blocks.length - 1;

	for await (const event of events) {
		if (event.type === "response.created") {
			output.responseId = event.response.id;
		} else if (event.type === "response.output_item.added") {
			const key = getOutputItemKey(event.item, event.output_index);
			outputIndexToKey.set(event.output_index, key);
			if (event.item.type === "message" && event.item.role === "assistant") {
				const block: TextContent = { type: "text", text: "" };
				output.content.push(block);
				blockStates.set(key, { kind: "text", contentIndex: blockIndex(), block });
				stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
			} else if (event.item.type === "function_call") {
				const callId = event.item.call_id || event.item.id || `call_${event.output_index}`;
				const toolCallId = joinRealtimeToolCallId(callId, event.item.id);
				const block: ToolCallBlock = {
					type: "toolCall",
					id: toolCallId,
					name: event.item.name,
					arguments: parseStreamingJson(event.item.arguments),
					partialJson: event.item.arguments || "",
				};
				output.content.push(block);
				blockStates.set(key, { kind: "tool", contentIndex: blockIndex(), block });
				stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
			}
		} else if (event.type === "response.output_text.delta") {
			const state = getBlockState(blockStates, outputIndexToKey, event.item_id, event.output_index);
			if (state?.kind === "text") {
				state.block.text += event.delta;
				stream.push({ type: "text_delta", contentIndex: state.contentIndex, delta: event.delta, partial: output });
			}
		} else if (event.type === "response.output_text.done") {
			const state = getBlockState(blockStates, outputIndexToKey, event.item_id, event.output_index);
			if (state?.kind === "text") {
				state.block.text = event.text;
			}
		} else if (event.type === "response.function_call_arguments.delta") {
			const state = getBlockState(blockStates, outputIndexToKey, event.item_id, event.output_index);
			if (state?.kind === "tool") {
				state.block.partialJson = (state.block.partialJson || "") + event.delta;
				state.block.arguments = parseStreamingJson(state.block.partialJson);
				stream.push({
					type: "toolcall_delta",
					contentIndex: state.contentIndex,
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.function_call_arguments.done") {
			const state = getBlockState(blockStates, outputIndexToKey, event.item_id, event.output_index);
			if (state?.kind === "tool") {
				const previousPartialJson = state.block.partialJson || "";
				state.block.partialJson = event.arguments;
				state.block.name = event.name;
				state.block.id = joinRealtimeToolCallId(event.call_id, event.item_id);
				state.block.arguments = parseStreamingJson(state.block.partialJson);

				if (event.arguments.startsWith(previousPartialJson)) {
					const delta = event.arguments.slice(previousPartialJson.length);
					if (delta.length > 0) {
						stream.push({
							type: "toolcall_delta",
							contentIndex: state.contentIndex,
							delta,
							partial: output,
						});
					}
				}
			}
		} else if (event.type === "response.output_item.done") {
			const key = getOutputItemKey(event.item, event.output_index);
			const state = getBlockState(
				blockStates,
				outputIndexToKey,
				getConversationItemId(event.item),
				event.output_index,
			);
			if (event.item.type === "message" && event.item.role === "assistant") {
				const text = getAssistantMessageText(event.item);
				const textState = state?.kind === "text" ? state : createTextState(output, stream);
				textState.block.text = text;
				textState.block.textSignature = event.item.id;
				stream.push({
					type: "text_end",
					contentIndex: textState.contentIndex,
					content: textState.block.text,
					partial: output,
				});
				blockStates.delete(key);
			} else if (event.item.type === "function_call") {
				const toolState =
					state?.kind === "tool" ? state : createToolState(output, stream, event.item, event.output_index);
				const fallbackCallId = splitRealtimeToolCallId(toolState.block.id).callId;
				toolState.block.id = joinRealtimeToolCallId(event.item.call_id || fallbackCallId, event.item.id);
				toolState.block.name = event.item.name;
				toolState.block.partialJson = event.item.arguments || toolState.block.partialJson;
				toolState.block.arguments = parseStreamingJson(toolState.block.partialJson);
				delete toolState.block.partialJson;
				stream.push({
					type: "toolcall_end",
					contentIndex: toolState.contentIndex,
					toolCall: toolState.block,
					partial: output,
				});
				blockStates.delete(key);
			}
		} else if (event.type === "response.done") {
			if (event.response?.id) {
				output.responseId = event.response.id;
			}
			updateUsage(output, event.response, model);
			output.stopReason = mapRealtimeStopReason(event.response);
			if (output.stopReason === "error") {
				output.errorMessage = formatRealtimeResponseError(event.response);
			}
			if (output.content.some((block) => block.type === "toolCall") && output.stopReason === "stop") {
				output.stopReason = "toolUse";
			}
			finalizeOpenBlocks(blockStates, stream, output);
			return;
		} else if (event.type === "error") {
			throw new Error(formatRealtimeEventError(event));
		}
	}
}

function buildRealtimeHeaders(
	modelHeaders: Record<string, string> | undefined,
	optionsHeaders: Record<string, string> | undefined,
): Record<string, string> {
	return {
		...modelHeaders,
		...optionsHeaders,
	};
}

function normalizeTimeoutMs(value: number | undefined): number | undefined {
	if (value === undefined) return undefined;
	if (!Number.isFinite(value) || value < 0) {
		throw new Error(`Invalid timeoutMs: ${String(value)}`);
	}
	return Math.floor(value);
}

function addSocketListener(
	socket: SocketLike,
	type: "open" | "error" | "close",
	listener: (...args: unknown[]) => void,
): void {
	socket.on(type, listener);
}

function removeSocketListener(
	socket: SocketLike,
	type: "open" | "error" | "close",
	listener: (...args: unknown[]) => void,
): void {
	if (socket.off) {
		socket.off(type, listener);
		return;
	}
	socket.removeListener?.(type, listener);
}

function removeRealtimeListener(
	realtime: RealtimeConnectionLike,
	type: "event",
	listener: (event: RealtimeServerEvent) => void,
): void;
function removeRealtimeListener(
	realtime: RealtimeConnectionLike,
	type: "error",
	listener: (error: Error) => void,
): void;
function removeRealtimeListener(
	realtime: RealtimeConnectionLike,
	type: "event" | "error",
	listener: ((event: RealtimeServerEvent) => void) | ((error: Error) => void),
): void {
	if (type === "event") {
		const typedListener = listener as (event: RealtimeServerEvent) => void;
		if (realtime.off) {
			realtime.off(type, typedListener);
			return;
		}
		realtime.removeListener?.(type, typedListener);
		return;
	}
	const typedListener = listener as (error: Error) => void;
	if (realtime.off) {
		realtime.off(type, typedListener);
		return;
	}
	realtime.removeListener?.(type, typedListener);
}

function waitForRealtimeOpen(
	realtime: RealtimeConnectionLike,
	signal?: AbortSignal,
	connectTimeoutMs = DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS,
): Promise<void> {
	if (signal?.aborted) {
		closeRealtimeSilently(realtime, 1000, "aborted");
		return Promise.reject(new Error("Request was aborted"));
	}
	if (realtime.socket.readyState === 1) {
		return Promise.resolve();
	}

	return new Promise<void>((resolve, reject) => {
		let settled = false;
		let timeout: ReturnType<typeof setTimeout> | undefined;

		const cleanup = () => {
			if (timeout) {
				clearTimeout(timeout);
				timeout = undefined;
			}
			removeSocketListener(realtime.socket, "open", onOpen);
			removeSocketListener(realtime.socket, "error", onError);
			removeSocketListener(realtime.socket, "close", onClose);
			signal?.removeEventListener("abort", onAbort);
		};
		const fail = (error: Error, closeReason?: string) => {
			if (settled) return;
			settled = true;
			cleanup();
			if (closeReason) {
				closeRealtimeSilently(realtime, 1000, closeReason);
			}
			reject(error);
		};
		const onOpen = () => {
			if (settled) return;
			settled = true;
			cleanup();
			resolve();
		};
		const onError = (...args: unknown[]) => {
			fail(extractErrorFromUnknown(args[0], "Realtime WebSocket error"));
		};
		const onClose = (...args: unknown[]) => {
			fail(new Error(formatWebSocketClose(args)));
		};
		const onAbort = () => {
			fail(new Error("Request was aborted"), "aborted");
		};

		addSocketListener(realtime.socket, "open", onOpen);
		addSocketListener(realtime.socket, "error", onError);
		addSocketListener(realtime.socket, "close", onClose);
		signal?.addEventListener("abort", onAbort, { once: true });
		if (connectTimeoutMs > 0) {
			timeout = setTimeout(() => {
				fail(new Error(`Realtime WebSocket connect timeout after ${connectTimeoutMs}ms`), "connect_timeout");
			}, connectTimeoutMs);
		}
	});
}

async function* readRealtimeEvents(
	realtime: RealtimeConnectionLike,
	signal?: AbortSignal,
	idleTimeoutMs?: number,
): AsyncGenerator<RealtimeServerEvent> {
	const queue: RealtimeQueueItem[] = [];
	let pending: (() => void) | undefined;
	let done = false;

	const wake = () => {
		if (!pending) return;
		const resolve = pending;
		pending = undefined;
		resolve();
	};
	const enqueue = (item: RealtimeQueueItem) => {
		queue.push(item);
		wake();
	};
	const onEvent = (event: RealtimeServerEvent) => enqueue({ type: "event", event });
	const onError = (error: Error) => {
		done = true;
		enqueue({ type: "error", error });
	};
	const onClose = (...args: unknown[]) => {
		done = true;
		enqueue({ type: "close", error: new Error(formatWebSocketClose(args)) });
	};
	const onAbort = () => {
		done = true;
		closeRealtimeSilently(realtime, 1000, "aborted");
		enqueue({ type: "error", error: new Error("Request was aborted") });
	};

	realtime.on("event", onEvent);
	realtime.on("error", onError);
	addSocketListener(realtime.socket, "close", onClose);
	signal?.addEventListener("abort", onAbort, { once: true });

	try {
		while (true) {
			if (signal?.aborted) {
				throw new Error("Request was aborted");
			}
			const item = queue.shift();
			if (item) {
				if (item.type === "event") {
					yield item.event;
					continue;
				}
				throw item.error;
			}
			if (done) break;

			let timeout: ReturnType<typeof setTimeout> | undefined;
			await new Promise<void>((resolve, reject) => {
				pending = resolve;
				if (idleTimeoutMs !== undefined && idleTimeoutMs > 0) {
					timeout = setTimeout(() => {
						const error = new Error(`Realtime WebSocket idle timeout after ${idleTimeoutMs}ms`);
						done = true;
						pending = undefined;
						closeRealtimeSilently(realtime, 1000, "idle_timeout");
						reject(error);
					}, idleTimeoutMs);
				}
			}).finally(() => {
				if (timeout) {
					clearTimeout(timeout);
				}
			});
		}
	} finally {
		removeRealtimeListener(realtime, "event", onEvent);
		removeRealtimeListener(realtime, "error", onError);
		removeSocketListener(realtime.socket, "close", onClose);
		signal?.removeEventListener("abort", onAbort);
	}
}

function closeRealtimeSilently(realtime: RealtimeConnectionLike | undefined, code = 1000, reason = "done"): void {
	try {
		realtime?.close({ code, reason });
	} catch {}
}

function parseTextSignature(signature: string | undefined): { id: string } | undefined {
	if (!signature) return undefined;
	if (signature.startsWith("{")) {
		try {
			const parsed = JSON.parse(signature) as Partial<TextSignatureV1>;
			if (parsed.v === 1 && typeof parsed.id === "string") {
				return { id: parsed.id };
			}
		} catch {
			// Fall through to legacy plain-string handling.
		}
	}
	return { id: signature };
}

function getAssistantMessageItemId(text: TextContent, messageIndex: number, textBlockIndex: number): string {
	const parsed = parseTextSignature(text.textSignature);
	const fallback = textBlockIndex === 0 ? `msg_pi_${messageIndex}` : `msg_pi_${messageIndex}_${textBlockIndex}`;
	return normalizeRealtimeIdPart(parsed?.id || fallback);
}

function normalizeRealtimeIdPart(id: string): string {
	const sanitized = id.replace(/[^a-zA-Z0-9_-]/g, "_");
	const normalized = sanitized.length > 64 ? sanitized.slice(0, 64) : sanitized;
	return normalized.replace(/_+$/, "") || "item";
}

function splitRealtimeToolCallId(id: string): { callId: string; itemId?: string } {
	const [callIdRaw, itemIdRaw] = id.split("|");
	return {
		callId: normalizeRealtimeIdPart(callIdRaw || id),
		itemId: itemIdRaw ? normalizeRealtimeIdPart(itemIdRaw) : undefined,
	};
}

function joinRealtimeToolCallId(callId: string, itemId: string | undefined): string {
	const normalizedCallId = normalizeRealtimeIdPart(callId);
	return itemId ? `${normalizedCallId}|${normalizeRealtimeIdPart(itemId)}` : normalizedCallId;
}

function normalizeRealtimeToolCallId(id: string): string {
	const { callId, itemId } = splitRealtimeToolCallId(id);
	return itemId ? `${callId}|${itemId}` : callId;
}

function getConversationItemId(item: ConversationItem): string | undefined {
	return "id" in item && typeof item.id === "string" ? item.id : undefined;
}

function getOutputItemKey(item: ConversationItem, outputIndex: number): string {
	const itemId = getConversationItemId(item);
	return itemId ? normalizeRealtimeIdPart(itemId) : `output_index:${outputIndex}`;
}

function getBlockState(
	states: Map<string, BlockState>,
	outputIndexToKey: Map<number, string>,
	itemId: string | undefined,
	outputIndex: number,
): BlockState | undefined {
	const key = itemId ? normalizeRealtimeIdPart(itemId) : outputIndexToKey.get(outputIndex);
	return key ? states.get(key) : undefined;
}

function getAssistantMessageText(item: Extract<ConversationItem, { type: "message" }>): string {
	return item.content
		.map((part) => {
			if (part.type === "output_text") return part.text || "";
			if (part.type === "output_audio") return part.transcript || "";
			return "";
		})
		.join("");
}

function createTextState(output: AssistantMessage, stream: AssistantMessageEventStream): TextBlockState {
	const block: TextContent = { type: "text", text: "" };
	output.content.push(block);
	const state = { kind: "text" as const, contentIndex: output.content.length - 1, block };
	stream.push({ type: "text_start", contentIndex: state.contentIndex, partial: output });
	return state;
}

function createToolState(
	output: AssistantMessage,
	stream: AssistantMessageEventStream,
	item: Extract<ConversationItem, { type: "function_call" }>,
	outputIndex: number,
): ToolBlockState {
	const block: ToolCallBlock = {
		type: "toolCall",
		id: joinRealtimeToolCallId(item.call_id || item.id || `call_${outputIndex}`, item.id),
		name: item.name,
		arguments: parseStreamingJson(item.arguments),
		partialJson: item.arguments || "",
	};
	output.content.push(block);
	const state = { kind: "tool" as const, contentIndex: output.content.length - 1, block };
	stream.push({ type: "toolcall_start", contentIndex: state.contentIndex, partial: output });
	return state;
}

function finalizeOpenBlocks(
	states: Map<string, BlockState>,
	stream: AssistantMessageEventStream,
	output: AssistantMessage,
): void {
	for (const state of states.values()) {
		if (state.kind === "text") {
			stream.push({
				type: "text_end",
				contentIndex: state.contentIndex,
				content: state.block.text,
				partial: output,
			});
		} else {
			state.block.arguments = parseStreamingJson(state.block.partialJson);
			delete state.block.partialJson;
			stream.push({
				type: "toolcall_end",
				contentIndex: state.contentIndex,
				toolCall: state.block,
				partial: output,
			});
		}
	}
	states.clear();
}

function updateUsage(output: AssistantMessage, response: RealtimeResponse, model: Model<"openai-realtime">): void {
	const usage = response.usage;
	if (!usage) {
		calculateCost(model, output.usage);
		return;
	}

	const cachedTokens = usage.input_token_details?.cached_tokens || 0;
	const inputTokens = usage.input_tokens || 0;
	const outputTokens = usage.output_tokens || 0;
	output.usage = {
		input: Math.max(0, inputTokens - cachedTokens),
		output: outputTokens,
		cacheRead: cachedTokens,
		cacheWrite: 0,
		totalTokens: usage.total_tokens || inputTokens + outputTokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	calculateCost(model, output.usage);
}

function mapRealtimeStopReason(response: RealtimeResponse | undefined): StopReason {
	const status = response?.status;
	switch (status) {
		case "completed":
			return "stop";
		case "incomplete":
			return "length";
		case "cancelled":
		case "failed":
			return "error";
		case "in_progress":
		case undefined:
			return "stop";
		default: {
			const _exhaustive: never = status;
			throw new Error(`Unhandled Realtime response status: ${_exhaustive}`);
		}
	}
}

function formatRealtimeResponseError(response: RealtimeResponse | undefined): string {
	const status = response?.status || "unknown";
	const details = response?.status_details;
	const error = details?.error;
	const reason = details?.reason;
	const code = error?.code || error?.type;
	return ["Realtime response", status, code, reason].filter((part): part is string => !!part).join(": ");
}

function formatRealtimeEventError(event: Extract<RealtimeServerEvent, { type: "error" }>): string {
	const error = event.error;
	const code = error.code ? ` code=${error.code}` : "";
	const param = error.param ? ` param=${error.param}` : "";
	const type = error.type ? ` type=${error.type}` : "";
	return `${error.message}${code}${param}${type}`;
}

function formatOpenAIRealtimeError(error: unknown): string {
	if (error instanceof Error) {
		return error.message;
	}
	try {
		return JSON.stringify(error);
	} catch {
		return String(error);
	}
}

function extractErrorFromUnknown(value: unknown, fallback: string): Error {
	if (value instanceof Error) return value;
	if (value && typeof value === "object" && "message" in value) {
		const message = (value as { message?: unknown }).message;
		if (typeof message === "string" && message.length > 0) return new Error(message);
	}
	if (typeof value === "string" && value.length > 0) return new Error(value);
	return new Error(fallback);
}

function formatWebSocketClose(args: unknown[]): string {
	const first = args[0];
	if (first && typeof first === "object") {
		const code = "code" in first && typeof first.code === "number" ? first.code : undefined;
		const reason = "reason" in first ? formatReason(first.reason) : undefined;
		return `Realtime WebSocket closed${code === undefined ? "" : ` ${code}`}${reason ? ` ${reason}` : ""}`.trim();
	}
	const code = typeof first === "number" ? first : undefined;
	const reason = formatReason(args[1]);
	return `Realtime WebSocket closed${code === undefined ? "" : ` ${code}`}${reason ? ` ${reason}` : ""}`.trim();
}

function formatReason(reason: unknown): string | undefined {
	if (typeof reason === "string") return reason.length > 0 ? reason : undefined;
	if (reason instanceof Uint8Array) {
		const text = new TextDecoder().decode(reason);
		return text.length > 0 ? text : undefined;
	}
	return undefined;
}

function cleanupStreamingScratch(output: AssistantMessage): void {
	for (const block of output.content) {
		delete (block as { partialJson?: string }).partialJson;
	}
}
