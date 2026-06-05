import { describe, expect, it } from "vitest";
import { getModel, getModelSessionMode, isRealtimeModel, isTurnBasedModel, type Model } from "../src/index.ts";

describe("model session mode", () => {
	it("defaults existing model metadata to turn-based", () => {
		const model = getModel("openai", "gpt-5.5");

		expect(getModelSessionMode(model)).toBe("turn");
		expect(isTurnBasedModel(model)).toBe(true);
		expect(isRealtimeModel(model)).toBe(false);
	});

	it("recognizes explicitly realtime model metadata", () => {
		const turnModel = getModel("openai", "gpt-5.5");
		const realtimeModel: Model<typeof turnModel.api> = {
			...turnModel,
			id: "gpt-realtime-2",
			name: "GPT Realtime 2",
			sessionMode: "realtime",
		};

		expect(getModelSessionMode(realtimeModel)).toBe("realtime");
		expect(isTurnBasedModel(realtimeModel)).toBe(false);
		expect(isRealtimeModel(realtimeModel)).toBe(true);
	});
});
