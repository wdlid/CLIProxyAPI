package chat_completions

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertGeminiResponseToOpenAICompletionTokensIncludeThoughts(t *testing.T) {
	var param any
	chunk := []byte(`{"usageMetadata":{"promptTokenCount":16,"candidatesTokenCount":5,"thoughtsTokenCount":42,"totalTokenCount":63}}`)

	result := ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk, &param)
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	completionTokens := gjson.GetBytes(result[0], "usage.completion_tokens")
	if !completionTokens.Exists() || completionTokens.Int() != 47 {
		t.Fatalf("completion_tokens = %s, want present with value 47. Output: %s", completionTokens.Raw, result[0])
	}
}

func TestConvertGeminiResponseToOpenAINonStreamCompletionTokensIncludeThoughts(t *testing.T) {
	response := []byte(`{"usageMetadata":{"promptTokenCount":16,"thoughtsTokenCount":42,"totalTokenCount":58}}`)

	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)
	completionTokens := gjson.GetBytes(result, "usage.completion_tokens")
	if !completionTokens.Exists() || completionTokens.Int() != 42 {
		t.Fatalf("completion_tokens = %s, want present with value 42. Output: %s", completionTokens.Raw, result)
	}
}

func TestGeminiFinishReasonOnlyOnFinalChunk(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"list_dir","args":{"path":"C:/"}}}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"}}`)
	result1 := ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk1, &param)
	if len(result1) != 1 {
		t.Fatalf("expected 1 result from chunk1, got %d", len(result1))
	}
	fr1 := gjson.GetBytes(result1[0], "choices.0.finish_reason")
	if fr1.Exists() && fr1.String() != "" && fr1.Type.String() != "Null" {
		t.Fatalf("expected null finish_reason on tool chunk, got %v", fr1.String())
	}

	chunk2 := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"list_dir","args":{"path":"D:/"}}}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"}}`)
	ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk2, &param)

	chunk3 := []byte(`{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`)
	result3 := ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk3, &param)
	if len(result3) != 1 {
		t.Fatalf("expected 1 result from chunk3, got %d", len(result3))
	}
	fr3 := gjson.GetBytes(result3[0], "choices.0.finish_reason").String()
	if fr3 != "tool_calls" {
		t.Fatalf("expected finish_reason tool_calls, got %s", fr3)
	}
	nfr3 := gjson.GetBytes(result3[0], "choices.0.native_finish_reason").String()
	if nfr3 != "stop" {
		t.Fatalf("expected native_finish_reason stop, got %s", nfr3)
	}
}

func TestConvertGeminiResponseToOpenAINonStream_EmptyTextProducesEmptyString(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"text":""},{"text":"","thought":true}]},"finishReason":"STOP"}]}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "" || content.Type == gjson.Null {
		t.Fatalf("expected content to be empty string \"\", got %v (type %v)", content.Value(), content.Type)
	}

	reasoning := gjson.GetBytes(result, "choices.0.message.reasoning_content")
	if !reasoning.Exists() || reasoning.String() != "" || reasoning.Type == gjson.Null {
		t.Fatalf("expected reasoning_content to be empty string \"\", got %v (type %v)", reasoning.Value(), reasoning.Type)
	}
}

// Gemini 3.5 Transcribe returns its transcript in an audioTranscription part instead
// of the regular text part. Without handling it, the transcript is silently dropped
// and the OpenAI client sees an empty assistant message.
func TestConvertGeminiResponseToOpenAINonStream_AudioTranscriptionPartProducesContent(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"text":""},{"audioTranscription":{"text":"Hello world"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":185,"totalTokenCount":185}}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "Hello world" {
		t.Fatalf("expected content to carry the transcript, got %v. Output: %s", content.Raw, result)
	}
}

func TestConvertGeminiResponseToOpenAINonStream_SingleAudioTranscriptionPart(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"audioTranscription":{"text":"Single transcription part"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":185,"totalTokenCount":185}}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "Single transcription part" {
		t.Fatalf("expected content to carry the transcript, got %v. Output: %s", content.Raw, result)
	}
}

func TestConvertGeminiResponseToOpenAI_AudioTranscriptionPartStreamsContent(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk := []byte(`{"candidates":[{"content":{"parts":[{"text":""},{"audioTranscription":{"text":"Testing one two three."}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":185,"candidatesTokenCount":8,"totalTokenCount":193}}`)
	result := ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk, &param)
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}

	content := gjson.GetBytes(result[0], "choices.0.delta.content")
	if !content.Exists() || content.String() != "Testing one two three." {
		t.Fatalf("expected delta.content to carry the transcript, got %v. Output: %s", content.Raw, result[0])
	}
}

func TestConvertGeminiResponseToOpenAINonStream_TextPrecedenceOverAudioTranscription(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"text":"explicit text","audioTranscription":{"text":"ignored transcription"}}]},"finishReason":"STOP"}]}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "explicit text" {
		t.Fatalf("expected content to prioritize explicit text, got %v. Output: %s", content.Raw, result)
	}
}

func TestConvertGeminiResponseToOpenAI_PinsSanitizedNameMapPerStream(t *testing.T) {
	ctx := context.Background()

	// 1. Only the first chunk carries the request with a sanitized tool name.
	var param any
	reqWithTool := []byte(`{"model":"gemini-2.5-flash","tools":[{"name":"mcp/search"}]}`)
	chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"text":"thinking"}]}}]}`)
	chunk2 := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"mcp_search","args":{"q":"test"}}}]}}]}`)

	ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", reqWithTool, nil, chunk1, &param)
	out := ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", nil, nil, chunk2, &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk output, got %d", len(out))
	}
	if got := gjson.GetBytes(out[0], "choices.0.delta.tool_calls.0.function.name").String(); got != "mcp/search" {
		t.Fatalf("restored tool name = %q, want mcp/search", got)
	}

	// 2. When the first chunk's request produces a nil SanitizedNameMap, later chunks must not rebuild it.
	var nilMapParam any
	reqNoSanitize := []byte(`{"model":"gemini-2.5-flash","tools":[{"name":"plain_tool"}]}`)
	ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", reqNoSanitize, nil, chunk1, &nilMapParam)
	outNil := ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", reqWithTool, nil, chunk2, &nilMapParam)
	if len(outNil) != 1 {
		t.Fatalf("expected 1 chunk output, got %d", len(outNil))
	}
	if got := gjson.GetBytes(outNil[0], "choices.0.delta.tool_calls.0.function.name").String(); got != "mcp_search" {
		t.Fatalf("tool name = %q, want mcp_search (must not rebuild nil SanitizedNameMap on later chunks)", got)
	}
}

func TestConvertGeminiResponseToOpenAI_SynthesizesFinishReasonOnDoneWhenOmitted(t *testing.T) {
	ctx := context.Background()
	var param any

	// Tool call chunk with usageMetadata (or filtered cpaUsageMetadata) but no finishReason.
	filtered := []byte(`{"candidates":[{"content":{"parts":[{"thoughtSignature":"sig","functionCall":{"name":"bash","args":{"command":"ls -la"},"id":"call_1"}}],"role":"model"}}],"modelVersion":"gemini-4-argon-high","responseId":"resp-baic-nofr","cpaUsageMetadata":{"promptTokenCount":120,"candidatesTokenCount":15,"thoughtsTokenCount":30,"totalTokenCount":165}}`)
	result := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, filtered, &param)
	if len(result) != 1 {
		t.Fatalf("expected 1 intermediate chunk, got %d", len(result))
	}
	if fr := gjson.GetBytes(result[0], "choices.0.finish_reason").String(); fr != "" {
		t.Fatalf("intermediate finish_reason = %q, want empty/null", fr)
	}
	if usage := gjson.GetBytes(result[0], "usage"); usage.Exists() {
		t.Fatalf("intermediate filtered usage must not be emitted before terminal: %s", usage.Raw)
	}

	done := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, []byte("[DONE]"), &param)
	if len(done) != 1 {
		t.Fatalf("expected 1 synthesized terminal chunk on [DONE], got %d", len(done))
	}
	if fr := gjson.GetBytes(done[0], "choices.0.finish_reason").String(); fr != "tool_calls" {
		t.Fatalf("terminal finish_reason = %q, want tool_calls", fr)
	}
	if got := gjson.GetBytes(done[0], "usage.prompt_tokens").Int(); got != 120 {
		t.Fatalf("terminal prompt_tokens = %d, want 120", got)
	}
	if got := gjson.GetBytes(done[0], "usage.completion_tokens").Int(); got != 45 {
		t.Fatalf("terminal completion_tokens = %d, want 45", got)
	}
	if got := gjson.GetBytes(done[0], "usage.total_tokens").Int(); got != 165 {
		t.Fatalf("terminal total_tokens = %d, want 165", got)
	}
	if got := gjson.GetBytes(done[0], "model").String(); got != "gemini-4-argon-high" {
		t.Fatalf("terminal model = %q, want gemini-4-argon-high", got)
	}
	if got := gjson.GetBytes(done[0], "id").String(); got != "resp-baic-nofr" {
		t.Fatalf("terminal id = %q, want resp-baic-nofr", got)
	}
}

func TestConvertGeminiResponseToOpenAI_SplitFinishReasonAndPureUsageChunk(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],"modelVersion":"gemini-4-argon-high","responseId":"resp-split-pure"}`)
	res1 := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, chunk1, &param)
	if len(res1) != 1 {
		t.Fatalf("expected 1 result from chunk1, got %d", len(res1))
	}
	if fr := gjson.GetBytes(res1[0], "choices.0.finish_reason").String(); fr != "" {
		t.Fatalf("chunk1 finish_reason = %q, want empty before usage arrives", fr)
	}

	// Pure usageMetadata chunk without candidates array
	chunk2 := []byte(`{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},"modelVersion":"gemini-4-argon-high","responseId":"resp-split-pure"}`)
	res2 := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, chunk2, &param)
	if len(res2) != 1 {
		t.Fatalf("expected 1 result from chunk2, got %d", len(res2))
	}
	if fr := gjson.GetBytes(res2[0], "choices.0.finish_reason").String(); fr != "stop" {
		t.Fatalf("chunk2 finish_reason = %q, want stop", fr)
	}
	if got := gjson.GetBytes(res2[0], "usage.total_tokens").Int(); got != 15 {
		t.Fatalf("chunk2 total_tokens = %d, want 15", got)
	}

	// Subsequent [DONE] must not emit a duplicate finish_reason chunk
	done := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, []byte("[DONE]"), &param)
	if len(done) != 0 {
		t.Fatalf("expected 0 chunks on [DONE] after finish_reason already emitted, got %d", len(done))
	}
}

func TestConvertGeminiResponseToOpenAI_UnwrapsResponseEnvelopeAndHandlesEmptyStream(t *testing.T) {
	ctx := context.Background()

	// Empty stream must not synthesize finish_reason on [DONE]
	var emptyParam any
	if done := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, []byte("[DONE]"), &emptyParam); len(done) != 0 {
		t.Fatalf("empty stream must not synthesize terminal chunk, got %d", len(done))
	}

	// Wrapped {"response": ...} envelope
	var wrappedParam any
	wrapped := []byte(`{"response":{"candidates":[{"content":{"parts":[{"text":"wrapped hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6},"modelVersion":"gemini-4-argon-high","responseId":"resp-wrapped"}}`)
	res := ConvertGeminiResponseToOpenAI(ctx, "gemini-4-argon-high", nil, nil, wrapped, &wrappedParam)
	if len(res) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(res))
	}
	if got := gjson.GetBytes(res[0], "choices.0.delta.content").String(); got != "wrapped hello" {
		t.Fatalf("delta.content = %q, want wrapped hello", got)
	}
	if fr := gjson.GetBytes(res[0], "choices.0.finish_reason").String(); fr != "stop" {
		t.Fatalf("finish_reason = %q, want stop", fr)
	}
}

func BenchmarkConvertGeminiResponseToOpenAI_TextDelta(b *testing.B) {
	request := []byte(`{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"` + strings.Repeat("x", 1024*1024) + `"}]}`)
	delta := []byte(`{"candidates":[{"content":{"parts":[{"text":"token "}]}}],"modelVersion":"gemini-2.5-flash","responseId":"resp_bench"}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var param any
		for j := 0; j < 64; j++ {
			ConvertGeminiResponseToOpenAI(context.Background(), "gemini-2.5-flash", request, nil, delta, &param)
		}
	}
}
