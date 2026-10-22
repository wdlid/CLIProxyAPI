// This file implements experimental support for Antigravity's Enterprise (BAIC / "Business AI
// Code") backend. Enterprise-tier accounts (user_tier prefixed "gcp-ge-") route actual
// generation traffic to businessaicode.<region>.rep.googleapis.com instead of the free/individual
// cloudcode-pa backend used elsewhere in this file. The exact wire format was not fully captured
// (only the request URL was observed from the official client's debug log), so this reuses the
// standard "gemini" translator payload shape, matching how the Vertex executor talks to Google's
// public Generative AI API. Expect to need adjustments once real responses are observed.
package executor

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
	internalsignature "github.com/router-for-me/CLIProxyAPI/v8/internal/signature"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/usage"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	antigravityBAICHostDefault         = "https://businessaicode.googleapis.com"
	antigravityBAICHostFormat          = "https://businessaicode.%s.rep.googleapis.com"
	antigravityBAICAPIVersion          = "v1beta"
	antigravityDefaultEnterpriseRegion = "global"
)

// isAntigravityEnterpriseTier reports whether auth was onboarded under a Gemini Enterprise
// tier (e.g. "gcp-ge-standard-tier"), which requires routing generation traffic to BAIC.
func isAntigravityEnterpriseTier(auth *cliproxyauth.Auth) bool {
	if auth == nil || auth.Metadata == nil {
		return false
	}
	tier, _ := auth.Metadata["user_tier"].(string)
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(tier)), "gcp-ge-")
}

// antigravityEnterpriseRegion resolves the GCP region BAIC requests must target. The region
// reported by businessaicode's fetchLicenses call at login time (stored as auth.Metadata["region"])
// takes precedence, since it's the authoritative value for the account's license.
func antigravityEnterpriseRegion(auth *cliproxyauth.Auth) string {
	if auth != nil && auth.Metadata != nil {
		if region, ok := auth.Metadata["region"].(string); ok {
			if region = strings.TrimSpace(region); region != "" {
				return region
			}
		}
	}
	if v := strings.TrimSpace(os.Getenv("ANTIGRAVITY_ENTERPRISE_REGION")); v != "" {
		return v
	}
	return antigravityDefaultEnterpriseRegion
}

func antigravityBAICURL(projectID, region, action string) string {
	var host string
	if region == "" || region == "global" {
		host = antigravityBAICHostDefault
		region = "global"
	} else {
		host = fmt.Sprintf(antigravityBAICHostFormat, region)
	}
	return fmt.Sprintf("%s/%s/projects/%s/locations/%s:%s", host, antigravityBAICAPIVersion, projectID, region, action)
}

// stripUnsupportedBAICSafetyCategories removes safetySettings entries whose category isn't
// recognized by businessaicode's HarmCategory enum (e.g. HARM_CATEGORY_CIVIC_INTEGRITY, which
// the public Gemini/Vertex APIs support but BAIC rejects with INVALID_ARGUMENT).
func stripUnsupportedBAICSafetyCategories(body []byte) []byte {
	settings := gjson.GetBytes(body, "safetySettings")
	if !settings.IsArray() {
		return body
	}
	kept := make([]any, 0, len(settings.Array()))
	for _, entry := range settings.Array() {
		if entry.Get("category").String() == "HARM_CATEGORY_CIVIC_INTEGRITY" {
			continue
		}
		kept = append(kept, entry.Value())
	}
	updated, err := sjson.SetBytes(body, "safetySettings", kept)
	if err != nil {
		return body
	}
	return updated
}

// antigravityEnterpriseTier returns the auth's onboarded tier ID (e.g. "gcp-ge-standard-tier").
func antigravityEnterpriseTier(auth *cliproxyauth.Auth) string {
	if auth == nil || auth.Metadata == nil {
		return ""
	}
	tier, _ := auth.Metadata["user_tier"].(string)
	return strings.TrimSpace(tier)
}

// setBAICEntitlement sets the top-level "entitlement" field BAIC requires, identifying which
// license/SKU the request should be billed against.
func setBAICEntitlement(body []byte, tier string) []byte {
	if tier == "" {
		return body
	}
	// BAIC expects entitlement as an Entitlement message ({"userTier": tier}), not a bare string.
	updated, err := sjson.SetBytes(body, "entitlement.userTier", tier)
	if err != nil {
		return body
	}
	return updated
}

// setBAICModelExperience selects the model via "aicode.experience" instead of a top-level "model"
// field, which BAIC's generateContent/streamGenerateContent surfaces reject as an invalid argument.
func setBAICModelExperience(body []byte, model string) []byte {
	updated, err := sjson.DeleteBytes(body, "model")
	if err == nil {
		body = updated
	}
	if model == "" {
		return body
	}
	updated, err = sjson.SetBytes(body, "aicode.experience", model)
	if err != nil {
		return body
	}
	return updated
}

// executeBAIC performs a non-streaming request against the Enterprise BAIC backend.
func (e *AntigravityExecutor) executeBAIC(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	token, updatedAuth, errToken := e.ensureAccessToken(ctx, auth)
	if errToken != nil {
		return resp, errToken
	}
	if updatedAuth != nil {
		auth = updatedAuth
	}
	projectID, errProject := e.projectIDForRequest(ctx, auth, token)
	if errProject != nil {
		return resp, errProject
	}

	from := opts.SourceFormat
	responseFormat := cliproxyexecutor.ResponseFormatOrSource(opts)
	to := sdktranslator.FromString("gemini")

	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := helps.TranslateRequestWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, baseModel, originalPayload, false)
	body := helps.TranslateRequestWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, baseModel, req.Payload, false)

	body, err = helps.ApplyRequestThinking(body, req, opts, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	body = e.obfuscateSensitiveWords(body)
	body = setBAICModelExperience(body, baseModel)
	body, _ = sjson.DeleteBytes(body, "session_id")
	body = helps.StripVertexOpenAIResponsesToolCallIDs(body, from.String())
	body = internalsignature.SanitizeGeminiRequestThoughtSignatures(body, "contents")
	body = helps.EnsureGeminiLeadingUserContent(body, "contents")
	body = helps.EnsureGeminiTrailingUserContent(body, "contents")
	body = stripUnsupportedBAICSafetyCategories(body)
	body = setBAICEntitlement(body, antigravityEnterpriseTier(auth))
	reporter.SetTranslatedReasoningEffort(body, to.String())
	body = helps.ApplyPayloadConfigWithRequest(e.cfg, baseModel, to.String(), from.String(), "", body, originalTranslated, requestedModel, requestPath, opts.Headers)

	region := antigravityEnterpriseRegion(auth)
	requestURL := antigravityBAICURL(projectID, region, "generateContent")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
	if errNewReq != nil {
		return resp, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	applyGeminiHeaders(httpReq, auth, opts.Headers)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs, opts.Headers)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       requestURL,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newAntigravityHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("antigravity BAIC executor: close response body error: %v", errClose)
		}
	}()
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("BAIC request error, status: %d, message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		return resp, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errRead)
		return resp, errRead
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, data)
	reporter.Publish(ctx, helps.ParseGeminiUsage(data))
	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, responseFormat, req.Model, opts.OriginalRequest, body, data, &param)
	if responseFormat == sdktranslator.FormatOpenAIResponse {
		out = helps.EnsureResponsesUsageDetails(out)
	}
	resp = cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}
	return resp, nil
}

// executeStreamBAIC performs a streaming request against the Enterprise BAIC backend.
func (e *AntigravityExecutor) executeStreamBAIC(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	token, updatedAuth, errToken := e.ensureAccessToken(ctx, auth)
	if errToken != nil {
		return nil, errToken
	}
	if updatedAuth != nil {
		auth = updatedAuth
	}
	projectID, errProject := e.projectIDForRequest(ctx, auth, token)
	if errProject != nil {
		return nil, errProject
	}

	from := opts.SourceFormat
	responseFormat := cliproxyexecutor.ResponseFormatOrSource(opts)
	to := sdktranslator.FromString("gemini")

	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := helps.TranslateRequestWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, baseModel, originalPayload, true)
	body := helps.TranslateRequestWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, baseModel, req.Payload, true)

	body, err = helps.ApplyRequestThinking(body, req, opts, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	body = e.obfuscateSensitiveWords(body)
	body = setBAICModelExperience(body, baseModel)
	body, _ = sjson.DeleteBytes(body, "session_id")
	body = helps.StripVertexOpenAIResponsesToolCallIDs(body, from.String())
	body = internalsignature.SanitizeGeminiRequestThoughtSignatures(body, "contents")
	body = helps.EnsureGeminiBoundaryUserContent(body, "contents")
	body = stripUnsupportedBAICSafetyCategories(body)
	body = setBAICEntitlement(body, antigravityEnterpriseTier(auth))
	reporter.SetTranslatedReasoningEffort(body, to.String())
	body = helps.ApplyPayloadConfigWithRequest(e.cfg, baseModel, to.String(), from.String(), "", body, originalTranslated, requestedModel, requestPath, opts.Headers)

	region := antigravityEnterpriseRegion(auth)
	requestURL := antigravityBAICURL(projectID, region, "streamGenerateContent")
	if opts.Alt == "" {
		requestURL += "?alt=sse"
	} else {
		requestURL += fmt.Sprintf("?$alt=%s", opts.Alt)
	}

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
	if errNewReq != nil {
		return nil, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	applyGeminiHeaders(httpReq, auth, opts.Headers)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs, opts.Headers)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       requestURL,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newAntigravityHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return nil, errDo
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("BAIC stream request error, status: %d, message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("antigravity BAIC executor: close response body error: %v", errClose)
		}
		return nil, statusErr{code: httpResp.StatusCode, msg: string(b)}
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	stopDeliverySupport := usage.SupportStreamDelivery(ctx)
	go func() {
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("antigravity BAIC executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, streamScannerBuffer)
		claudeInputTokens := helps.NewClaudeInputTokenState(from, to, responseFormat, originalPayload)
		var streamUsage helps.StreamUsageBuffer
		var streamErr error
		var pendingJSON []byte
		var terminalDelivered bool
		var param any
		defer func() {
			defer stopDeliverySupport()
			if helps.ApplyPatchTranslationError(param) != nil {
				streamErr = statusErr{code: http.StatusBadGateway, msg: helps.ApplyPatchUpstreamErrorMessage}
				stopDeliverySupport()
			}
			close(out)
			if streamErr == nil {
				streamErr = ctx.Err()
			}
			if deliveryErr, tracked := usage.WaitStreamDelivery(ctx); tracked {
				if streamErr == nil || errors.Is(streamErr, context.Canceled) {
					streamErr = deliveryErr
				}
			}
			if streamErr != nil {
				if terminalDelivered && errors.Is(streamErr, context.Canceled) {
					streamUsage.Publish(ctx, reporter)
					reporter.EnsurePublished(ctx)
					return
				}
				if !streamUsage.PublishFailure(ctx, reporter, streamErr) {
					reporter.PublishFailure(ctx, streamErr)
				}
				return
			}
			streamUsage.Publish(ctx, reporter)
			reporter.EnsurePublished(ctx)
		}()
		helps.InitializeApplyPatchStream(ctx, to, responseFormat, req.Model, helps.ApplyPatchOriginalRequest(req, opts), body, &param)
		for scanner.Scan() {
			line := scanner.Bytes()
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			if detail, ok := helps.ParseGeminiStreamUsage(line); ok {
				streamUsage.Observe(detail, true)
			} else if detail, ok := helps.ParseAntigravityStreamUsage(line); ok {
				streamUsage.Observe(detail, true)
			}

			line = helps.FilterSSEUsageMetadata(line)

			payload := helps.JSONPayload(line)
			if len(pendingJSON) > 0 {
				trimmedLine := bytes.TrimSpace(line)
				if bytes.HasPrefix(trimmedLine, []byte("data:")) {
					trimmedLine = bytes.TrimSpace(trimmedLine[len("data:"):])
				}
				if len(trimmedLine) > 0 {
					pendingJSON = append(pendingJSON, '\n')
					pendingJSON = append(pendingJSON, trimmedLine...)
				}
				if !gjson.ValidBytes(pendingJSON) {
					continue
				}
				payload = pendingJSON
				pendingJSON = nil
			} else if payload != nil && !gjson.ValidBytes(payload) {
				pendingJSON = append([]byte(nil), payload...)
				continue
			}
			if payload == nil {
				continue
			}
			if errorResult := gjson.GetBytes(payload, "error"); errorResult.Exists() {
				statusCode := int(errorResult.Get("code").Int())
				if statusCode < http.StatusBadRequest || statusCode > 599 {
					statusCode = http.StatusBadGateway
				}
				streamErr = newAntigravityStatusErr(statusCode, payload)
				stopDeliverySupport()
				helps.RecordAPIResponseError(ctx, e.cfg, streamErr)
				select {
				case out <- cliproxyexecutor.StreamChunk{Err: streamErr}:
				case <-ctx.Done():
				}
				return
			}
			reporter.ObserveResponseModel(payload)

			lines := helps.TranslateStreamWithClaudeInputTokens(ctx, to, responseFormat, req.Model, helps.ApplyPatchOriginalRequest(req, opts), body, bytes.Clone(payload), &param, claudeInputTokens)
			helps.RecordApplyPatchStreamFailure(ctx, param, reporter, statusErr{code: http.StatusBadGateway, msg: helps.ApplyPatchUpstreamErrorMessage})
			helps.CaptureTranslatedClaudeStreamError(responseFormat, lines, &streamErr, func(msg string) error {
				return statusErr{code: http.StatusBadGateway, msg: msg}
			})
			var isTerminalChunk bool
			if finishReason := gjson.GetBytes(payload, "candidates.0.finishReason"); finishReason.Exists() && strings.TrimSpace(finishReason.String()) != "" {
				isTerminalChunk = true
			} else if finishReason := gjson.GetBytes(payload, "response.candidates.0.finishReason"); finishReason.Exists() && strings.TrimSpace(finishReason.String()) != "" {
				isTerminalChunk = true
			}
			if !isTerminalChunk {
				for _, chunk := range lines {
					for _, cLine := range bytes.Split(chunk, []byte("\n")) {
						trimmed := bytes.TrimSpace(cLine)
						if bytes.Equal(trimmed, []byte("data: [DONE]")) || bytes.Equal(trimmed, []byte("[DONE]")) {
							isTerminalChunk = true
							break
						}
						cPayload := helps.JSONPayload(cLine)
						if len(cPayload) == 0 {
							continue
						}
						cType := gjson.GetBytes(cPayload, "type").String()
						if cType == "response.completed" || cType == "message_stop" {
							isTerminalChunk = true
							break
						}
						if finishReason := gjson.GetBytes(cPayload, "choices.0.finish_reason"); finishReason.Exists() && strings.TrimSpace(finishReason.String()) != "" {
							isTerminalChunk = true
							break
						}
					}
					if isTerminalChunk {
						break
					}
				}
			}
			for i := range lines {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: lines[i]}:
				case <-ctx.Done():
					return
				}
			}
			if isTerminalChunk {
				terminalDelivered = true
			}
			if helps.StopApplyPatchStream(ctx, param, reporter, out, statusErr{code: http.StatusBadGateway, msg: helps.ApplyPatchUpstreamErrorMessage}) {
				return
			}
			if helps.StopTranslatedClaudeStream(ctx, &streamErr, reporter, &streamUsage, out) {
				return
			}
		}
		if helps.EndApplyPatchStream(ctx, param, reporter, out, statusErr{code: http.StatusBadGateway, msg: helps.ApplyPatchUpstreamErrorMessage}) {
			return
		}
		if helps.StopTranslatedClaudeStream(ctx, &streamErr, reporter, &streamUsage, out) {
			return
		}
		if errScan := scanner.Err(); errScan != nil {
			streamErr = errScan
			if errors.Is(errScan, context.Canceled) && ctx.Err() != nil {
				if terminalDelivered {
					return
				}
				if deliveryErr, tracked := usage.WaitStreamDelivery(ctx); tracked && deliveryErr == nil {
					return
				}
			}
			stopDeliverySupport()
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			select {
			case out <- cliproxyexecutor.StreamChunk{Err: errScan}:
			case <-ctx.Done():
			}
		} else if ctx.Err() == nil {
			lines := helps.TranslateStreamWithClaudeInputTokens(ctx, to, responseFormat, req.Model, helps.ApplyPatchOriginalRequest(req, opts), body, []byte("[DONE]"), &param, claudeInputTokens)
			helps.RecordApplyPatchStreamFailure(ctx, param, reporter, statusErr{code: http.StatusBadGateway, msg: helps.ApplyPatchUpstreamErrorMessage})
			helps.CaptureTranslatedClaudeStreamError(responseFormat, lines, &streamErr, func(msg string) error {
				return statusErr{code: http.StatusBadGateway, msg: msg}
			})
			for i := range lines {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: lines[i]}:
				case <-ctx.Done():
					return
				}
			}
			terminalDelivered = true
			_ = helps.StopApplyPatchStream(ctx, param, reporter, out, statusErr{code: http.StatusBadGateway, msg: helps.ApplyPatchUpstreamErrorMessage})
			_ = helps.StopTranslatedClaudeStream(ctx, &streamErr, reporter, &streamUsage, out)
		}
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}
