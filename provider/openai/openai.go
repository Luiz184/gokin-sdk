// Package openai provides an OpenAI-compatible client implementation for the SDK.
// It works with api.openai.com and any OpenAI-compatible API (vLLM, LM Studio,
// Together AI, Groq, etc.) via a configurable base URL.
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	sdk "github.com/ginkida/gokin-sdk"

	"google.golang.org/genai"
)

// Option configures an OpenAIClient.
type Option func(*OpenAIClient)

// WithBaseURL sets a custom base URL (for compatible APIs like vLLM, LM Studio, Groq).
func WithBaseURL(url string) Option {
	return func(c *OpenAIClient) {
		c.baseURL = url
	}
}

// WithMaxTokens sets the maximum output tokens.
func WithMaxTokens(n int32) Option {
	return func(c *OpenAIClient) {
		c.maxTokens = n
	}
}

// WithTemperature sets the temperature for generation.
func WithTemperature(t float32) Option {
	return func(c *OpenAIClient) {
		c.temperature = t
	}
}

// WithMaxRetries sets the maximum number of retries.
func WithMaxRetries(n int) Option {
	return func(c *OpenAIClient) {
		c.maxRetries = n
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *OpenAIClient) {
		c.httpClient = client
	}
}

// OpenAIClient implements sdk.Client for the OpenAI Chat Completions API.
type OpenAIClient struct {
	apiKey            string
	baseURL           string
	model             string
	maxTokens         int32
	temperature       float32
	maxRetries        int
	retryDelay        time.Duration
	httpClient        *http.Client
	tools             []*genai.Tool
	systemInstruction string
	mu                sync.RWMutex
}

// New creates a new OpenAI-compatible client.
func New(apiKey string, model string, opts ...Option) (*OpenAIClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	if model == "" {
		return nil, fmt.Errorf("model name is required")
	}

	c := &OpenAIClient{
		apiKey:     apiKey,
		baseURL:    "https://api.openai.com",
		model:      model,
		maxTokens:  4096,
		maxRetries: 3,
		retryDelay: 1 * time.Second,
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// SendMessage sends a message and returns a streaming response.
func (c *OpenAIClient) SendMessage(ctx context.Context, message string) (*sdk.StreamResponse, error) {
	return c.SendMessageWithHistory(ctx, nil, message)
}

// SendMessageWithHistory sends a message with conversation history.
func (c *OpenAIClient) SendMessageWithHistory(ctx context.Context, history []*genai.Content, message string) (*sdk.StreamResponse, error) {
	c.mu.RLock()
	sysInstruction := c.systemInstruction
	c.mu.RUnlock()

	messages := convertHistoryToMessages(history, sysInstruction, message)

	requestBody := map[string]any{
		"model":      c.model,
		"max_tokens": c.maxTokens,
		"messages":   messages,
		"stream":     true,
		"stream_options": map[string]any{
			"include_usage": true,
		},
	}

	if c.temperature > 0 {
		requestBody["temperature"] = c.temperature
	}

	c.mu.RLock()
	if len(c.tools) > 0 {
		requestBody["tools"] = convertToolsToOpenAI(c.tools)
	}
	c.mu.RUnlock()

	return c.streamRequest(ctx, requestBody)
}

// SendFunctionResponse sends function call results back to the model.
func (c *OpenAIClient) SendFunctionResponse(ctx context.Context, history []*genai.Content, results []*genai.FunctionResponse) (*sdk.StreamResponse, error) {
	c.mu.RLock()
	sysInstruction := c.systemInstruction
	c.mu.RUnlock()

	messages := convertHistoryWithResults(history, results, sysInstruction)

	requestBody := map[string]any{
		"model":      c.model,
		"max_tokens": c.maxTokens,
		"messages":   messages,
		"stream":     true,
		"stream_options": map[string]any{
			"include_usage": true,
		},
	}

	if c.temperature > 0 {
		requestBody["temperature"] = c.temperature
	}

	c.mu.RLock()
	if len(c.tools) > 0 {
		requestBody["tools"] = convertToolsToOpenAI(c.tools)
	}
	c.mu.RUnlock()

	return c.streamRequest(ctx, requestBody)
}

// SetTools sets the tools available for function calling.
func (c *OpenAIClient) SetTools(tools []*genai.Tool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tools = tools
}

// SetSystemInstruction sets the system-level instruction.
func (c *OpenAIClient) SetSystemInstruction(instruction string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.systemInstruction = instruction
}

// GetModel returns the model name.
func (c *OpenAIClient) GetModel() string {
	return c.model
}

// Clone returns an independent copy that shares the underlying http.Client
// but has its own tools and systemInstruction state.
func (c *OpenAIClient) Clone() sdk.Client {
	c.mu.RLock()
	defer c.mu.RUnlock()
	clone := &OpenAIClient{
		apiKey:      c.apiKey,
		baseURL:     c.baseURL,
		model:       c.model,
		maxTokens:   c.maxTokens,
		temperature: c.temperature,
		maxRetries:  c.maxRetries,
		retryDelay:  c.retryDelay,
		httpClient:  c.httpClient,
	}
	if c.tools != nil {
		clone.tools = make([]*genai.Tool, len(c.tools))
		copy(clone.tools, c.tools)
	}
	clone.systemInstruction = c.systemInstruction
	return clone
}

// Close closes the client.
func (c *OpenAIClient) Close() error {
	return nil
}

// streamRequest performs a streaming request with retry logic.
func (c *OpenAIClient) streamRequest(ctx context.Context, requestBody map[string]any) (*sdk.StreamResponse, error) {
	var lastErr error
	maxDelay := 30 * time.Second

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			delay := backoff(c.retryDelay, attempt-1, maxDelay)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		response, err := c.doStreamRequest(ctx, requestBody)
		if err == nil {
			return response, nil
		}

		lastErr = err
		if !isRetryable(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("max retries (%d) exceeded: %w", c.maxRetries, lastErr)
}

// doStreamRequest performs a single streaming request.
func (c *OpenAIClient) doStreamRequest(ctx context.Context, requestBody map[string]any) (*sdk.StreamResponse, error) {
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := strings.TrimSuffix(c.baseURL, "/") + "/v1/chat/completions"

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	chunks := make(chan sdk.ResponseChunk, 10)
	done := make(chan struct{})

	go func() {
		defer close(chunks)
		defer close(done)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		acc := &toolCallAccumulator{}

		for scanner.Scan() {
			line := scanner.Text()

			var data string
			var found bool
			if data, found = strings.CutPrefix(line, "data: "); !found {
				if data, found = strings.CutPrefix(line, "data:"); !found {
					continue
				}
			}

			if data == "[DONE]" {
				finalCalls := acc.finalize()
				if len(finalCalls) > 0 {
					chunks <- sdk.ResponseChunk{FunctionCalls: finalCalls, Done: true}
				} else {
					chunks <- sdk.ResponseChunk{Done: true}
				}
				return
			}

			var event streamChunk
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			chunk := processStreamChunk(event, acc)
			if chunk.Text != "" || chunk.Done || len(chunk.FunctionCalls) > 0 || chunk.InputTokens > 0 {
				select {
				case chunks <- chunk:
				case <-ctx.Done():
					return
				}
			}

			if chunk.Done {
				return
			}
		}
	}()

	return &sdk.StreamResponse{
		Chunks: chunks,
		Done:   done,
	}, nil
}

// streamChunk represents a parsed SSE chunk from the OpenAI API.
type streamChunk struct {
	Choices []struct {
		Index        int    `json:"index"`
		Delta        delta  `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usage `json:"usage,omitempty"`
}

type delta struct {
	Content   string          `json:"content"`
	ToolCalls []deltaToolCall `json:"tool_calls"`
}

type deltaToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// toolCallAccumulator tracks parallel tool calls during streaming.
// OpenAI supports multiple concurrent tool calls indexed by position.
type toolCallAccumulator struct {
	calls []accumulatedToolCall
}

type accumulatedToolCall struct {
	id       string
	name     string
	argsJSON strings.Builder
}

// accumulate processes a delta tool call, creating or appending to the
// accumulated call at the given index.
func (a *toolCallAccumulator) accumulate(dtc deltaToolCall) {
	idx := dtc.Index
	for len(a.calls) <= idx {
		a.calls = append(a.calls, accumulatedToolCall{})
	}
	if dtc.ID != "" {
		a.calls[idx].id = dtc.ID
	}
	if dtc.Function.Name != "" {
		a.calls[idx].name = dtc.Function.Name
	}
	if dtc.Function.Arguments != "" {
		a.calls[idx].argsJSON.WriteString(dtc.Function.Arguments)
	}
}

// finalize converts accumulated tool calls into genai.FunctionCall values.
func (a *toolCallAccumulator) finalize() []*genai.FunctionCall {
	if len(a.calls) == 0 {
		return nil
	}
	result := make([]*genai.FunctionCall, 0, len(a.calls))
	for i := range a.calls {
		call := &a.calls[i]
		if call.name == "" {
			continue
		}
		var args map[string]any
		raw := call.argsJSON.String()
		if raw != "" {
			if err := json.Unmarshal([]byte(raw), &args); err != nil {
				args = make(map[string]any)
			}
		} else {
			args = make(map[string]any)
		}
		result = append(result, &genai.FunctionCall{
			ID:   call.id,
			Name: call.name,
			Args: args,
		})
	}
	return result
}

// processStreamChunk converts an OpenAI stream chunk to a ResponseChunk.
func processStreamChunk(event streamChunk, acc *toolCallAccumulator) sdk.ResponseChunk {
	chunk := sdk.ResponseChunk{}

	// Extract usage from the final chunk (available via stream_options.include_usage).
	if event.Usage != nil {
		chunk.InputTokens = event.Usage.PromptTokens
		chunk.OutputTokens = event.Usage.CompletionTokens
	}

	if len(event.Choices) == 0 {
		return chunk
	}

	choice := event.Choices[0]

	// Accumulate text content.
	if choice.Delta.Content != "" {
		chunk.Text = choice.Delta.Content
	}

	// Accumulate tool calls by index.
	for _, tc := range choice.Delta.ToolCalls {
		acc.accumulate(tc)
	}

	// Handle finish reasons.
	switch choice.FinishReason {
	case "stop":
		chunk.Done = true
		chunk.FinishReason = genai.FinishReasonStop
	case "tool_calls":
		chunk.Done = true
		chunk.FinishReason = genai.FinishReasonStop
		chunk.FunctionCalls = acc.finalize()
	case "length":
		chunk.Done = true
		chunk.FinishReason = genai.FinishReasonMaxTokens
	case "content_filter":
		chunk.Done = true
		chunk.FinishReason = genai.FinishReasonSafety
	}

	return chunk
}

// convertHistoryToMessages converts genai history to OpenAI messages format.
func convertHistoryToMessages(history []*genai.Content, systemInstruction string, newMessage string) []map[string]any {
	messages := make([]map[string]any, 0)

	if systemInstruction != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": systemInstruction,
		})
	}

	for _, content := range history {
		msgs := convertContent(content)
		messages = append(messages, msgs...)
	}

	if newMessage == "" {
		newMessage = "Continue."
	}
	messages = append(messages, map[string]any{
		"role":    "user",
		"content": newMessage,
	})

	return messages
}

// convertHistoryWithResults converts history with function results to messages.
func convertHistoryWithResults(history []*genai.Content, results []*genai.FunctionResponse, systemInstruction string) []map[string]any {
	messages := make([]map[string]any, 0)

	if systemInstruction != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": systemInstruction,
		})
	}

	for _, content := range history {
		msgs := convertContent(content)
		messages = append(messages, msgs...)
	}

	// Each function response becomes a separate "tool" message.
	for _, result := range results {
		toolCallID := result.ID
		if toolCallID == "" {
			toolCallID = result.Name
		}

		contentStr := extractResponseContent(result.Response)

		messages = append(messages, map[string]any{
			"role":         "tool",
			"tool_call_id": toolCallID,
			"content":      contentStr,
		})
	}

	return messages
}

// convertContent converts a single genai.Content to one or more OpenAI messages.
func convertContent(content *genai.Content) []map[string]any {
	if content.Role == "model" {
		return []map[string]any{buildAssistantMessage(content.Parts)}
	}

	// For user role, separate text parts and function response parts.
	// Text parts become a user message; function responses become individual tool messages.
	var textParts []string
	var toolMessages []map[string]any

	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
		if part.FunctionResponse != nil {
			toolCallID := part.FunctionResponse.ID
			if toolCallID == "" {
				toolCallID = part.FunctionResponse.Name
			}
			contentStr := extractResponseContent(part.FunctionResponse.Response)
			toolMessages = append(toolMessages, map[string]any{
				"role":         "tool",
				"tool_call_id": toolCallID,
				"content":      contentStr,
			})
		}
	}

	var result []map[string]any
	if len(textParts) > 0 {
		result = append(result, map[string]any{
			"role":    "user",
			"content": strings.Join(textParts, "\n"),
		})
	}
	result = append(result, toolMessages...)

	if len(result) == 0 {
		result = append(result, map[string]any{
			"role":    "user",
			"content": "Continue.",
		})
	}

	return result
}

// buildAssistantMessage builds an OpenAI assistant message from parts.
func buildAssistantMessage(parts []*genai.Part) map[string]any {
	msg := map[string]any{
		"role": "assistant",
	}

	var textContent string
	var toolCalls []map[string]any

	for _, part := range parts {
		if part.Text != "" {
			textContent += part.Text
		}
		if part.FunctionCall != nil {
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			toolID := part.FunctionCall.ID
			if toolID == "" {
				toolID = part.FunctionCall.Name
			}

			toolCalls = append(toolCalls, map[string]any{
				"id":   toolID,
				"type": "function",
				"function": map[string]any{
					"name":      part.FunctionCall.Name,
					"arguments": string(argsJSON),
				},
			})
		}
	}

	if textContent != "" {
		msg["content"] = textContent
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}

	// OpenAI requires at least content or tool_calls on assistant messages.
	if textContent == "" && len(toolCalls) == 0 {
		msg["content"] = ""
	}

	return msg
}

// extractResponseContent extracts a string from a function response map.
func extractResponseContent(response map[string]any) string {
	if response == nil {
		return "Operation completed"
	}
	if c, ok := response["content"].(string); ok {
		return c
	}
	if errStr, ok := response["error"].(string); ok && errStr != "" {
		return "Error: " + errStr
	}
	return "Operation completed"
}

// convertToolsToOpenAI converts genai tools to OpenAI function-calling format.
func convertToolsToOpenAI(tools []*genai.Tool) []map[string]any {
	result := make([]map[string]any, 0)

	for _, tool := range tools {
		for _, decl := range tool.FunctionDeclarations {
			result = append(result, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        decl.Name,
					"description": decl.Description,
					"parameters":  convertSchemaToJSON(decl.Parameters),
				},
			})
		}
	}

	return result
}

// convertSchemaToJSON converts a genai.Schema to JSON Schema.
func convertSchemaToJSON(schema *genai.Schema) map[string]any {
	if schema == nil {
		return nil
	}

	result := make(map[string]any)

	if schema.Type != "" {
		result["type"] = strings.ToLower(string(schema.Type))
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}
	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for name, propSchema := range schema.Properties {
			props[name] = convertSchemaToJSON(propSchema)
		}
		result["properties"] = props
	}
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}
	if schema.Items != nil {
		result["items"] = convertSchemaToJSON(schema.Items)
	}

	return result
}

// isRetryable returns true if the error should trigger a retry.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	patterns := []string{"429", "500", "502", "503", "504", "timeout", "connection refused", "connection reset"}
	for _, p := range patterns {
		if strings.Contains(errStr, p) {
			return true
		}
	}
	return false
}

// backoff calculates exponential backoff delay.
func backoff(base time.Duration, attempt int, max time.Duration) time.Duration {
	delay := base
	for range attempt {
		delay *= 2
	}
	if delay > max {
		delay = max
	}
	return delay
}
