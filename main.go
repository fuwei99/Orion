package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

const (
	onyxBase       = "https://cloud.onyx.app"
	defaultPersona = 1
	maxRetries     = 3
	ver            = "0.5.0"
)

var (
	retryBackoff = [3]time.Duration{2 * time.Second, 5 * time.Second, 10 * time.Second}
	retryStatus  = map[int]bool{502: true, 503: true, 504: true, 429: true}
	httpClient   = &http.Client{
		Timeout: 5 * time.Minute,
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          200,
			MaxIdleConnsPerHost:   100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}
	keys       []string
	clientKeys map[string]struct{}
	keyIdx     uint64
)

// ── Key rotation ────────────────────────────────────────────────────────

func nextKey() string {
	n := len(keys)
	if n == 0 {
		return ""
	}
	return keys[atomic.AddUint64(&keyIdx, 1)%uint64(n)]
}

func extractToken(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if strings.HasPrefix(strings.ToLower(s), "bearer ") {
		return strings.TrimSpace(s[7:])
	}
	return s
}

func resolveAuth(headers ...string) string {
	if len(keys) > 0 {
		return nextKey()
	}
	for _, h := range headers {
		if h != "" {
			return extractToken(h)
		}
	}
	return ""
}

func checkClientAuth(r *http.Request) bool {
	if len(clientKeys) == 0 {
		return true
	}
	token := strings.TrimSpace(r.Header.Get("x-api-key"))
	if token == "" {
		token = extractToken(r.Header.Get("Authorization"))
	}
	if token == "" {
		return false
	}
	_, ok := clientKeys[token]
	return ok
}

// ── Model mapping ───────────────────────────────────────────────────────

type pm struct{ provider, version string }

var modelMap map[string]pm

func init() {
	modelMap = make(map[string]pm)
	data, err := os.ReadFile("models.json")
	if err == nil {
		var rawMap map[string]struct {
			Provider string `json:"provider"`
			Version  string `json:"version"`
		}
		if err := json.Unmarshal(data, &rawMap); err == nil {
			for k, v := range rawMap {
				modelMap[k] = pm{provider: v.Provider, version: v.Version}
			}
		} else {
			log.Printf("Warning: failed to parse models.json: %v", err)
		}
	} else {
		log.Printf("Warning: failed to read models.json (using fallback logic): %v", err)
	}
}

func buildLLMOverride(model string, temp float64) map[string]any {
	m, ok := modelMap[model]
	if !ok {
		parts := strings.SplitN(model, "__", 3)
		if len(parts) == 3 {
			m = pm{parts[0], parts[2]}
		} else {
			lower := strings.ToLower(model)
			if strings.Contains(lower, "gemini") {
				m = pm{"Google", model}
			} else if strings.Contains(lower, "chatgpt") || strings.Contains(lower, "gpt") || strings.Contains(lower, "codex") || strings.Contains(lower, "o1") || strings.Contains(lower, "o3") || strings.Contains(lower, "o4") {
				m = pm{"OpenAI", model}
			} else if strings.Contains(lower, "claude") {
				m = pm{"Anthropic", model}
			} else {
				m = pm{"Anthropic", model} // 默认
			}
		}
	}
	return map[string]any{"model_provider": m.provider, "model_version": m.version, "temperature": temp}
}

// ── Message conversion ──────────────────────────────────────────────────

func textContent(raw any) string {
	switch v := raw.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, item := range v {
			if m, ok := item.(map[string]any); ok && m["type"] == "text" {
				if s, ok := m["text"].(string); ok {
					parts = append(parts, s)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return fmt.Sprint(v)
	}
}

type chatMsg struct {
	Role       string `json:"role"`
	Content    any    `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

func messagesToOnyx(system string, msgs []chatMsg) string {
	var conv []string
	var lastUser string

	for _, m := range msgs {
		c := textContent(m.Content)
		switch m.Role {
		case "system":
			if system == "" {
				system = c
			} else {
				system += "\n" + c
			}
		case "user":
			lastUser = c
			conv = append(conv, "\bHuman: "+c)
		case "assistant":
			conv = append(conv, "\bAssistant: "+c)
		case "tool":
			tid := m.ToolCallID
			if tid == "" {
				tid = "unknown"
			}
			conv = append(conv, fmt.Sprintf("Tool result (%s): %s", tid, c))
		}
	}

	if system == "" && len(conv) == 1 && len(msgs) == 1 && msgs[0].Role == "user" {
		return lastUser
	}
	if system != "" && len(conv) == 1 && len(msgs) >= 1 {
		return "[System: " + system + "]\n\n" + lastUser
	}

	var b strings.Builder
	if system != "" {
		b.WriteString("[System: " + system + "]\n\n")
	}
	b.WriteString(strings.Join(conv, "\n\n"))
	return b.String()
}

// ── Helpers ─────────────────────────────────────────────────────────────

func genID(prefix string) string {
	b := make([]byte, 15)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%s%d", prefix, time.Now().UnixNano())
	}
	return prefix + hex.EncodeToString(b)[:29]
}

func toStrSlice(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, item := range arr {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func sleepOrDone(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// ── Core: send to Onyx with retry ──────────────────────────────────────

type onyxError struct {
	Status int
	Body   string
}

func (e *onyxError) Error() string {
	return fmt.Sprintf("Onyx HTTP %d: %s", e.Status, e.Body)
}

func doOnyxRequest(ctx context.Context, token, model string, msgs []chatMsg, system string, temp float64, persona int) (*http.Response, error) {
	bodyBytes, err := json.Marshal(map[string]any{
		"message":           messagesToOnyx(system, msgs),
		"chat_session_info": map[string]any{"persona_id": persona},
		"llm_override":      buildLLMOverride(model, temp),
		"stream":            true,
		"file_descriptors":  []any{},
		"deep_research":     false,
		"origin":            "api",
	})
	if err != nil {
		return nil, fmt.Errorf("marshal onyx request: %w", err)
	}

	var lastErr error
	for attempt := range maxRetries + 1 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		cur := token
		if attempt > 0 && len(keys) > 0 {
			cur = nextKey()
			log.Printf("Retry %d/%d, switched key", attempt, maxRetries)
		}

		req, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			onyxBase+"/api/chat/send-chat-message",
			bytes.NewReader(bodyBytes),
		)
		if err != nil {
			return nil, fmt.Errorf("create onyx request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+cur)

		resp, err := httpClient.Do(req)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			lastErr = err
			if attempt < maxRetries {
				wait := retryBackoff[min(attempt, len(retryBackoff)-1)]
				log.Printf("Attempt %d failed (%v), retry in %v", attempt+1, err, wait)
				if !sleepOrDone(ctx, wait) {
					return nil, ctx.Err()
				}
				continue
			}
			break
		}

		if resp.StatusCode != 200 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
			resp.Body.Close()
			log.Printf("Onyx HTTP %d for model=%s: %s", resp.StatusCode, model, string(body))
			if retryStatus[resp.StatusCode] && attempt < maxRetries {
				wait := retryBackoff[min(attempt, len(retryBackoff)-1)]
				log.Printf("Attempt %d failed (HTTP %d), retry in %v", attempt+1, resp.StatusCode, wait)
				if !sleepOrDone(ctx, wait) {
					return nil, ctx.Err()
				}
				continue
			}
			return nil, &onyxError{Status: resp.StatusCode, Body: string(body)}
		}
		return resp, nil
	}
	return nil, fmt.Errorf("all retries failed: %v", lastErr)
}

// ── Onyx NDJSON scanner ────────────────────────────────────────────────

type onyxScanner struct {
	sc *bufio.Scanner
}

type onyxEvent struct {
	Type string
	Obj  map[string]any
	Err  string
}

func newOnyxScanner(r io.Reader) *onyxScanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return &onyxScanner{sc: sc}
}

func (s *onyxScanner) Next() (onyxEvent, bool) {
	for s.sc.Scan() {
		line := strings.TrimSpace(s.sc.Text())
		if line == "" {
			continue
		}
		var raw map[string]any
		if json.Unmarshal([]byte(line), &raw) != nil {
			continue
		}
		if raw["user_message_id"] != nil {
			continue
		}
		if raw["error"] != nil && raw["obj"] == nil {
			return onyxEvent{Type: "error", Err: fmt.Sprint(raw["error"])}, true
		}
		obj, _ := raw["obj"].(map[string]any)
		if obj == nil {
			continue
		}
		t, _ := obj["type"].(string)
		return onyxEvent{Type: t, Obj: obj}, true
	}
	return onyxEvent{}, false
}

// ══════════════════════════════════════════════════════════════════════════
// OpenAI FORMAT
// ══════════════════════════════════════════════════════════════════════════

func sse(v any) []byte {
	b, _ := json.Marshal(v)
	return append(append([]byte("data: "), b...), '\n', '\n')
}

var sseDone = []byte("data: [DONE]\n\n")

func makeChunk(id string, created int64, model string, delta map[string]any, finish *string) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	}
}

func streamOpenAI(w io.Writer, flush func(), body io.Reader, model, rid string) {
	created := time.Now().Unix()
	sentRole := false
	toolActive := false
	stop := "stop"
	scan := newOnyxScanner(body)

	emit := func(delta map[string]any, fin *string) {
		w.Write(sse(makeChunk(rid, created, model, delta, fin)))
		flush()
	}
	ensureRole := func(d map[string]any) {
		if !sentRole {
			d["role"] = "assistant"
			sentRole = true
		}
	}

	for {
		ev, ok := scan.Next()
		if !ok {
			break
		}
		if ev.Err != "" {
			emit(map[string]any{"content": "\n\n[Error: " + ev.Err + "]"}, &stop)
			w.Write(sseDone)
			flush()
			return
		}
		obj := ev.Obj
		switch ev.Type {
		case "reasoning_start", "reasoning_done":
		case "reasoning_delta":
			if s, _ := obj["reasoning"].(string); s != "" {
				emit(map[string]any{"reasoning_content": s}, nil)
			}
		case "message_start":
			if !sentRole {
				emit(map[string]any{"role": "assistant", "content": ""}, nil)
				sentRole = true
			}
		case "message_delta":
			if c, _ := obj["content"].(string); c != "" {
				d := map[string]any{"content": c}
				ensureRole(d)
				emit(d, nil)
			}
		case "search_tool_start":
			toolActive = true
			label := "Internal Search"
			if v, _ := obj["is_internet_search"].(bool); v {
				label = "Web Search"
			}
			d := map[string]any{"content": "\n[" + label + "] "}
			ensureRole(d)
			emit(d, nil)
		case "search_tool_queries_delta":
			if qs := toStrSlice(obj["queries"]); len(qs) > 0 {
				var q []string
				for _, s := range qs {
					q = append(q, "\u201c"+s+"\u201d")
				}
				emit(map[string]any{"content": "Searching: " + strings.Join(q, ", ") + "\n"}, nil)
			}
		case "search_tool_documents_delta":
			if docs, ok := obj["documents"].([]any); ok && len(docs) > 0 {
				emit(map[string]any{"content": fmt.Sprintf("Found %d results.\n", len(docs))}, nil)
			}
		case "open_url_start":
			toolActive = true
			d := map[string]any{"content": "\n[Opening URL] "}
			ensureRole(d)
			emit(d, nil)
		case "open_url_urls":
			if urls := toStrSlice(obj["urls"]); len(urls) > 0 {
				emit(map[string]any{"content": strings.Join(urls, ", ") + "\n"}, nil)
			}
		case "open_url_documents":
			if docs, ok := obj["documents"].([]any); ok && len(docs) > 0 {
				emit(map[string]any{"content": fmt.Sprintf("Loaded %d pages.\n", len(docs))}, nil)
			}
		case "image_generation_start":
			toolActive = true
			d := map[string]any{"content": "\n[Generating Image...]\n"}
			ensureRole(d)
			emit(d, nil)
		case "image_generation_heartbeat":
		case "image_generation_final":
			if images, ok := obj["images"].([]any); ok {
				for _, img := range images {
					if m, ok := img.(map[string]any); ok {
						u, _ := m["url"].(string)
						p, _ := m["revised_prompt"].(string)
						if u != "" {
							emit(map[string]any{"content": fmt.Sprintf("![%s](%s)\n", p, u)}, nil)
						}
					}
				}
			}
		case "python_tool_start":
			toolActive = true
			code, _ := obj["code"].(string)
			text := "\n[Code Interpreter]\n"
			if code != "" {
				text += "```python\n" + code + "\n```\n"
			}
			d := map[string]any{"content": text}
			ensureRole(d)
			emit(d, nil)
		case "python_tool_delta":
			var parts []string
			if s, _ := obj["stdout"].(string); s != "" {
				parts = append(parts, "Output: "+s)
			}
			if s, _ := obj["stderr"].(string); s != "" {
				parts = append(parts, "Error: "+s)
			}
			if len(parts) > 0 {
				emit(map[string]any{"content": strings.Join(parts, "\n") + "\n"}, nil)
			}
		case "custom_tool_start":
			toolActive = true
			tn, _ := obj["tool_name"].(string)
			if tn == "" {
				tn = "custom_tool"
			}
			d := map[string]any{"content": "\n[Tool: " + tn + "]\n"}
			ensureRole(d)
			emit(d, nil)
		case "custom_tool_delta":
			if td := obj["data"]; td != nil {
				var text string
				if m, ok := td.(map[string]any); ok {
					b, _ := json.Marshal(m)
					text = string(b)
				} else {
					text = fmt.Sprint(td)
				}
				if text != "" {
					emit(map[string]any{"content": text + "\n"}, nil)
				}
			}
		case "file_reader_start":
			toolActive = true
			d := map[string]any{"content": "\n[Reading File...]\n"}
			ensureRole(d)
			emit(d, nil)
		case "file_reader_result":
			if fn, _ := obj["file_name"].(string); fn != "" {
				emit(map[string]any{"content": "Read: " + fn + "\n"}, nil)
			}
		case "deep_research_plan_start":
			d := map[string]any{"content": "\n[Research Plan]\n"}
			ensureRole(d)
			emit(d, nil)
		case "deep_research_plan_delta", "intermediate_report_start", "intermediate_report_delta":
			if c, _ := obj["content"].(string); c != "" {
				emit(map[string]any{"content": c}, nil)
			}
		case "research_agent_start":
			task, _ := obj["research_task"].(string)
			emit(map[string]any{"content": "\n[Researching: " + task + "]\n"}, nil)
		case "memory_tool_start", "memory_tool_delta", "memory_tool_no_access",
			"citation_info", "top_level_branching", "intermediate_report_cited_docs", "tool_call_debug":
		case "section_end":
			if toolActive {
				emit(map[string]any{"content": "\n"}, nil)
				toolActive = false
			}
		case "stop":
			emit(map[string]any{}, &stop)
			w.Write(sseDone)
			flush()
			return
		case "error":
			msg, _ := obj["error"].(string)
			if msg == "" {
				msg = "Unknown error"
			}
			emit(map[string]any{"content": "\n[Error: " + msg + "]"}, &stop)
			w.Write(sseDone)
			flush()
			return
		default:
			if ev.Type != "" {
				log.Printf("Unknown SSE event: %s", ev.Type)
			}
		}
	}
	w.Write(sseDone)
	flush()
}

func collectOpenAI(body io.Reader) string {
	var parts, toolCtx []string
	scan := newOnyxScanner(body)
	for {
		ev, ok := scan.Next()
		if !ok {
			break
		}
		if ev.Err != "" {
			parts = append(parts, "\n[Error: "+ev.Err+"]")
			break
		}
		obj := ev.Obj
		switch ev.Type {
		case "message_delta":
			if c, _ := obj["content"].(string); c != "" {
				parts = append(parts, c)
			}
		case "search_tool_start":
			label := "Internal Search"
			if v, _ := obj["is_internet_search"].(bool); v {
				label = "Web Search"
			}
			toolCtx = append(toolCtx, "["+label+"]")
		case "search_tool_queries_delta":
			if qs := toStrSlice(obj["queries"]); len(qs) > 0 {
				toolCtx = append(toolCtx, "Searching: "+strings.Join(qs, ", "))
			}
		case "search_tool_documents_delta":
			if docs, ok := obj["documents"].([]any); ok && len(docs) > 0 {
				toolCtx = append(toolCtx, fmt.Sprintf("Found %d results.", len(docs)))
			}
		case "open_url_start":
			toolCtx = append(toolCtx, "[Opening URL]")
		case "open_url_urls":
			if urls := toStrSlice(obj["urls"]); len(urls) > 0 {
				toolCtx = append(toolCtx, strings.Join(urls, ", "))
			}
		case "python_tool_start":
			code, _ := obj["code"].(string)
			toolCtx = append(toolCtx, "[Code Interpreter]\n```python\n"+code+"\n```")
		case "python_tool_delta":
			if s, _ := obj["stdout"].(string); s != "" {
				toolCtx = append(toolCtx, "Output: "+s)
			}
			if s, _ := obj["stderr"].(string); s != "" {
				toolCtx = append(toolCtx, "Error: "+s)
			}
		case "image_generation_final":
			if images, ok := obj["images"].([]any); ok {
				for _, img := range images {
					if m, ok := img.(map[string]any); ok {
						u, _ := m["url"].(string)
						p, _ := m["revised_prompt"].(string)
						if u != "" {
							toolCtx = append(toolCtx, fmt.Sprintf("![%s](%s)", p, u))
						}
					}
				}
			}
		case "custom_tool_start":
			tn, _ := obj["tool_name"].(string)
			toolCtx = append(toolCtx, "[Tool: "+tn+"]")
		case "custom_tool_delta":
			if td := obj["data"]; td != nil {
				toolCtx = append(toolCtx, fmt.Sprint(td))
			}
		case "error":
			msg, _ := obj["error"].(string)
			parts = append(parts, "\n[Error: "+msg+"]")
		case "stop":
			goto done
		}
	}
done:
	var b strings.Builder
	if len(toolCtx) > 0 {
		b.WriteString(strings.Join(toolCtx, "\n") + "\n\n")
	}
	for _, p := range parts {
		b.WriteString(p)
	}
	return b.String()
}

// ══════════════════════════════════════════════════════════════════════════
// Anthropic FORMAT
// ══════════════════════════════════════════════════════════════════════════

func anthropicSSE(event string, data any) []byte {
	b, _ := json.Marshal(data)
	return []byte("event: " + event + "\ndata: " + string(b) + "\n\n")
}

func streamAnthropic(w io.Writer, flush func(), body io.Reader, model, rid string) {
	scan := newOnyxScanner(body)
	blockIdx := 0
	inThinking := false
	inText := false

	// message_start
	w.Write(anthropicSSE("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": rid, "type": "message", "role": "assistant", "model": model,
			"content": []any{}, "stop_reason": nil,
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	}))
	flush()

	startBlock := func(btype string) {
		block := map[string]any{"type": btype}
		if btype == "thinking" {
			block["thinking"] = ""
		} else {
			block["text"] = ""
		}
		w.Write(anthropicSSE("content_block_start", map[string]any{
			"type": "content_block_start", "index": blockIdx, "content_block": block,
		}))
		flush()
	}
	stopBlock := func() {
		w.Write(anthropicSSE("content_block_stop", map[string]any{
			"type": "content_block_stop", "index": blockIdx,
		}))
		flush()
		blockIdx++
	}
	finishMsg := func(reason string) {
		if inThinking {
			stopBlock()
			inThinking = false
		}
		if inText {
			stopBlock()
			inText = false
		}
		w.Write(anthropicSSE("message_delta", map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": reason},
			"usage": map[string]any{"output_tokens": 0},
		}))
		w.Write(anthropicSSE("message_stop", map[string]any{"type": "message_stop"}))
		flush()
	}

	for {
		ev, ok := scan.Next()
		if !ok {
			break
		}
		if ev.Err != "" {
			if !inText {
				startBlock("text")
				inText = true
			}
			w.Write(anthropicSSE("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": blockIdx,
				"delta": map[string]any{"type": "text_delta", "text": "\n[Error: " + ev.Err + "]"},
			}))
			flush()
			finishMsg("end_turn")
			return
		}
		obj := ev.Obj
		switch ev.Type {
		case "reasoning_start":
			if !inThinking {
				startBlock("thinking")
				inThinking = true
			}
		case "reasoning_delta":
			if s, _ := obj["reasoning"].(string); s != "" {
				if !inThinking {
					startBlock("thinking")
					inThinking = true
				}
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "thinking_delta", "thinking": s},
				}))
				flush()
			}
		case "reasoning_done":
			if inThinking {
				stopBlock()
				inThinking = false
			}
		case "message_start":
			if !inText {
				if inThinking {
					stopBlock()
					inThinking = false
				}
				startBlock("text")
				inText = true
			}
		case "message_delta":
			if c, _ := obj["content"].(string); c != "" {
				if !inText {
					if inThinking {
						stopBlock()
						inThinking = false
					}
					startBlock("text")
					inText = true
				}
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": c},
				}))
				flush()
			}

		// ── Tool events → text deltas ──
		case "search_tool_start":
			if !inText {
				if inThinking {
					stopBlock()
					inThinking = false
				}
				startBlock("text")
				inText = true
			}
			label := "Internal Search"
			if v, _ := obj["is_internet_search"].(bool); v {
				label = "Web Search"
			}
			w.Write(anthropicSSE("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": blockIdx,
				"delta": map[string]any{"type": "text_delta", "text": "\n[" + label + "] "},
			}))
			flush()
		case "search_tool_queries_delta":
			if qs := toStrSlice(obj["queries"]); len(qs) > 0 {
				var q []string
				for _, s := range qs {
					q = append(q, "\u201c"+s+"\u201d")
				}
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": "Searching: " + strings.Join(q, ", ") + "\n"},
				}))
				flush()
			}
		case "search_tool_documents_delta":
			if docs, ok := obj["documents"].([]any); ok && len(docs) > 0 {
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": fmt.Sprintf("Found %d results.\n", len(docs))},
				}))
				flush()
			}
		case "open_url_start", "python_tool_start", "custom_tool_start",
			"image_generation_start", "file_reader_start", "deep_research_plan_start":
			if !inText {
				if inThinking {
					stopBlock()
					inThinking = false
				}
				startBlock("text")
				inText = true
			}
			var label string
			switch ev.Type {
			case "open_url_start":
				label = "[Opening URL] "
			case "python_tool_start":
				code, _ := obj["code"].(string)
				label = "[Code Interpreter]\n"
				if code != "" {
					label += "```python\n" + code + "\n```\n"
				}
			case "custom_tool_start":
				tn, _ := obj["tool_name"].(string)
				if tn == "" {
					tn = "custom_tool"
				}
				label = "[Tool: " + tn + "]\n"
			case "image_generation_start":
				label = "[Generating Image...]\n"
			case "file_reader_start":
				label = "[Reading File...]\n"
			case "deep_research_plan_start":
				label = "[Research Plan]\n"
			}
			w.Write(anthropicSSE("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": blockIdx,
				"delta": map[string]any{"type": "text_delta", "text": "\n" + label},
			}))
			flush()
		case "open_url_urls":
			if urls := toStrSlice(obj["urls"]); len(urls) > 0 {
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": strings.Join(urls, ", ") + "\n"},
				}))
				flush()
			}
		case "open_url_documents":
			if docs, ok := obj["documents"].([]any); ok && len(docs) > 0 {
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": fmt.Sprintf("Loaded %d pages.\n", len(docs))},
				}))
				flush()
			}
		case "python_tool_delta":
			var parts []string
			if s, _ := obj["stdout"].(string); s != "" {
				parts = append(parts, "Output: "+s)
			}
			if s, _ := obj["stderr"].(string); s != "" {
				parts = append(parts, "Error: "+s)
			}
			if len(parts) > 0 {
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": strings.Join(parts, "\n") + "\n"},
				}))
				flush()
			}
		case "custom_tool_delta":
			if td := obj["data"]; td != nil {
				var text string
				if m, ok := td.(map[string]any); ok {
					b, _ := json.Marshal(m)
					text = string(b)
				} else {
					text = fmt.Sprint(td)
				}
				if text != "" {
					w.Write(anthropicSSE("content_block_delta", map[string]any{
						"type": "content_block_delta", "index": blockIdx,
						"delta": map[string]any{"type": "text_delta", "text": text + "\n"},
					}))
					flush()
				}
			}
		case "image_generation_final":
			if images, ok := obj["images"].([]any); ok {
				for _, img := range images {
					if m, ok := img.(map[string]any); ok {
						u, _ := m["url"].(string)
						p, _ := m["revised_prompt"].(string)
						if u != "" {
							w.Write(anthropicSSE("content_block_delta", map[string]any{
								"type": "content_block_delta", "index": blockIdx,
								"delta": map[string]any{"type": "text_delta", "text": fmt.Sprintf("![%s](%s)\n", p, u)},
							}))
							flush()
						}
					}
				}
			}
		case "file_reader_result":
			if fn, _ := obj["file_name"].(string); fn != "" {
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": "Read: " + fn + "\n"},
				}))
				flush()
			}
		case "deep_research_plan_delta", "intermediate_report_start",
			"intermediate_report_delta":
			if c, _ := obj["content"].(string); c != "" {
				w.Write(anthropicSSE("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": blockIdx,
					"delta": map[string]any{"type": "text_delta", "text": c},
				}))
				flush()
			}
		case "research_agent_start":
			task, _ := obj["research_task"].(string)
			w.Write(anthropicSSE("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": blockIdx,
				"delta": map[string]any{"type": "text_delta", "text": "\n[Researching: " + task + "]\n"},
			}))
			flush()
		case "image_generation_heartbeat", "memory_tool_start", "memory_tool_delta",
			"memory_tool_no_access", "citation_info", "top_level_branching",
			"intermediate_report_cited_docs", "tool_call_debug":
		case "section_end":
			// skip
		case "stop":
			finishMsg("end_turn")
			return
		case "error":
			msg, _ := obj["error"].(string)
			if msg == "" {
				msg = "Unknown error"
			}
			if !inText {
				startBlock("text")
				inText = true
			}
			w.Write(anthropicSSE("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": blockIdx,
				"delta": map[string]any{"type": "text_delta", "text": "\n[Error: " + msg + "]"},
			}))
			flush()
			finishMsg("end_turn")
			return
		default:
			if ev.Type != "" {
				log.Printf("Unknown SSE event: %s", ev.Type)
			}
		}
	}
	finishMsg("end_turn")
}

func collectAnthropic(body io.Reader) (text, thinking string) {
	var textParts, thinkParts, toolCtx []string
	scan := newOnyxScanner(body)
	for {
		ev, ok := scan.Next()
		if !ok {
			break
		}
		if ev.Err != "" {
			textParts = append(textParts, "\n[Error: "+ev.Err+"]")
			break
		}
		obj := ev.Obj
		switch ev.Type {
		case "reasoning_delta":
			if s, _ := obj["reasoning"].(string); s != "" {
				thinkParts = append(thinkParts, s)
			}
		case "message_delta":
			if c, _ := obj["content"].(string); c != "" {
				textParts = append(textParts, c)
			}
		case "search_tool_start":
			label := "Internal Search"
			if v, _ := obj["is_internet_search"].(bool); v {
				label = "Web Search"
			}
			toolCtx = append(toolCtx, "["+label+"]")
		case "search_tool_queries_delta":
			if qs := toStrSlice(obj["queries"]); len(qs) > 0 {
				toolCtx = append(toolCtx, "Searching: "+strings.Join(qs, ", "))
			}
		case "search_tool_documents_delta":
			if docs, ok := obj["documents"].([]any); ok && len(docs) > 0 {
				toolCtx = append(toolCtx, fmt.Sprintf("Found %d results.", len(docs)))
			}
		case "open_url_start":
			toolCtx = append(toolCtx, "[Opening URL]")
		case "open_url_urls":
			if urls := toStrSlice(obj["urls"]); len(urls) > 0 {
				toolCtx = append(toolCtx, strings.Join(urls, ", "))
			}
		case "python_tool_start":
			code, _ := obj["code"].(string)
			toolCtx = append(toolCtx, "[Code Interpreter]\n```python\n"+code+"\n```")
		case "python_tool_delta":
			if s, _ := obj["stdout"].(string); s != "" {
				toolCtx = append(toolCtx, "Output: "+s)
			}
			if s, _ := obj["stderr"].(string); s != "" {
				toolCtx = append(toolCtx, "Error: "+s)
			}
		case "image_generation_final":
			if images, ok := obj["images"].([]any); ok {
				for _, img := range images {
					if m, ok := img.(map[string]any); ok {
						u, _ := m["url"].(string)
						p, _ := m["revised_prompt"].(string)
						if u != "" {
							toolCtx = append(toolCtx, fmt.Sprintf("![%s](%s)", p, u))
						}
					}
				}
			}
		case "custom_tool_start":
			tn, _ := obj["tool_name"].(string)
			toolCtx = append(toolCtx, "[Tool: "+tn+"]")
		case "custom_tool_delta":
			if td := obj["data"]; td != nil {
				toolCtx = append(toolCtx, fmt.Sprint(td))
			}
		case "error":
			msg, _ := obj["error"].(string)
			textParts = append(textParts, "\n[Error: "+msg+"]")
		case "stop":
			goto done
		}
	}
done:
	var tb strings.Builder
	if len(toolCtx) > 0 {
		tb.WriteString(strings.Join(toolCtx, "\n") + "\n\n")
	}
	for _, p := range textParts {
		tb.WriteString(p)
	}
	return tb.String(), strings.Join(thinkParts, "")
}

// ══════════════════════════════════════════════════════════════════════════
// HTTP HANDLERS
// ══════════════════════════════════════════════════════════════════════════

func handleModels(w http.ResponseWriter, r *http.Request) {
	models := []string{
		"claude-opus-4-6", "claude-opus-4-5",
		"claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4",
		"claude-3-5-sonnet", "claude-3-5-haiku", "claude-haiku-4-5",
		"gpt-4o", "gpt-4o-mini", "o1", "o3-mini",
		"gemini-2.0-flash", "gemini-2.5-pro",
	}
	data := make([]any, len(models))
	for i, m := range models {
		data[i] = map[string]any{"id": m, "object": "model", "created": 1700000000, "owned_by": "onyx"}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"status": "ok", "version": ver, "keys": len(keys)})
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, 200, map[string]any{
		"name":      "onyx2api",
		"version":   ver,
		"endpoints": []string{"/v1/chat/completions", "/v1/messages", "/v1/models", "/health"},
	})
}

// ── OpenAI: POST /v1/chat/completions ──

type openaiReq struct {
	Model       string    `json:"model"`
	Messages    []chatMsg `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature *float64  `json:"temperature,omitempty"`
	PersonaID   *int      `json:"persona_id,omitempty"`
}

func handleOpenAI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]any{"error": "method not allowed"})
		return
	}
	if !checkClientAuth(r) {
		writeJSON(w, 401, map[string]any{"error": "invalid api key"})
		return
	}
	var req openaiReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"error": "invalid JSON"})
		return
	}
	token := resolveAuth(r.Header.Get("Authorization"))
	if token == "" {
		writeJSON(w, 401, map[string]any{"error": "No auth. Set ONYX_KEYS env var."})
		return
	}
	if len(req.Messages) == 0 {
		writeJSON(w, 400, map[string]any{"error": "messages is required"})
		return
	}

	model := req.Model
	if model == "" {
		model = "claude-opus-4-6"
	}
	temp := 0.5
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	persona := defaultPersona
	if req.PersonaID != nil {
		persona = *req.PersonaID
	}

	rid := genID("chatcmpl-")
	resp, err := doOnyxRequest(r.Context(), token, model, req.Messages, "", temp, persona)
	if err != nil {
		if oe, ok := err.(*onyxError); ok {
			writeJSON(w, oe.Status, map[string]any{"error": oe.Error(), "detail": oe.Body})
		} else {
			writeJSON(w, 502, map[string]any{"error": err.Error()})
		}
		return
	}
	defer resp.Body.Close()

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeJSON(w, 500, map[string]any{"error": "streaming unsupported"})
			return
		}
		streamOpenAI(w, flusher.Flush, resp.Body, model, rid)
	} else {
		content := collectOpenAI(resp.Body)
		writeJSON(w, 200, map[string]any{
			"id": rid, "object": "chat.completion", "created": time.Now().Unix(), "model": model,
			"choices": []any{map[string]any{
				"index": 0, "message": map[string]any{"role": "assistant", "content": content},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
		})
	}
}

// ── Anthropic: POST /v1/messages ──

type anthropicReq struct {
	Model       string    `json:"model"`
	Messages    []chatMsg `json:"messages"`
	System      any       `json:"system,omitempty"` // string or []content_block
	Stream      bool      `json:"stream"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	PersonaID   *int      `json:"persona_id,omitempty"`
}

func handleAnthropic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]any{"error": map[string]any{"type": "invalid_request_error", "message": "method not allowed"}})
		return
	}
	if !checkClientAuth(r) {
		writeJSON(w, 401, map[string]any{"type": "error", "error": map[string]any{"type": "authentication_error", "message": "invalid api key"}})
		return
	}
	var req anthropicReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"type": "error", "error": map[string]any{"type": "invalid_request_error", "message": "invalid JSON"}})
		return
	}
	token := resolveAuth(r.Header.Get("x-api-key"), r.Header.Get("Authorization"))
	if token == "" {
		writeJSON(w, 401, map[string]any{"type": "error", "error": map[string]any{"type": "authentication_error", "message": "No auth. Set ONYX_KEYS env var."}})
		return
	}
	if len(req.Messages) == 0 {
		writeJSON(w, 400, map[string]any{"type": "error", "error": map[string]any{"type": "invalid_request_error", "message": "messages is required"}})
		return
	}

	model := req.Model
	if model == "" {
		model = "claude-opus-4-6"
	}
	temp := 0.5
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	persona := defaultPersona
	if req.PersonaID != nil {
		persona = *req.PersonaID
	}

	system := textContent(req.System)

	rid := genID("msg_")
	resp, err := doOnyxRequest(r.Context(), token, model, req.Messages, system, temp, persona)
	if err != nil {
		if oe, ok := err.(*onyxError); ok {
			writeJSON(w, oe.Status, map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": oe.Error()}})
		} else {
			writeJSON(w, 502, map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": err.Error()}})
		}
		return
	}
	defer resp.Body.Close()

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeJSON(w, 500, map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": "streaming unsupported"}})
			return
		}
		streamAnthropic(w, flusher.Flush, resp.Body, model, rid)
	} else {
		text, thinking := collectAnthropic(resp.Body)
		var content []any
		if thinking != "" {
			content = append(content, map[string]any{"type": "thinking", "thinking": thinking})
		}
		content = append(content, map[string]any{"type": "text", "text": text})
		writeJSON(w, 200, map[string]any{
			"id": rid, "type": "message", "role": "assistant", "model": model,
			"content":     content,
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 0, "output_tokens": 0},
		})
	}
}

// ── Main ────────────────────────────────────────────────────────────────

func main() {
	if env := os.Getenv("ONYX_KEYS"); env != "" {
		for _, k := range strings.Split(env, ",") {
			if k = strings.TrimSpace(k); k != "" {
				keys = append(keys, k)
			}
		}
	}
	if env := firstNonEmpty(os.Getenv("CLIENT_API_KEYS"), os.Getenv("CLIENT_API_KEY")); env != "" {
		clientKeys = map[string]struct{}{}
		for _, k := range strings.Split(env, ",") {
			if k = strings.TrimSpace(k); k != "" {
				clientKeys[k] = struct{}{}
			}
		}
	}
	if len(clientKeys) > 0 && len(keys) == 0 {
		log.Fatal("CLIENT_API_KEYS/CLIENT_API_KEY requires ONYX_KEYS to be set")
	}
	listenAddr := resolveListenAddr()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleRoot)
	mux.HandleFunc("/v1/chat/completions", handleOpenAI)
	mux.HandleFunc("/v1/messages", handleAnthropic)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/health", handleHealth)

	srv := &http.Server{
		Addr:              listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	log.Printf("onyx2api v%s | onyx_keys=%d | client_keys=%d | listen=%s", ver, len(keys), len(clientKeys), listenAddr)
	log.Fatal(srv.ListenAndServe())
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func resolveListenAddr() string {
	if addr := strings.TrimSpace(os.Getenv("LISTEN_ADDR")); addr != "" {
		return addr
	}
	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = "9898"
	}
	if strings.Contains(port, ":") {
		return port
	}
	return ":" + port
}
