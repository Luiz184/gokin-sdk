# gokin-sdk

[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Go framework for building AI agents with tool use, multi-agent orchestration, planning, and reflection. Supports **Gemini**, **OpenAI**, **Anthropic (Claude)**, and **Ollama** as LLM providers.

## Features

- **Multi-provider** — Gemini, OpenAI, Anthropic, and Ollama with unified interface
- **Tool use** — 29 built-in tools (bash, file I/O, git, grep, web search, and more)
- **Multi-agent** — Runner/Coordinator for parallel and sequential agent execution
- **Planning** — Beam search, MCTS, and A* strategies for complex task decomposition
- **Reflection** — Self-correcting agents via reflector middleware
- **Smart routing** — Adaptive task routing with strategy learning
- **MCP support** — Model Context Protocol for external tool servers
- **Sessions** — Persistent conversation state with auto-save
- **Security** — Command sandboxing, path validation, permission system

## Installation

```bash
go get github.com/ginkida/gokin-sdk
```

Requires Go 1.23 or later.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "os"

    sdk "github.com/ginkida/gokin-sdk"
    "github.com/ginkida/gokin-sdk/provider/gemini"
    "github.com/ginkida/gokin-sdk/tools"
)

func main() {
    ctx := context.Background()

    client, err := gemini.New(ctx, os.Getenv("GEMINI_API_KEY"), "gemini-2.0-flash")
    if err != nil {
        panic(err)
    }
    defer client.Close()

    workDir, _ := os.Getwd()
    registry := sdk.NewRegistry()
    registry.MustRegister(tools.NewBash(workDir))
    registry.MustRegister(tools.NewRead())
    registry.MustRegister(tools.NewGlob(workDir))
    registry.MustRegister(tools.NewGrep(workDir))

    agent, err := sdk.NewAgent("assistant", client, registry,
        sdk.WithSystemPrompt("You are a helpful coding assistant."),
        sdk.WithMaxTurns(20),
        sdk.WithOnText(func(text string) {
            fmt.Print(text)
        }),
    )
    if err != nil {
        panic(err)
    }

    result, err := agent.Run(ctx, "Find all Go files and count lines of code")
    if err != nil {
        panic(err)
    }
    fmt.Printf("\nDone in %d turns\n", result.Turns)
}
```

## Provider Examples

### Gemini

```go
client, err := gemini.New(ctx, os.Getenv("GEMINI_API_KEY"), "gemini-2.0-flash")
```

### OpenAI

```go
client, err := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
```

Works with any OpenAI-compatible API (vLLM, LM Studio, Together AI, Groq):

```go
client, err := openai.New(apiKey, "model-name", openai.WithBaseURL("http://localhost:8000"))
```

### Anthropic (Claude)

```go
client, err := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-5-20250929")
```

### Ollama (local)

```go
client, err := ollama.New("http://localhost:11434", "llama3")
```

## Multi-Agent Execution

```go
runner := sdk.NewRunner(client, registry,
    sdk.WithRunnerMaxTurns(10),
)

coordinator := sdk.NewCoordinator(runner, 3) // max 3 parallel agents

results, err := coordinator.RunParallel(ctx, []sdk.AgentTask{
    {Prompt: "Count Go files", Type: sdk.AgentTypeBash, Description: "Count files"},
    {Prompt: "Find TODO comments", Type: sdk.AgentTypeExplore, Description: "Find TODOs"},
})
```

## Built-in Tools

| Category | Tools |
|----------|-------|
| **File I/O** | `read`, `write`, `edit`, `glob`, `grep`, `delete`, `move`, `copy`, `mkdir`, `list_dir`, `tree`, `diff` |
| **Execution** | `bash`, `run_tests`, `batch` |
| **Git** | `git`, `git_branch`, `git_pr` |
| **Search** | `web_fetch`, `web_search`, `semantic_search` |
| **Agent** | `ask_user`, `ask_agent`, `task`, `task_output`, `task_stop`, `coordinate` |
| **Planning** | `plan_mode`, `shared_memory` |

## Project Structure

```
gokin-sdk/
├── provider/          # LLM providers (gemini, openai, anthropic, ollama)
├── tools/             # Built-in tool implementations
├── plan/              # Planning engine (beam, MCTS, A*)
├── context/           # Context management and summarization
├── config/            # Configuration and loading
├── security/          # Sandboxing and validation
├── permission/        # Permission system
├── mcp/               # Model Context Protocol client
├── memory/            # Error store, project learning
├── tasks/             # Background task management
├── audit/             # Audit logging
├── session.go         # Session persistence
├── middleware.go       # Reflector and middleware
├── pool.go            # Client connection pool
├── examples/          # Usage examples
│   ├── simple/        # Basic agent
│   ├── openai/        # OpenAI / compatible APIs
│   ├── anthropic/     # Anthropic provider
│   ├── multi_agent/   # Parallel agents
│   ├── planner/       # Plan-driven execution
│   ├── smart_router/  # Adaptive routing
│   ├── mcp/           # MCP integration
│   └── session/       # Session persistence
└── ...
```

## Examples

See the [`examples/`](examples/) directory for runnable demos:

- **[simple](examples/simple/)** — Basic agent with Gemini
- **[openai](examples/openai/)** — OpenAI and compatible APIs (vLLM, Groq, etc.)
- **[anthropic](examples/anthropic/)** — Using Claude as the provider
- **[multi_agent](examples/multi_agent/)** — Parallel task execution
- **[planner](examples/planner/)** — Plan-driven agent with beam search
- **[smart_router](examples/smart_router/)** — Adaptive task routing
- **[mcp](examples/mcp/)** — External tool servers via MCP
- **[session](examples/session/)** — Persistent conversations

## License

MIT
