// Package main demonstrates using the OpenAI provider.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	sdk "github.com/ginkida/gokin-sdk"
	"github.com/ginkida/gokin-sdk/provider/openai"
	"github.com/ginkida/gokin-sdk/tools"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY environment variable is required")
	}

	model := "gpt-4o"
	if m := os.Getenv("OPENAI_MODEL"); m != "" {
		model = m
	}

	// Create an OpenAI client.
	// For compatible APIs, use: openai.WithBaseURL("http://localhost:8000")
	client, err := openai.New(apiKey, model)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer client.Close()

	// Create a registry with tools
	workDir, _ := os.Getwd()
	registry := sdk.NewRegistry()
	registry.MustRegister(tools.NewBash(workDir))
	registry.MustRegister(tools.NewRead())
	registry.MustRegister(tools.NewGlob(workDir))
	registry.MustRegister(tools.NewGrep(workDir))
	registry.MustRegister(tools.NewEdit())
	registry.MustRegister(tools.NewWrite())

	// Create an agent
	agent, err := sdk.NewAgent("openai-assistant", client, registry,
		sdk.WithSystemPrompt("You are a helpful coding assistant powered by OpenAI."),
		sdk.WithMaxTurns(15),
		sdk.WithOnText(func(text string) {
			fmt.Print(text)
		}),
		sdk.WithOnToolCall(func(name string, args map[string]any) {
			fmt.Printf("\n[Tool: %s]\n", name)
		}),
	)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	message := "What Go files are in the current directory? Briefly describe their purpose."
	if len(os.Args) > 1 {
		message = os.Args[1]
	}

	fmt.Printf("User: %s\n\nAssistant: ", message)
	result, err := agent.Run(ctx, message)
	if err != nil {
		return fmt.Errorf("run failed: %w", err)
	}

	fmt.Printf("\n\nCompleted in %d turns (%v)\n", result.Turns, result.Duration.Round(time.Millisecond))
	return nil
}
