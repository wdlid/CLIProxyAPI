package cmd

import (
	"context"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	sdkAuth "github.com/router-for-me/CLIProxyAPI/v8/sdk/auth"
	log "github.com/sirupsen/logrus"
)

// DoAntigravityLogin triggers the OAuth flow for the antigravity provider and saves tokens.
func DoAntigravityLogin(cfg *config.Config, options *LoginOptions) {
	doAntigravityLoginWithClientMode(cfg, options, "")
}

// DoAntigravityEnterpriseLogin triggers the OAuth flow for Antigravity using the enterprise OAuth client.
func DoAntigravityEnterpriseLogin(cfg *config.Config, options *LoginOptions) {
	doAntigravityLoginWithClientMode(cfg, options, "enterprise")
}

// DoAntigravityLegacyLogin triggers the OAuth flow for Antigravity using the legacy (personal) OAuth client.
func DoAntigravityLegacyLogin(cfg *config.Config, options *LoginOptions) {
	doAntigravityLoginWithClientMode(cfg, options, "legacy")
}

func doAntigravityLoginWithClientMode(cfg *config.Config, options *LoginOptions, clientMode string) {
	if options == nil {
		options = &LoginOptions{}
	}

	promptFn := options.Prompt
	if promptFn == nil {
		promptFn = defaultProjectPrompt()
	}

	metadata := map[string]string{}
	if clientMode != "" {
		metadata["oauth_client"] = clientMode
	}

	manager := newAuthManager()
	authOpts := &sdkAuth.LoginOptions{
		NoBrowser:    options.NoBrowser,
		CallbackPort: options.CallbackPort,
		Metadata:     metadata,
		Prompt:       promptFn,
	}

	record, savedPath, err := manager.Login(context.Background(), "antigravity", cfg, authOpts)
	if err != nil {
		log.Errorf("Antigravity authentication failed: %v", err)
		return
	}

	if savedPath != "" {
		fmt.Printf("Authentication saved to %s\n", savedPath)
	}
	if record != nil && record.Label != "" {
		fmt.Printf("Authenticated as %s\n", record.Label)
	}
	fmt.Println("Antigravity authentication successful!")
}
