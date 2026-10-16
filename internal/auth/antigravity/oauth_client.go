package antigravity

import (
	"errors"
	"strings"
	"sync/atomic"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
)

// OAuth client selection modes for new Antigravity logins (config: antigravity.oauth-client).
const (
	OAuthClientAuto       = "auto"
	OAuthClientEnterprise = "enterprise"
	OAuthClientLegacy     = "legacy"
)

// ErrPersonalAccountNeedsLegacyClient is returned by login flows when an account signed in with
// the enterprise OAuth client has neither a Gemini Enterprise license nor a discoverable project.
// Google refuses the free tier for that client ("GOOGLE_TOS_NOT_SUPPORTED_BY_CLIENT"), so the
// account must sign in again using the legacy client.
var ErrPersonalAccountNeedsLegacyClient = errors.New("antigravity: no Gemini Enterprise license or project found for this account with the enterprise OAuth client; personal accounts need the legacy client, so sign in again (the next login will use the legacy client automatically)")

// legacyNextLogin is set in "auto" mode after a personal account fails discovery on the enterprise
// client, so the next login attempt is issued with the legacy client.
var legacyNextLogin atomic.Bool

// OAuthClient is an OAuth client ID/secret pair.
type OAuthClient struct {
	ID     string
	Secret string
}

// IsLegacy reports whether the client is the legacy (personal / free-tier) client.
func (c OAuthClient) IsLegacy() bool { return c.ID == LegacyClientID }

// EnterpriseOAuthClient returns the current Antigravity OAuth client.
func EnterpriseOAuthClient() OAuthClient { return OAuthClient{ID: ClientID, Secret: ClientSecret} }

// LegacyOAuthClient returns the legacy Antigravity OAuth client.
func LegacyOAuthClient() OAuthClient {
	return OAuthClient{ID: LegacyClientID, Secret: LegacyClientSecret}
}

// OAuthClientMode returns the normalized configured mode.
func OAuthClientMode(cfg *config.Config) string {
	if cfg == nil {
		return OAuthClientAuto
	}
	switch strings.ToLower(strings.TrimSpace(cfg.Antigravity.OAuthClient)) {
	case OAuthClientEnterprise:
		return OAuthClientEnterprise
	case OAuthClientLegacy:
		return OAuthClientLegacy
	default:
		return OAuthClientAuto
	}
}

// SelectLoginOAuthClient picks the OAuth client for a new login attempt. In "auto" mode the
// pending legacy-next flag is consumed, so only one login attempt is redirected to legacy.
func SelectLoginOAuthClient(cfg *config.Config) OAuthClient {
	switch OAuthClientMode(cfg) {
	case OAuthClientLegacy:
		return LegacyOAuthClient()
	case OAuthClientEnterprise:
		return EnterpriseOAuthClient()
	default:
		if legacyNextLogin.Swap(false) {
			return LegacyOAuthClient()
		}
		return EnterpriseOAuthClient()
	}
}

// HandleEmptyDiscovery is called when a login finishes with no project ID. In "auto" mode with
// the enterprise client it arms the legacy-next flag and returns ErrPersonalAccountNeedsLegacyClient.
// It returns nil when no fallback applies (caller keeps its existing behavior).
func HandleEmptyDiscovery(cfg *config.Config, client OAuthClient) error {
	if client.IsLegacy() || OAuthClientMode(cfg) != OAuthClientAuto {
		return nil
	}
	legacyNextLogin.Store(true)
	return ErrPersonalAccountNeedsLegacyClient
}

// ApplyOAuthClientMetadata records the OAuth client on the credential so refreshes use the
// matching client. Nothing is stored for the default enterprise client.
func ApplyOAuthClientMetadata(metadata map[string]any, client OAuthClient) {
	if metadata == nil || !client.IsLegacy() {
		return
	}
	metadata["client_id"] = client.ID
	metadata["client_secret"] = client.Secret
}

// WithOAuthClient returns the auth service configured to use the given client for
// BuildAuthURL / ExchangeCodeForTokens.
func (o *AntigravityAuth) WithOAuthClient(client OAuthClient) *AntigravityAuth {
	if o == nil || strings.TrimSpace(client.ID) == "" {
		return o
	}
	o.client = client
	return o
}

func (o *AntigravityAuth) oauthClient() OAuthClient {
	if o == nil || strings.TrimSpace(o.client.ID) == "" {
		return EnterpriseOAuthClient()
	}
	return o.client
}
