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

// ErrEnterpriseAccountNeedsEnterpriseClient is returned by login flows when an account signed in with
// the legacy OAuth client has no discoverable project. Enterprise accounts need the enterprise OAuth
// client, so the next login in "auto" mode will use the enterprise client automatically.
var ErrEnterpriseAccountNeedsEnterpriseClient = errors.New("antigravity: no project found for this account with the legacy OAuth client; enterprise accounts need the enterprise OAuth client, so sign in again (the next login will use the enterprise client automatically)")

// nextLoginClientOverride is set in "auto" mode after a login fails project discovery,
// so the next login attempt is issued with the alternate OAuth client ("legacy" or "enterprise").
var nextLoginClientOverride atomic.Value // stores string: "" | OAuthClientLegacy | OAuthClientEnterprise

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

// NormalizeOAuthClientMode normalizes an explicit mode string ("enterprise", "legacy", "personal", "auto").
func NormalizeOAuthClientMode(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case OAuthClientEnterprise, "baic":
		return OAuthClientEnterprise
	case OAuthClientLegacy, "personal":
		return OAuthClientLegacy
	case OAuthClientAuto:
		return OAuthClientAuto
	default:
		return ""
	}
}

// OAuthClientMode returns the normalized configured mode.
func OAuthClientMode(cfg *config.Config) string {
	if cfg == nil {
		return OAuthClientAuto
	}
	if mode := NormalizeOAuthClientMode(cfg.Antigravity.OAuthClient); mode != "" {
		return mode
	}
	return OAuthClientAuto
}

// SelectLoginOAuthClient picks the OAuth client for a new login attempt.
// In "auto" mode, it defaults to the legacy (personal) client first so personal accounts succeed
// on the first login, and consumes any pending fallback override if the previous login failed discovery.
func SelectLoginOAuthClient(cfg *config.Config) OAuthClient {
	return SelectLoginOAuthClientWithOverride(cfg, "")
}

// SelectLoginOAuthClientWithOverride picks the OAuth client for a new login attempt, allowing
// an explicit per-request mode override ("legacy"/"personal" or "enterprise").
func SelectLoginOAuthClientWithOverride(cfg *config.Config, modeOverride string) OAuthClient {
	mode := NormalizeOAuthClientMode(modeOverride)
	if mode == "" {
		mode = OAuthClientMode(cfg)
	}
	switch mode {
	case OAuthClientLegacy:
		return LegacyOAuthClient()
	case OAuthClientEnterprise:
		return EnterpriseOAuthClient()
	default:
		if prev, _ := nextLoginClientOverride.Swap("").(string); prev != "" {
			if prev == OAuthClientEnterprise {
				return EnterpriseOAuthClient()
			}
			if prev == OAuthClientLegacy {
				return LegacyOAuthClient()
			}
		}
		return LegacyOAuthClient()
	}
}

// HandleEmptyDiscovery is called when a login finishes with no project ID. In "auto" mode it arms
// the alternate OAuth client for the next login attempt and returns a descriptive error.
// It returns nil when no fallback applies (caller keeps its existing behavior).
func HandleEmptyDiscovery(cfg *config.Config, client OAuthClient) error {
	return HandleEmptyDiscoveryWithOverride(cfg, client, "")
}

// HandleEmptyDiscoveryWithOverride behaves like HandleEmptyDiscovery while respecting an explicit per-request mode override.
func HandleEmptyDiscoveryWithOverride(cfg *config.Config, client OAuthClient, modeOverride string) error {
	mode := NormalizeOAuthClientMode(modeOverride)
	if mode == "" {
		mode = OAuthClientMode(cfg)
	}
	if mode != OAuthClientAuto {
		return nil
	}
	if client.IsLegacy() {
		nextLoginClientOverride.Store(OAuthClientEnterprise)
		return ErrEnterpriseAccountNeedsEnterpriseClient
	}
	nextLoginClientOverride.Store(OAuthClientLegacy)
	return ErrPersonalAccountNeedsLegacyClient
}

// ApplyOAuthClientMetadata records the OAuth client on the credential so refreshes use the
// matching client.
func ApplyOAuthClientMetadata(metadata map[string]any, client OAuthClient) {
	if metadata == nil || strings.TrimSpace(client.ID) == "" {
		return
	}
	if client.IsLegacy() {
		metadata["client_id"] = client.ID
		metadata["client_secret"] = client.Secret
		return
	}
	delete(metadata, "client_id")
	delete(metadata, "client_secret")
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
