"""Simple OAuth for the remote MCP server: one password, no database.

Why OAuth: Claude on the web connects to a custom connector with no authentication or with OAuth.
A fixed token in a header is a beta that not every account has. So the server is its own
authorization server, as the MCP spec allows, and its "login" is a page that asks for one password.

Why no database: on Vercel, each request may run on a different instance, and memory is not
shared. The SDK's examples keep clients, codes and tokens in a dict; here, the first /register
would land on one instance and the /token on another, which would not know the client. So every
piece of state the flow needs (the client, the authorization code, the tokens) is a signed message
that carries its own data: any instance checks the signature with AUTH_SECRET and reads it.

What that costs, said plainly:
  - A token cannot be revoked one by one. Changing AUTH_SECRET revokes everything at once.
  - An authorization code could be used twice within its 5 minutes. PKCE limits the damage:
    only whoever holds the code_verifier, which never left the client, can exchange it.
  - Only Claude's callback addresses may register (ALLOWED_REDIRECTS), so a code never goes
    to another site.

Env:  AUTH_SECRET      random key that signs everything (python -c "import secrets; print(secrets.token_urlsafe(32))")
      ACCESS_PASSWORD  the password typed on the login page (16 characters or more)
      PUBLIC_URL       the server's address (on Vercel, taken from VERCEL_PROJECT_PRODUCTION_URL)
      ALLOWED_REDIRECTS  comma-separated; default: Claude's two callbacks
"""
import asyncio
import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import time

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

SCOPE = "read"
ACCESS_TTL = 3600  # 1 hour; Claude renews it with the refresh token
REFRESH_TTL = 30 * 24 * 3600
CODE_TTL = 300
LOGIN_TTL = 600


def public_url():
    if url := os.environ.get("PUBLIC_URL"):
        return url.rstrip("/")
    if host := os.environ.get("VERCEL_PROJECT_PRODUCTION_URL"):
        return f"https://{host}"
    return "http://localhost:8000"


BASE = public_url()
RESOURCE = f"{BASE}/mcp"
ALLOWED_REDIRECTS = [u.strip() for u in os.environ.get(
    "ALLOWED_REDIRECTS",
    "https://claude.ai/api/mcp/auth_callback,https://claude.com/api/mcp/auth_callback",
).split(",") if u.strip()]


def _key():
    key = os.environ.get("AUTH_SECRET", "")
    if len(key) < 32:
        raise RuntimeError("AUTH_SECRET: use at least 32 random characters")
    return key.encode()


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(kind, data, ttl=None):
    """A message only this server could have written. kind is signed too, so a refresh token
    cannot pass as an access token, nor a client_id as an authorization code."""
    payload = {**data, "kind": kind}
    if ttl:
        payload["exp"] = int(time.time()) + ttl
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    mac = _b64(hmac.new(_key(), f"{kind}.{body}".encode(), hashlib.sha256).digest())
    return f"{body}.{mac}"


def verify(kind, token):
    """The message's data, or None if the signature, the kind or the expiry do not match."""
    try:
        body, mac = token.split(".")
        expected = _b64(hmac.new(_key(), f"{kind}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(mac, expected):
            return None
        data = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("kind") != kind or data.get("exp", float("inf")) < time.time():
        return None
    return data


def _client_secret(client_id):
    # Derived, not stored: any instance can recompute it. It never travels inside the client_id,
    # which is public (it shows up in URLs).
    return _b64(hmac.new(_key(), f"secret.{client_id}".encode(), hashlib.sha256).digest())


class StatelessProvider:
    """The SDK's OAuthAuthorizationServerProvider, with signed messages instead of storage."""

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        uris = [str(u) for u in client_info.redirect_uris or []]
        if not uris or any(u not in ALLOWED_REDIRECTS for u in uris):
            raise RegistrationError("invalid_redirect_uri", f"only {', '.join(ALLOWED_REDIRECTS)}")
        # The SDK made up a random client_id and expects us to store it. We replace it, in the
        # object the SDK is about to send back, with one that carries the client's data.
        client_info.client_id = sign("client", {
            "uris": uris,
            "auth": client_info.token_endpoint_auth_method,
            "name": (client_info.client_name or "")[:60],
            "nonce": secrets.token_urlsafe(8),
        })
        if client_info.client_secret is not None:
            client_info.client_secret = _client_secret(client_info.client_id)

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        data = verify("client", client_id)
        if data is None:
            return None
        return OAuthClientInformationFull(
            client_id=client_id,
            client_secret=None if data["auth"] == "none" else _client_secret(client_id),
            client_secret_expires_at=0,
            token_endpoint_auth_method=data["auth"],
            redirect_uris=data["uris"],
            client_name=data["name"] or None,
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=SCOPE,
        )

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        request = sign("login", {
            "client": client.client_id,
            "name": client.client_name or "",
            "redirect": str(params.redirect_uri),
            "explicit": params.redirect_uri_provided_explicitly,
            "challenge": params.code_challenge,
            "state": params.state,
            "resource": params.resource or RESOURCE,
        }, LOGIN_TTL)
        return f"{BASE}/login?{_query(request=request)}"

    async def load_authorization_code(self, client, authorization_code):
        data = verify("code", authorization_code)
        if data is None or data["client"] != client.client_id:
            return None
        return AuthorizationCode(
            code=authorization_code, scopes=[SCOPE], expires_at=data["exp"], client_id=data["client"],
            code_challenge=data["challenge"], redirect_uri=data["redirect"],
            redirect_uri_provided_explicitly=data["explicit"], resource=data["resource"],
        )

    async def exchange_authorization_code(self, client, authorization_code) -> OAuthToken:
        return _tokens(client.client_id, authorization_code.resource)

    async def load_refresh_token(self, client, refresh_token):
        data = verify("refresh", refresh_token)
        if data is None or data["client"] != client.client_id:
            return None
        return RefreshToken(token=refresh_token, client_id=data["client"], scopes=[SCOPE],
                            expires_at=data["exp"], resource=data["resource"])

    async def exchange_refresh_token(self, client, refresh_token, scopes) -> OAuthToken:
        if scopes and set(scopes) != {SCOPE}:
            raise TokenError("invalid_scope", f"the only scope is {SCOPE}")
        return _tokens(client.client_id, refresh_token.resource)

    async def load_access_token(self, token: str) -> AccessToken | None:
        data = verify("access", token)
        if data is None:
            return None
        return AccessToken(token=token, client_id=data["client"], scopes=[SCOPE],
                           expires_at=data["exp"], resource=data["resource"])

    async def revoke_token(self, token) -> None:
        pass  # no list to remove it from; revocation is disabled in the settings below


def _tokens(client_id, resource):
    data = {"client": client_id, "resource": resource or RESOURCE}
    return OAuthToken(
        access_token=sign("access", data, ACCESS_TTL), token_type="Bearer", expires_in=ACCESS_TTL,
        refresh_token=sign("refresh", data, REFRESH_TTL), scope=SCOPE,
    )


def _query(**params):
    return construct_redirect_uri("", **params).lstrip("?")


SETTINGS = AuthSettings(
    issuer_url=AnyHttpUrl(BASE),
    resource_server_url=AnyHttpUrl(RESOURCE),
    validate_token_resource=True,  # a token issued for another server is refused here
    required_scopes=[SCOPE],
    client_registration_options=ClientRegistrationOptions(
        enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE],
    ),
)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Monetary Policy Report MCP</title>
<style>body{{font:16px system-ui,sans-serif;max-width:26rem;margin:4rem auto;padding:0 1rem;color:#15191a}}
input,button{{font:inherit;padding:.6rem;width:100%;box-sizing:border-box;margin-top:.6rem}}
button{{background:#d9348c;color:#fff;border:0;cursor:pointer}}.err{{color:#b00020}}</style></head>
<body><h1>Monetary Policy Report MCP</h1>
<p><strong>{name}</strong> wants read-only access to this server's tools.</p>{error}
<form method="post"><input type="hidden" name="request" value="{request}">
<label>Password<input type="password" name="password" autofocus required></label>
<button>Allow</button></form></body></html>"""


async def login(request: Request) -> Response:
    """GET shows the password form; POST checks it and sends the browser back to Claude
    with an authorization code."""
    form = await request.form() if request.method == "POST" else {}
    token = str(form.get("request") or request.query_params.get("request", ""))
    data = verify("login", token)
    if data is None:
        return HTMLResponse("This login link expired. Start again from Claude.", status_code=400)

    def page(error=""):
        return HTMLResponse(PAGE.format(name=html.escape(data["name"] or "An MCP client"),
                                        request=html.escape(token), error=error))

    if request.method == "GET":
        return page()
    password = os.environ.get("ACCESS_PASSWORD", "")
    if len(password) < 16:
        return HTMLResponse("ACCESS_PASSWORD on the server is missing or shorter than 16 characters.",
                            status_code=500)
    if not hmac.compare_digest(str(form.get("password", "")).encode(), password.encode()):
        await asyncio.sleep(1)  # slows down guessing; a long password is what really stops it
        return page('<p class="err">Wrong password.</p>')
    code = sign("code", {k: data[k] for k in ("client", "redirect", "explicit", "challenge", "resource")}
                | {"nonce": secrets.token_urlsafe(16)}, CODE_TTL)
    return RedirectResponse(construct_redirect_uri(data["redirect"], code=code, state=data["state"]),
                            status_code=302)
