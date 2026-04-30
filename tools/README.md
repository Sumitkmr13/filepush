# Diagnostic tools for SharePoint access

You have **two ways** to test, both using your existing browser login (no token copy needed for the easy path).

## Easy path: test directly in `/docs` (recommended)

Steps:

1. Set `DEBUG_AUTH=1` in your `.env`.
2. Restart the app and login normally via the UI.
3. In the **same browser**, open: `http://localhost:8000/docs`.
4. Use the endpoints under the **debug** tag — your session cookie auto-authenticates them.

### What to test

| Endpoint | Use it for |
|---|---|
| `GET /debug/token-info` | See logged-in user's UPN, scopes, audience, tenant. |
| `POST /debug/sharepoint/test-url` | Paste any SharePoint URL (share link, site URL, OneDrive URL). Returns step-by-step diagnostic. |
| `GET /debug/app-credentials` | Verify `.env` app credentials and (optionally) list drives/root for a site. |
| `GET /auth/debug/token` | Returns raw access token (for CLI tools). |

### Example: test a URL

In Swagger UI, expand `POST /debug/sharepoint/test-url`, click **Try it out**, paste:

```json
{
  "url": "https://solenis-my.sharepoint.com/:f:/r/personal/your_user_solenis_com/Documents/test-folder",
  "folder_path": "",
  "use_app_only": false
}
```

Click **Execute**. Response shows two checks:

- `user_check`: what your delegated token can do.
- `app_check`: what `.env` app credentials can do.

Each contains `steps[]` with `ok=true/false` per Graph call and the reason if failed.

### Interpreting results

- User `ok=true` and app `ok=true` → all good.
- User fails, app succeeds → URL valid; logged-in user lacks SharePoint access.
- Both fail with status `403` → token scope insufficient (e.g. missing `Files.Read`).
- Both fail with status `404` → URL itself is wrong / unreachable.

## CLI path (optional)

Useful when you want to script tests or use a different shell.

1. Visit `http://localhost:8000/auth/debug/token` and copy `access_token`.
2. PowerShell:

```powershell
$env:GEMRAG_USER_TOKEN = "<paste_token>"
python tools/test_sharepoint_url.py "https://tenant.sharepoint.com/:f:/s/Site/abc..."
```

Bash:

```bash
export GEMRAG_USER_TOKEN="<paste_token>"
python tools/test_sharepoint_url.py "https://tenant.sharepoint.com/:f:/s/Site/abc..."
```

Optional flags:

- `--token <jwt>` to pass token directly.
- `--app-only` to skip user-token testing.
- `--folder-path "Shared Documents/Foo"` to verify a specific folder under a site URL.

## App-credentials standalone test

```powershell
python tools/check_app_credentials.py
python tools/check_app_credentials.py --site-url "https://tenant.sharepoint.com/sites/Foo"
```

## IMPORTANT

- Always set `DEBUG_AUTH=0` (or remove it) before deploying to a shared environment.
- Tokens are short-lived (~1 hr); refresh by reloading `/auth/debug/token`.
