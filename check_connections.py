"""
Connectivity and config checks: run with real credentials to verify GCP, SharePoint,
Gemini/Vertex, and app config. Tells you exactly which check failed and why.

Usage:
  python check_connections.py                    # run all checks
  python check_connections.py --gcp              # GCP + Gemini only
  python check_connections.py --sharepoint        # SharePoint only
  python check_connections.py --config           # config summary only (no network)
  python check_connections.py --sharepoint-browse                  # list drive root (discover folder paths)
  python check_connections.py --sharepoint-browse "pmo"            # list contents of folder pmo
  python check_connections.py --sharepoint-browse-interactive      # connect, list drives, navigate to desired path
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load env before importing app modules
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent
load_dotenv(_repo_root / ".env")
_gac = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
if _gac and not Path(_gac).is_absolute():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((_repo_root / Path(_gac)).resolve())

# Result: (success: bool, detail: str)
CheckResult = tuple[bool, str]


def _truncate(s: str, max_len: int = 80) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def check_gcp_credentials_file() -> CheckResult:
    """Verify GOOGLE_APPLICATION_CREDENTIALS points to a valid JSON with project_id."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        return False, "GOOGLE_APPLICATION_CREDENTIALS not set"
    if not os.path.isfile(path):
        return False, f"File not found: {path}"
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = data.get("project_id")
        if not pid:
            return False, "JSON has no 'project_id'"
        return True, f"project_id={pid}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except OSError as e:
        return False, f"Read error: {e}"


def check_gcp_project() -> CheckResult:
    """Report current GCP project (from config, which reads env or credentials file)."""
    try:
        from config import GCP_PROJECT, GCP_LOCATION
    except Exception as e:
        return False, f"config import failed: {e}"
    if not GCP_PROJECT:
        return False, "GCP_PROJECT is empty (set env or use a credentials JSON with project_id)"
    return True, f"project={GCP_PROJECT}, location={GCP_LOCATION}"


def check_gemini_model_config() -> CheckResult:
    """Report configured Vertex/Gemini model names."""
    try:
        from config import VERTEX_MODEL, VERTEX_EMBEDDING_MODEL
    except Exception as e:
        return False, f"config import failed: {e}"
    return True, f"LLM={VERTEX_MODEL}, embedding={VERTEX_EMBEDDING_MODEL}"


def check_gemini_reachable() -> CheckResult:
    """Actually call Vertex (Gemini) with a tiny request to verify the model is reachable."""
    try:
        from config import GCP_PROJECT, GCP_LOCATION, VERTEX_MODEL
    except Exception as e:
        return False, f"config failed: {e}"
    if not GCP_PROJECT:
        return False, "GCP_PROJECT empty: cannot call Vertex"
    try:
        from llama_index.llms.google_genai import GoogleGenAI
        llm = GoogleGenAI(
            model=VERTEX_MODEL,
            vertexai_config={"project": GCP_PROJECT, "location": GCP_LOCATION},
            max_tokens=10,
        )
        r = llm.complete("Say OK in one word.")
        out = (r.text or "").strip()
        return True, f"model={VERTEX_MODEL} responded: {_truncate(out)}"
    except Exception as e:
        err_str = str(e).upper()
        hint = ""
        if "403" in err_str and ("VPC_SERVICE_CONTROLS" in err_str or "SECURITY_POLICY_VIOLATED" in err_str or "ORGANIZATION'S POLICY" in err_str):
            hint = " - VPC Service Controls: org policy is blocking access from this network. Ask GCP/security admin to allow the VDI for Vertex AI. See CLIENT_REQUIREMENTS.md §4 item 6."
        return False, f"Vertex/Gemini call failed: {type(e).__name__}: {e}{hint}"


def check_sharepoint_config() -> CheckResult:
    """Report whether SharePoint env vars are set (no network call)."""
    try:
        from sharepoint_utils import is_sharepoint_configured
        from config import SHAREPOINT_SITE_URL, SHAREPOINT_DRIVE_PATH, SHAREPOINT_DRIVE_ID
    except Exception as e:
        return False, f"import failed: {e}"
    if not is_sharepoint_configured():
        return False, "Missing: set SHAREPOINT_SITE_URL, CLIENT_ID, CLIENT_SECRET, TENANT_ID"
    drive_path = (SHAREPOINT_DRIVE_PATH or "(root)").strip()
    extra = f"; drive_id={_truncate(SHAREPOINT_DRIVE_ID, 24)}" if SHAREPOINT_DRIVE_ID else ""
    return True, f"site={_truncate(SHAREPOINT_SITE_URL)}; drive_path={drive_path}{extra}"


def check_sharepoint_token() -> CheckResult:
    """Get MSAL token for Microsoft Graph (validates client id, secret, tenant)."""
    try:
        from sharepoint_utils import get_access_token
        token = get_access_token()
        if not token:
            return False, "get_access_token() returned empty"
        return True, f"token length={len(token)}"
    except ValueError as e:
        return False, f"config error in get_access_token: {e}"
    except RuntimeError as e:
        return False, f"MSAL token failed (check client id/secret/tenant): {e}"
    except Exception as e:
        return False, f"get_access_token failed: {type(e).__name__}: {e}"


def check_sharepoint_site_and_drive() -> CheckResult:
    """Resolve site ID and drive ID. When SHAREPOINT_DRIVE_ID is set, skip the default-drive API call to avoid hanging."""
    try:
        from sharepoint_utils import get_access_token, _get_site_id, _resolve_drive_id
        from config import SHAREPOINT_DRIVE_ID
    except Exception as e:
        return False, f"import failed: {e}"
    try:
        token = get_access_token()
        site_id = _get_site_id(token)
        drive_id = _resolve_drive_id(token, site_id)
        note = " (from env)" if SHAREPOINT_DRIVE_ID else ""
        return True, f"site_id={_truncate(site_id, 24)}, drive_id={_truncate(drive_id, 24)}{note}"
    except Exception as e:
        return False, f"site/drive resolve failed: {type(e).__name__}: {e}"


def check_sharepoint_path_quick() -> CheckResult:
    """Verify path is reachable (first page only). Fast; use for default --sharepoint so it does not timeout."""
    import threading
    result_holder: list = []

    def run() -> None:
        try:
            from sharepoint_utils import verify_sharepoint_path_reachable
            ok, msg = verify_sharepoint_path_reachable()
            result_holder.append(("ok", ok, msg))
        except Exception as e:
            result_holder.append(("err", e))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=90)
    if not t.is_alive() and result_holder:
        row = result_holder[0]
        if row[0] == "err":
            return False, f"path check failed: {type(row[1]).__name__}: {row[1]}"
        _, ok, msg = row
        return ok, msg
    return False, "Path check timed out after 90s. Path may still work in main.py (no timeout there)."


# Full PDF list can be slow for thousands of files / many nested folders
SHAREPOINT_FULL_LIST_TIMEOUT = 1800  # 30 minutes

def check_sharepoint_list_pdfs_full() -> CheckResult:
    """List all PDFs under path via Search API (or recursive fallback), same as main.py. Use --sharepoint-full for full list."""
    import threading
    result_holder: list = []
    print("         Listing all PDFs (recursive crawl if Search fails)... may take up to 30 min for large libraries, please wait.", flush=True)

    def run() -> None:
        try:
            from sharepoint_utils import list_pdf_items_with_drive
            from state_manager import filter_unprocessed
            items, drive_id = list_pdf_items_with_drive()
            to_process_ids = filter_unprocessed([it["id"] for it in items])
            result_holder.append(("ok", items, to_process_ids, drive_id))
        except Exception as e:
            result_holder.append(("err", e))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=SHAREPOINT_FULL_LIST_TIMEOUT)
    if not t.is_alive() and result_holder:
        row = result_holder[0]
        if row[0] == "err":
            return False, f"list PDFs failed: {type(row[1]).__name__}: {row[1]}"
        _, items, to_process_ids, _ = row
        n = len(items)
        new_count = len(to_process_ids)
        done_count = n - new_count
        sample = ""
        if items:
            names = [it.get("name") or it.get("id", "")[:40] for it in items[:5]]
            sample = "; sample: " + ", ".join(names) + (" ..." if n > 5 else "")
        msg = f"{n} PDF(s) found (incl. nested), {new_count} new to process, {done_count} already in state{sample}"
        return True, msg
    # Timeout: listing takes longer than check window; extraction in main.py has no timeout
    return True, f"List timed out after {SHAREPOINT_FULL_LIST_TIMEOUT // 60} min (library large). Extraction (main.py) has no timeout — run it to process all PDFs."


def check_data_dir_writable() -> CheckResult:
    """Check DATA_DIR exists and is writable (for state file and Excel)."""
    try:
        from config import DATA_DIR, EXTRACTION_STATE_PATH, EXCEL_OUTPUT_PATH
    except Exception as e:
        return False, f"config failed: {e}"
    if not DATA_DIR.exists():
        return False, f"DATA_DIR does not exist: {DATA_DIR}"
    test_file = DATA_DIR / ".connection_check"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        return True, f"DATA_DIR={DATA_DIR} (state={EXTRACTION_STATE_PATH.name}, excel={EXCEL_OUTPUT_PATH.name})"
    except OSError as e:
        return False, f"DATA_DIR not writable: {e}"


def check_pymupdf() -> CheckResult:
    """Check PyMuPDF available for PDF text extraction."""
    try:
        import fitz
        return True, "PyMuPDF (fitz) OK"
    except ImportError as e:
        return False, f"PyMuPDF not installed: {e}"


def check_gcs_output_bucket() -> CheckResult:
    """If GCS_OUTPUT_BUCKET is set, verify we can access the bucket (Excel output is uploaded here; PDFs are not stored)."""
    try:
        from config import GCS_OUTPUT_BUCKET
    except Exception as e:
        return False, f"Config failed: {e}"
    if not GCS_OUTPUT_BUCKET:
        return True, "GCS_OUTPUT_BUCKET not set (Excel and PDFs only local)"
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_OUTPUT_BUCKET)
        bucket.reload()
        # Verify write access
        blob = bucket.blob(".connection_test")
        blob.upload_from_string("ok")
        blob.delete()
        return True, f"bucket={GCS_OUTPUT_BUCKET} (Read/Write OK)"
    except Exception as e:
        err_str = str(e).upper()
        hint = ""
        if "403" in err_str and ("VPC_SERVICE_CONTROLS" in err_str or "SECURITY_POLICY_VIOLATED" in err_str or "ORGANIZATION'S POLICY" in err_str):
            hint = " - VPC Service Controls: org policy is blocking access. Provide the vpcServiceControlsUniqueIdentifier to your GCP Admin."
        return False, f"GCS bucket access failed: {type(e).__name__}: {e}{hint}"


def run_all_checks(
    *,
    gcp_only: bool = False,
    sharepoint_only: bool = False,
    config_only: bool = False,
    sharepoint_full: bool = False,
) -> None:
    checks: list[tuple[str, str, callable]] = []  # (category, name, fn)
    list_pdfs_check = check_sharepoint_list_pdfs_full if sharepoint_full else check_sharepoint_path_quick
    list_pdfs_name = "List PDFs (nested)" if sharepoint_full else "Path reachable"

    if config_only:
        checks = [
            ("Config", "GCP credentials file", check_gcp_credentials_file),
            ("Config", "GCP project", check_gcp_project),
            ("Config", "Gemini model names", check_gemini_model_config),
            ("Config", "SharePoint env", check_sharepoint_config),
            ("Config", "GCS output bucket", check_gcs_output_bucket),
            ("Config", "Data dir writable", check_data_dir_writable),
            ("Config", "PyMuPDF", check_pymupdf),
        ]
    elif gcp_only:
        checks = [
            ("GCP", "Credentials file", check_gcp_credentials_file),
            ("GCP", "GCP project", check_gcp_project),
            ("GCP", "Gemini model config", check_gemini_model_config),
            ("GCP", "Gemini/Vertex reachable", check_gemini_reachable),
            ("Config", "Data dir writable", check_data_dir_writable),
            ("Config", "PyMuPDF", check_pymupdf),
        ]
    elif sharepoint_only:
        checks = [
            ("SharePoint", "Env config", check_sharepoint_config),
            ("SharePoint", "get_access_token", check_sharepoint_token),
            ("SharePoint", "Site + drive ID", check_sharepoint_site_and_drive),
            ("SharePoint", list_pdfs_name, list_pdfs_check),
        ]
    else:
        checks = [
            ("Config", "GCP credentials file", check_gcp_credentials_file),
            ("Config", "GCP project", check_gcp_project),
            ("Config", "Gemini model names", check_gemini_model_config),
            ("GCP", "Gemini/Vertex reachable", check_gemini_reachable),
            ("Config", "SharePoint env", check_sharepoint_config),
            ("SharePoint", "get_access_token", check_sharepoint_token),
            ("SharePoint", "Site + drive ID", check_sharepoint_site_and_drive),
            ("SharePoint", list_pdfs_name, list_pdfs_check),
            ("GCP", "GCS output bucket", check_gcs_output_bucket),
            ("Config", "Data dir writable", check_data_dir_writable),
            ("Config", "PyMuPDF", check_pymupdf),
        ]

    failed = 0
    for category, name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"unexpected error: {type(e).__name__}: {e}"
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{status}] {category} / {name}")
        print(f"         {detail}")
    print()
    if failed:
        print(f"Result: {failed} check(s) failed. Fix the FAIL items above (function/reason in detail).")
        sys.exit(1)
    print("Result: All checks passed.")
    sys.exit(0)


def run_sharepoint_browse(path: str = "", max_depth: int = 2) -> None:
    """List SharePoint site contents at path (default root). Use to discover folder paths for SHAREPOINT_DRIVE_PATH."""
    try:
        from sharepoint_utils import browse_site_contents, is_sharepoint_configured
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not is_sharepoint_configured():
        print("SharePoint not configured (set SHAREPOINT_* in .env).")
        sys.exit(1)
    display_path = path or "(drive root)"
    print(f"SharePoint site contents at: {display_path}\n")
    try:
        items = browse_site_contents(folder_path=path, max_depth=max_depth)
    except Exception as e:
        print(f"Failed to list contents: {e}")
        sys.exit(1)
    if not items:
        print("  (empty or path not found)")
        if path:
            print("\nTip: Try an empty path to list drive root: python check_connections.py --sharepoint-browse")
        sys.exit(0)
    for it in items:
        icon = "[DIR] " if it.get("is_folder") else "[FILE]"
        print(f"  {icon} {it['name']}")
        print(f"       path for SHAREPOINT_DRIVE_PATH: {it['path']}")
        if it.get("is_folder") and it.get("children"):
            for c in it["children"]:
                cicon = "[DIR] " if c.get("is_folder") else "[FILE]"
                print(f"         {cicon} {c['name']}")
                print(f"                path: {c['path']}")
    print("\nTo list inside a folder, run:")
    print("  python check_connections.py --sharepoint-browse \"<path>\"")
    print("Or use interactive: python check_connections.py --sharepoint-browse-interactive")
    sys.exit(0)


def run_sharepoint_browse_interactive() -> None:
    """
    Connect to site → list all drives → list folders at current path → navigate into
    nested folders until user chooses 'done'. Then print the path for SHAREPOINT_DRIVE_PATH.
    """
    try:
        from sharepoint_utils import (
            get_site_connection,
            list_contents_at_path,
            is_sharepoint_configured,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    if not is_sharepoint_configured():
        print("SharePoint not configured (set SHAREPOINT_* in .env).")
        sys.exit(1)
    print("Connecting to SharePoint site...")
    try:
        token, site_id, drives, default_drive_id = get_site_connection()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
    print("Connected.\n")
    drive_id = default_drive_id
    if len(drives) > 1:
        print("Drives (document libraries) on this site:")
        for i, d in enumerate(drives, 1):
            default = " (default)" if d["id"] == default_drive_id else ""
            print(f"  {i}) {d['name']}{default}")
        try:
            choice = input("\nEnter number to browse that drive [default=1]: ").strip() or "1"
            idx = int(choice)
            if 1 <= idx <= len(drives):
                drive_id = drives[idx - 1]["id"]
            else:
                print("Using default drive.")
        except ValueError:
            print("Using default drive.")
    else:
        print(f"Drive: {drives[0]['name']}\n")
    chosen_drive_id = drive_id  # remember which drive user is browsing (may differ from default)
    path_stack: list[str] = []
    while True:
        current_path = "/".join(path_stack)
        try:
            items = list_contents_at_path(current_path, token=token, drive_id=drive_id)
        except Exception as e:
            print(f"Failed to list contents: {e}")
            sys.exit(1)
        folders = [i for i in items if i.get("is_folder")]
        files = [i for i in items if not i.get("is_folder")]
        location = current_path or "(drive root)"
        print(f"\n--- Current location: {location} ---")
        if not folders and not files:
            print("  (empty)")
        else:
            for i, f in enumerate(folders, 1):
                print(f"  {i}) [DIR]  {f['name']}")
            for f in files:
                print(f"      [FILE] {f['name']}")
        print("  0) " + (".. (up)" if path_stack else "Done - use this path as SHAREPOINT_DRIVE_PATH"))
        raw = input("\nEnter number (or 'done' / 'q' to finish): ").strip().lower()
        if raw in ("done", "q", ""):
            if path_stack:
                pass  # treat as done with current path
            else:
                pass  # done at root
            final_path = "/".join(path_stack)
            print("\n" + "=" * 50)
            if chosen_drive_id != default_drive_id:
                print("You browsed a non-default drive. Add to .env so list/PDF use the same drive:")
                print(f"  SHAREPOINT_DRIVE_ID={chosen_drive_id}")
                print()
            if final_path:
                print("Set in .env (no leading slash):")
                print(f"  SHAREPOINT_DRIVE_PATH={final_path}")
            else:
                print("You are at drive root. Set in .env:")
                print("  SHAREPOINT_DRIVE_PATH=")
                print("(empty = process PDFs from entire drive root)")
            print("=" * 50)
            sys.exit(0)
        try:
            num = int(raw)
        except ValueError:
            print("Enter a number, 'done', or 'q'.")
            continue
        if num == 0:
            if path_stack:
                path_stack.pop()
            else:
                final_path = "/".join(path_stack)
                print("\nSet SHAREPOINT_DRIVE_PATH=" + (f'"{final_path}"' if final_path else "(empty)"))
                sys.exit(0)
            continue
        if 1 <= num <= len(folders):
            path_stack.append(folders[num - 1]["name"])
        else:
            print("Invalid number.")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run connectivity and config checks (real credentials).")
    p.add_argument("--gcp", action="store_true", help="Only GCP + Gemini + data dir")
    p.add_argument("--sharepoint", action="store_true", help="Only SharePoint checks (quick path check; use --sharepoint-full for full nested PDF list)")
    p.add_argument("--sharepoint-full", action="store_true", help="With --sharepoint: list all nested PDFs (up to 30 min). Default is quick path-only.")
    p.add_argument("--config", action="store_true", help="Only config summary (no network calls)")
    p.add_argument(
        "--sharepoint-browse",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="List SharePoint site contents at PATH (default: drive root). Use to find the correct SHAREPOINT_DRIVE_PATH.",
    )
    p.add_argument(
        "--sharepoint-browse-interactive",
        action="store_true",
        help="Connect to site, list drives, then navigate into nested folders until you choose the desired path.",
    )
    args = p.parse_args()
    if args.sharepoint_browse_interactive:
        run_sharepoint_browse_interactive()
        return
    if args.sharepoint_browse is not None:
        run_sharepoint_browse(path=args.sharepoint_browse or "")
        return
    print("Connection and config checks (using .env / real credentials)\n")
    run_all_checks(
        gcp_only=args.gcp,
        sharepoint_only=args.sharepoint,
        config_only=args.config,
        sharepoint_full=args.sharepoint_full,
    )


if __name__ == "__main__":
    main()
