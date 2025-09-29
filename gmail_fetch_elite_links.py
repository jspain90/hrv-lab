#!/usr/bin/env python3
import base64, re, sys
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict

import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# --- Paths (relative to this file) ---
BASE = Path(__file__).resolve().parent
INBOUND_DIR = BASE / "data_pipeline" / "io" / "inbound"
CREDENTIALS = BASE / "credentials.json"  # OAuth client (Desktop app)
TOKEN       = BASE / "token.json"        # created on first auth

# --- Gmail search config ---
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
SUBJECT_QUERY = 'subject:"Elite HRV Export"'
PROCESSED_LABEL = "Imported/EliteHRV"
LINK_RE = re.compile(r"https://app\.elitehrv\.com/export/[A-Za-z0-9\-_]+", re.IGNORECASE)

# ----------------- Gmail helpers -----------------
def gmail_service():
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds)

def get_or_create_label(svc, name: str) -> str:
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    for lb in labels:
        if lb["name"] == name:
            return lb["id"]
    body = {"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
    created = svc.users().labels().create(userId="me", body=body).execute()
    return created["id"]

def list_unread_matching(svc) -> List[str]:
    res = svc.users().messages().list(userId="me", q=SUBJECT_QUERY + " is:unread").execute()
    msgs = res.get("messages", [])
    while "nextPageToken" in res:
        res = svc.users().messages().list(userId="me", q=SUBJECT_QUERY + " is:unread",
                                          pageToken=res["nextPageToken"]).execute()
        msgs.extend(res.get("messages", []))
    return [m["id"] for m in msgs] if msgs else []

def extract_links_from_message(svc, msg_id: str) -> List[str]:
    msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
    links = set()
    stack = [msg.get("payload", {})]
    while stack:
        p = stack.pop()
        if "parts" in p:
            stack.extend(p["parts"])
            continue
        data = p.get("body", {}).get("data")
        if not data:
            continue
        text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for m in LINK_RE.findall(text):
            links.add(m)
    return sorted(links)

def label_and_archive(svc, msg_id: str, label_id: str):
    svc.users().messages().modify(
        userId="me", id=msg_id,
        body={"removeLabelIds": ["INBOX", "UNREAD"], "addLabelIds": [label_id]}
    ).execute()

# ----------------- Download helper -----------------
def safe_save_zip(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = Path(urlparse(url).path).name
    out = out_dir / f"{slug}.zip"
    if out.exists():
        # avoid collisions
        i = 2
        while True:
            cand = out_dir / f"{slug} ({i}).zip"
            if not cand.exists():
                out = cand
                break
            i += 1
    r = requests.get(url, allow_redirects=True, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"download failed: {r.status_code}")
    out.write_bytes(r.content)
    return out

# ----------------- One-shot public API -----------------
def process_gmail() -> Dict[str, int]:
    """
    One-shot fetch:
      - find unread 'Elite HRV Export' messages
      - download all export links to INBOUND_DIR
      - label each message 'Imported/EliteHRV' and mark read
    Returns: dict summary with counts.
    """
    svc = gmail_service()
    label_id = get_or_create_label(svc, PROCESSED_LABEL)
    msg_ids = list_unread_matching(svc)

    downloaded = 0
    labeled = 0
    for msg_id in msg_ids:
        links = extract_links_from_message(svc, msg_id)
        for url in links:
            try:
                safe_save_zip(url, INBOUND_DIR)
                downloaded += 1
            except Exception as e:
                print(f"[GMAIL] download failed for {url}: {e}", file=sys.stderr)
        label_and_archive(svc, msg_id, label_id)
        labeled += 1

    return {"messages_labeled": labeled, "zips_downloaded": downloaded}

# ----------------- CLI entrypoint (optional) -----------------
if __name__ == "__main__":
    try:
        summary = process_gmail()
        print(json.dumps(summary))
    except HttpError as e:
        print(f"[GMAIL] error: {e}", file=sys.stderr)
        sys.exit(1)
