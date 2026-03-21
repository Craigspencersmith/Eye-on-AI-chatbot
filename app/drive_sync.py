"""
Google Drive sync — pulls Google Docs as plain text from a shared folder.
Tracks which files have been synced (by file ID + modifiedTime) to avoid re-processing.
"""

import json
import os
import logging
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config import settings

logger = logging.getLogger(__name__)

SYNC_STATE_FILE = os.path.join(settings.CHROMA_PERSIST_DIR, "sync_state.json")


def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        settings.GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


def load_sync_state() -> dict:
    """Load sync state: {file_id: {"name": str, "modifiedTime": str}}"""
    if os.path.exists(SYNC_STATE_FILE):
        with open(SYNC_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_sync_state(state: dict):
    os.makedirs(os.path.dirname(SYNC_STATE_FILE), exist_ok=True)
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def list_all_docs(service) -> list[dict]:
    """List all Google Docs in the configured Drive folder."""
    all_files = []
    page_token = None
    while True:
        results = (
            service.files()
            .list(
                q=f"'{settings.DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.document'",
                pageSize=100,
                fields="nextPageToken, files(id, name, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        all_files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    return all_files


def export_doc_as_text(service, file_id: str) -> str:
    """Export a Google Doc as plain text."""
    content = (
        service.files()
        .export(fileId=file_id, mimeType="text/plain")
        .execute()
    )
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return content


def get_new_or_updated_docs() -> list[dict]:
    """
    Returns list of docs that are new or modified since last sync.
    Each dict has: id, name, modifiedTime, content (plain text).
    """
    service = get_drive_service()
    all_docs = list_all_docs(service)
    sync_state = load_sync_state()

    to_process = []
    for doc in all_docs:
        file_id = doc["id"]
        modified = doc["modifiedTime"]
        prev = sync_state.get(file_id)

        if prev and prev.get("modifiedTime") == modified:
            continue  # unchanged

        logger.info(f"Fetching doc: {doc['name']} (id={file_id})")
        try:
            content = export_doc_as_text(service, file_id)
            to_process.append(
                {
                    "id": file_id,
                    "name": doc["name"],
                    "modifiedTime": modified,
                    "content": content,
                }
            )
        except Exception as e:
            logger.error(f"Failed to export {doc['name']}: {e}")

    return to_process


def mark_synced(docs: list[dict]):
    """Update sync state after successful indexing."""
    state = load_sync_state()
    for doc in docs:
        state[doc["id"]] = {
            "name": doc["name"],
            "modifiedTime": doc["modifiedTime"],
        }
    save_sync_state(state)


def get_total_doc_count() -> int:
    """Get total number of docs in the Drive folder."""
    service = get_drive_service()
    return len(list_all_docs(service))
