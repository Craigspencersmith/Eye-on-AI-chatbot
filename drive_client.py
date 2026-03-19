"""
Google Drive client for listing and exporting Google Docs as plain text.
Uses a service account for authentication.
"""

import logging
from typing import Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build, Resource

from config import config

logger = logging.getLogger(__name__)

# Scopes needed: read-only access to Drive files
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service() -> Resource:
    """Build and return an authenticated Google Drive API service."""
    creds = Credentials.from_service_account_file(
        config.GOOGLE_CREDENTIALS_PATH, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service


def list_google_docs(service: Resource | None = None) -> list[dict[str, Any]]:
    """
    List all Google Docs in the configured Drive folder.

    Returns a list of dicts with keys: id, name, modifiedTime, createdTime.
    Handles pagination automatically.
    """
    if service is None:
        service = get_drive_service()

    docs: list[dict[str, Any]] = []
    page_token: str | None = None

    query = (
        f"'{config.DRIVE_FOLDER_ID}' in parents "
        "and mimeType='application/vnd.google-apps.document' "
        "and trashed=false"
    )

    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, modifiedTime, createdTime)",
                pageSize=100,
                pageToken=page_token,
                orderBy="name",
            )
            .execute()
        )

        docs.extend(response.get("files", []))
        page_token = response.get("nextPageToken")

        if not page_token:
            break

    logger.info("Found %d Google Docs in folder %s", len(docs), config.DRIVE_FOLDER_ID)
    return docs


def export_doc_as_text(doc_id: str, service: Resource | None = None) -> str:
    """Export a Google Doc as plain text."""
    if service is None:
        service = get_drive_service()

    content = (
        service.files()
        .export(fileId=doc_id, mimeType="text/plain")
        .execute()
    )

    # API returns bytes
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)
