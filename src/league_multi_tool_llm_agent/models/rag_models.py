from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RagSearchResult(BaseModel):
    id: int
    doc_type: str
    champion_name: str | None = None
    skin_name: str | None = None
    main_role: str | None = None
    difficulty: int | None = None
    source_url: str | None = None
    content: str
    meta_json: dict[str, Any]
    distance: float
