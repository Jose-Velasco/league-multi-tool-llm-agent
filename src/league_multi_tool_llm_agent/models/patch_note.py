from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class PatchNoteSection(BaseModel):
    """A single patch note section with cleaned text content."""

    section_index: int
    section_title: str
    section_text_contents: str
    full_text: str


class PatchNotes(BaseModel):
    """Structured patch notes document scraped from Riot patch note pages."""

    model_config = ConfigDict(use_enum_values=True)

    url: HttpUrl
    title: str
    # URL-friendly identifier for a patch note page.
    patch_slug: str | None = None
    patch_version: str | None = None
    tagline: str | None = None
    authors: list[str] = Field(default_factory=list)
    date: str | None = None
    author_context: str | None = None
    text_contents: list[PatchNoteSection] = Field(default_factory=list)
