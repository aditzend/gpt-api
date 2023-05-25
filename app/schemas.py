from pydantic import BaseModel, Field


class KbResource(BaseModel):
    url: str
    type: str
    index: str


class KbIngestDto(BaseModel):
    resources: list[KbResource]
