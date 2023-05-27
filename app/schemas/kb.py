from pydantic import BaseModel, Field


class KbResource(BaseModel):
    url: str
    type: str
    index: str


class KbIngestDto(BaseModel):
    resources: list[KbResource]


class KbRetrievalDto(BaseModel):
    user_input: str
    index: str = "main"
    ranking_size: int = 3


class FlexKbResource(BaseModel):
    url: str
    type: str
    index: str = Field(default="main")


class FlexKbIngestDto(BaseModel):
    resources: list[FlexKbResource]


class KbAskDto(BaseModel):
    session_id: str = Field(default="test")
    personality: str = Field(default="amable")
    user_input: str
    language: str = Field(default="español de Argentina")
    company_name: str = Field(default="Yoizen")
    default_response: str = Field(default="Lo siento, no puedo responder eso")
    index: str = Field(default="main")
    emoji_level: int = Field(default=4)


class FlexKbAskDto(BaseModel):
    session_id: str = "test"
    personality: str = "amable"
    user_input: str
