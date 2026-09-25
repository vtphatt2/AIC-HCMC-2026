from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual: str = ""
    transcript: str | None = None
    min_offset_ms: int = Field(default=0, ge=0, le=120_000)

    @model_validator(mode="after")
    def require_a_query(self):
        self.visual = self.visual.strip()
        if self.transcript is not None:
            self.transcript = self.transcript.strip() or None
        if not self.visual and not self.transcript:
            raise ValueError("an event needs a visual or transcript query")
        return self


class SearchAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["frames", "transcript"]
    strategy: str
    events: list[SearchEvent] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_shape(self):
        self.strategy = self.strategy.strip()
        if not self.strategy:
            raise ValueError("strategy must not be empty")
        if self.mode == "transcript" and len(self.events) != 1:
            raise ValueError("transcript search accepts exactly one event")
        return self


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_summary: str = Field(min_length=1, max_length=240)
    primary: SearchAttempt
    fallback: SearchAttempt | None


class VortaCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_mode: str
    strategies: list[dict]
    transcript_default: str
    transcript_algorithms: list[dict]
