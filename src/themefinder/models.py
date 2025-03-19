import pydantic
from pydantic import BaseModel, model_validator


class SentimentAnalysisInput(BaseModel):
    response_id: str | int
    response: str

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self


class SentimentAnalysisOutput(BaseModel):
    response_id: str | int
    response: str
    position: str

    @model_validator(mode="after")
    def validate_position(self):
        allowed_positions = {"agreement", "disagreement", "unclear"}
        if self.position not in allowed_positions:
            raise ValueError(f"position must be one of {allowed_positions}")
        return self
    
    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self

class ThemeGenerationInput(BaseModel):
    response_id: str | int
    response: str
    position: str

    @model_validator(mode="after")
    def validate_position(self):
        allowed_positions = {"agreement", "disagreement", "unclear"}
        if self.position not in allowed_positions:
            raise ValueError(f"position must be one of {allowed_positions}")
        return self
    
    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self

class ThemeGenerationOutput(BaseModel):
    topic_label: str
    topic_description: str
    position: str

    @model_validator(mode="after")
    def validate_position(self):
        allowed_positions = {"agreement", "disagreement", "unclear"}
        if self.position not in allowed_positions:
            raise ValueError(f"position must be one of {allowed_positions}")
        return self

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self

class ThemeCondensationInput(BaseModel):
    topic_label: str
    topic_description: str
    position: str

    @model_validator(mode="after")
    def validate_position(self):
        allowed_positions = {"agreement", "disagreement", "unclear"}
        if self.position not in allowed_positions:
            raise ValueError(f"position must be one of {allowed_positions}")
        return self

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self

class ThemeCondensationOutput(BaseModel):
    topic_label: str
    topic_description: str
    source_topic_count: int

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} cannot be negative")
        return self

class ThemeRefinementInput(BaseModel):
    topic_label: str
    topic_description: str
    source_topic_count: int

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} cannot be negative")
        return self

class ThemeRefinementOutput(BaseModel):
    topic_id: str
    topic: str
    source_topic_count: int

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} cannot be negative")
        return self

class ThemeMappingResponseInput(BaseModel):
    response_id: str | int
    response: str
    position: str

    @model_validator(mode="after")
    def validate_position(self):
        allowed_positions = {"agreement", "disagreement", "unclear"}
        if self.position not in allowed_positions:
            raise ValueError(f"position must be one of {allowed_positions}")
        return self
    
    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self

class ThemeMappingThemeInput(BaseModel):

    topic_id: str
    topic: str

    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} cannot be negative")
        return self


class ThemeMappingOutput(BaseModel):

    response_id: int = Field(ge=0)
    response: str
    position: str
    labels: List[str]
    reasons: List[str]
    stances: List[str]
    
    @model_validator(mode="after")
    def validate_position(self):
        allowed_positions = {"agreement", "disagreement", "unclear"}
        if self.position not in allowed_positions:
            raise ValueError(f"position must be one of {allowed_positions}")
        return self
    
    @model_validator(mode="after")
    def validate_lengths(self):
        if len(self.reasons) != len(self.labels):
            raise ValueError("'reasons' must have the same length as 'labels'")
        if len(self.stances) != len(self.labels):
            raise ValueError("'stances' must have the same length as 'labels'")
        return self
    
    @model_validator(mode="after")
    def validate_non_empty_fields(self):
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or only whitespace")
            if isinstance(value, list) and not value:
                raise ValueError(f"{field_name} cannot be an empty list")
        return self
