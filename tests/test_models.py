import pytest
from pydantic import BaseModel, ValidationError

from themefinder.models import (
    SentimentAnalysisOutput,
    ThemeMappingOutput,
    validate_mapping_stance_lengths,
    validate_non_empty_fields,
    validate_position,
    validate_stances,
)


# Helper function to check validation errors
def raises_validation_error(model_class, data):
    with pytest.raises(ValidationError):
        model_class(**data)


# === Mock Model for Direct Validation Function Testing ===
class MockModel(BaseModel):
    response_id: int
    response: str
    position: str = None
    stances: list[str] = None
    labels: list[str] = None
    reasons: list[str] = None


# === Tests for validate_non_empty_fields ===
def test_validate_non_empty_fields_valid():
    model = MockModel(response_id=1, response="Valid response")
    validate_non_empty_fields(model)  # Should not raise error


def test_validate_non_empty_fields_invalid():
    model = MockModel(response_id=1, response="   ")
    with pytest.raises(ValueError, match="response cannot be empty or only whitespace"):
        validate_non_empty_fields(model)


# === Tests for validate_position ===
def test_validate_position_valid():
    model = MockModel(response_id=1, response="Valid response", position="AGREEMENT")
    validate_position(model)  # Should not raise error


def test_validate_position_invalid():
    model = MockModel(
        response_id=1, response="Valid response", position="invalid_value"
    )
    with pytest.raises(ValueError, match="position must be one of"):
        validate_position(model)


# === Tests for validate_stances ===
def test_validate_stances_valid():
    model = MockModel(
        response_id=1, response="Valid response", stances=["POSITIVE", "NEGATIVE"]
    )
    validate_stances(model)  # Should not raise error


def test_validate_stances_invalid():
    model = MockModel(response_id=1, response="Valid response", stances=["INVALID"])
    with pytest.raises(ValueError, match="stances must be one of"):
        validate_stances(model)


# The function to be tested
def validate_mapping_unique_labels(model):
    if len(model.labels) != len(set(model.labels)):
        raise ValueError("'labels' must be unique")
    return model


# Test case for valid unique labels
def test_validate_mapping_unique_labels_valid():
    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["label1", "label2", "label3"],
        reasons=["reason1", "reason2", "reason3"],
        stances=["POSITIVE", "NEGATIVE", "POSITIVE"],
    )
    # Should not raise an error and return the model itself.
    assert validate_mapping_unique_labels(model) == model


# Test case for invalid (duplicate) labels
def test_validate_mapping_unique_labels_invalid():
    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["label1", "label2", "label1"],
        reasons=["reason1", "reason2", "reason3"],
        stances=["POSITIVE", "NEGATIVE", "POSITIVE"],
    )
    with pytest.raises(ValueError, match="'labels' must be unique"):
        validate_mapping_unique_labels(model)


# === Tests for validate_mapping_stance_lengths ===
def test_validate_mapping_stance_lengths_valid():
    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["Label1", "Label2"],
        reasons=["Reason1", "Reason2"],
        stances=["POSITIVE", "NEGATIVE"],
    )
    validate_mapping_stance_lengths(model)  # Should not raise error


def test_validate_mapping_stance_lengths_invalid():
    model = MockModel(
        response_id=1,
        response="Valid response",
        labels=["Label1"],
        reasons=["Reason1"],
        stances=["POSITIVE", "NEGATIVE"],  # Mismatch in length
    )
    with pytest.raises(
        ValueError, match="'stances' must have the same length as 'labels'"
    ):
        validate_mapping_stance_lengths(model)


# === SentimentAnalysisOutput ===
def test_sentiment_analysis_output_valid():
    model = SentimentAnalysisOutput(
        response_id=1, response="Test response", position="AGREEMENT"
    )
    assert model.position == "AGREEMENT"


def test_sentiment_analysis_output_invalid_position():
    raises_validation_error(
        SentimentAnalysisOutput,
        {"response_id": 1, "position": "invalid"},
    )


# === ThemeMappingOutput ===
def test_theme_mapping_output_valid():
    model = ThemeMappingOutput(
        response_id=1,
        response="Valid response",
        position="agreement",
        labels=["Label1", "Label2"],
        reasons=["Reason1", "Reason2"],
        stances=["POSITIVE", "NEGATIVE"],
    )
    assert len(model.labels) == len(model.stances)


def test_theme_mapping_output_invalid_stance():
    raises_validation_error(
        ThemeMappingOutput,
        {
            "response_id": 1,
            "response": "Valid",
            "position": "agreement",
            "labels": ["Label1"],
            "reasons": ["Reason1"],
            "stances": ["InvalidStance"],  # Invalid stance value
        },
    )
