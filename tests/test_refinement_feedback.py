"""Tests for refinement feedback flow - verifying judge feedback is passed to refiner."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from langstruct.core.refinement import (
    Budget,
    BuiltinJudge,
    CandidateResult,
    CustomJudge,
    ExtractionRefiner,
    Refine,
    RefinementEngine,
    RefinementStrategy,
    RefinementTrace,
)
from langstruct.core.schemas import ExtractionResult, SourceSpan


class PersonTestSchema(BaseModel):
    """Test schema for refinement tests."""

    name: str = Field(description="Name of the person")
    age: int = Field(description="Age in years")


@pytest.fixture
def sample_extraction():
    """Sample extraction result for testing."""
    return ExtractionResult(
        entities={"name": "John Doe", "age": 30},
        sources={
            "name": [SourceSpan(start=0, end=8, text="John Doe")],
            "age": [SourceSpan(start=15, end=17, text="30")],
        },
        confidence=0.85,
        metadata={},
    )


@pytest.fixture
def sample_candidates(sample_extraction):
    """Multiple candidate extractions for testing."""
    candidate1 = sample_extraction
    candidate2 = ExtractionResult(
        entities={"name": "Jane Doe", "age": 25},
        sources={
            "name": [SourceSpan(start=0, end=8, text="Jane Doe")],
            "age": [SourceSpan(start=15, end=17, text="25")],
        },
        confidence=0.75,
        metadata={},
    )
    return [candidate1, candidate2]


class TestCandidateResultFeedback:
    """Test that CandidateResult properly stores feedback."""

    def test_candidate_result_stores_feedback(self, sample_extraction):
        """CandidateResult should store a feedback field."""
        # Should work with feedback
        result = CandidateResult(
            extraction=sample_extraction,
            score=0.9,
            reasoning="Good extraction",
            feedback="Consider verifying the age from additional context",
            candidate_id=0,
        )
        assert result.feedback == "Consider verifying the age from additional context"

    def test_candidate_result_feedback_is_required(self, sample_extraction):
        """CandidateResult should fail without feedback."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            CandidateResult(
                extraction=sample_extraction,
                score=0.9,
                reasoning="Good extraction",
                # Missing feedback
                candidate_id=0,
            )


class TestJudgeFeedbackReturn:
    """Test that judges return feedback in their tuples."""

    def test_builtin_judge_returns_feedback_tuple(self, sample_candidates):
        """BuiltinJudge should return (score, reasoning, feedback) tuples."""
        judge = BuiltinJudge(PersonTestSchema)

        # Mock the DSPy judge call
        mock_scores = json.dumps(
            [
                {
                    "score": 0.9,
                    "reasoning": "Good extraction with accurate values",
                    "feedback": "Consider adding source span for middle name",
                },
                {
                    "score": 0.7,
                    "reasoning": "Missing some details",
                    "feedback": "Verify age against document header",
                },
            ]
        )

        with patch.object(judge, "judge") as mock_judge:
            mock_result = MagicMock()
            mock_result.scores = mock_scores
            mock_judge.return_value = mock_result

            scores = judge.forward(
                "Sample text about John Doe, age 30", sample_candidates
            )

            assert len(scores) == 2
            # Each score should be a 3-tuple: (score, reasoning, feedback)
            assert len(scores[0]) == 3
            assert len(scores[1]) == 3

            # Verify feedback is extracted
            assert scores[0][2] == "Consider adding source span for middle name"
            assert scores[1][2] == "Verify age against document header"

    def test_custom_judge_returns_feedback_tuple(self, sample_candidates):
        """CustomJudge should return (score, reasoning, feedback) tuples."""
        custom_rubric = "Focus on name accuracy"
        judge = CustomJudge(PersonTestSchema, custom_rubric)

        mock_scores = json.dumps(
            [
                {
                    "score": 0.85,
                    "reasoning": "Name matches well",
                    "feedback": "Double-check spelling of last name",
                },
                {
                    "score": 0.65,
                    "reasoning": "Name partially correct",
                    "feedback": "Extract full legal name from signature block",
                },
            ]
        )

        with patch.object(judge, "judge") as mock_judge:
            mock_result = MagicMock()
            mock_result.scores = mock_scores
            mock_judge.return_value = mock_result

            scores = judge.forward("Sample text", sample_candidates)

            assert len(scores) == 2
            assert len(scores[0]) == 3
            assert scores[0][2] == "Double-check spelling of last name"
            assert scores[1][2] == "Extract full legal name from signature block"

    def test_judge_fallback_provides_default_feedback(self, sample_candidates):
        """Judge should provide default feedback when parsing fails."""
        judge = BuiltinJudge(PersonTestSchema)

        with patch.object(judge, "judge") as mock_judge:
            mock_result = MagicMock()
            mock_result.scores = "invalid json"
            mock_judge.return_value = mock_result

            scores = judge.forward("Sample text", sample_candidates)

            assert len(scores) == 2
            # Fallback should include feedback
            assert scores[0][2] == "No feedback available"
            assert scores[1][2] == "No feedback available"


class TestRefinementEngineFeedbackFlow:
    """Test that RefinementEngine passes feedback correctly."""

    def test_feedback_stored_in_trace_candidates(self, sample_candidates):
        """Feedback should be stored in trace.candidates."""
        # Create a mock extractor
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Mock the judge to return scores with feedback
        mock_scores = [
            (0.9, "Good extraction", "Improve source span accuracy"),
            (0.7, "Partial extraction", "Add missing fields"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                config = Refine(
                    strategy=RefinementStrategy.BON,
                    n_candidates=2,
                    max_refine_steps=1,  # Must be >= 1
                )

                result, trace = engine.forward("Sample text", config)

                # Verify feedback is stored in candidates
                assert len(trace.candidates) == 2
                assert trace.candidates[0].feedback == "Improve source span accuracy"
                assert trace.candidates[1].feedback == "Add missing fields"

    def test_feedback_passed_to_refiner(self, sample_candidates):
        """Best candidate's feedback should be passed to refiner as issues."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Mock scores - candidate 0 has higher score
        mock_scores = [
            (0.9, "Good extraction", "Check name spelling in header"),
            (0.7, "Partial extraction", "Add missing fields"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                with patch.object(engine.refiner, "forward") as mock_refiner:
                    # Make refiner return a result
                    mock_refiner.return_value = sample_candidates[0]

                    config = Refine(
                        strategy=RefinementStrategy.BON_THEN_REFINE,
                        n_candidates=2,
                        max_refine_steps=1,
                    )

                    result, trace = engine.forward("Sample text", config)

                    # Verify refiner was called with feedback as issues
                    mock_refiner.assert_called_once()
                    call_args = mock_refiner.call_args
                    # issues should be the feedback from the best candidate (index 0)
                    assert (
                        call_args.kwargs.get("issues")
                        == "Check name spelling in header"
                    )

    def test_feedback_tracked_in_refine_diffs(self, sample_candidates):
        """Feedback used should be tracked in trace.refine_diffs."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Need 2+ candidates for judge to be called
        mock_scores = [
            (0.9, "Good", "Verify age from birth date"),
            (0.7, "Partial", "Check name"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                with patch.object(
                    engine.refiner, "forward", return_value=sample_candidates[0]
                ):
                    with patch.object(
                        engine.refiner,
                        "_detect_issues",
                        return_value="No issues detected",
                    ):
                        config = Refine(
                            strategy=RefinementStrategy.BON_THEN_REFINE,
                            n_candidates=2,  # Need 2+ for judge to be called
                            max_refine_steps=1,
                        )

                        result, trace = engine.forward("Sample text", config)

                        # Verify feedback_used is tracked
                        assert len(trace.refine_diffs) == 1
                        assert (
                            trace.refine_diffs[0]["feedback_used"]
                            == "Verify age from birth date"
                        )

    def test_single_candidate_gets_default_feedback(self, sample_candidates):
        """Single candidate (no judging) should get default feedback."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        config = Refine(
            strategy=RefinementStrategy.REFINE,  # No BON, just refine
            n_candidates=1,
            max_refine_steps=1,  # Must be >= 1
        )

        with patch.object(engine.refiner, "forward", return_value=sample_candidates[0]):
            result, trace = engine.forward("Sample text", config)

            # Single candidate should have default feedback
            assert len(trace.candidates) == 1
            assert trace.candidates[0].feedback == "General quality improvement needed"


class TestExtractionRefinerReceivesFeedback:
    """Test that ExtractionRefiner properly uses the issues/feedback parameter."""

    def test_refiner_uses_issues_parameter(self, sample_extraction):
        """Refiner should use the issues parameter when provided."""
        refiner = ExtractionRefiner(PersonTestSchema)

        feedback = "The age value seems incorrect, verify against document"

        with patch.object(refiner, "refine") as mock_refine:
            mock_result = MagicMock()
            mock_result.refined_extraction = json.dumps({"name": "John Doe", "age": 31})
            mock_refine.return_value = mock_result

            result = refiner.forward("Sample text", sample_extraction, issues=feedback)

            # Verify the refine signature was called with the feedback as issues
            mock_refine.assert_called_once()
            call_kwargs = mock_refine.call_args.kwargs
            assert call_kwargs.get("issues") == feedback

    def test_refiner_auto_detects_issues_when_none_provided(self, sample_extraction):
        """Refiner should auto-detect issues when none provided."""
        refiner = ExtractionRefiner(PersonTestSchema)

        # Create extraction with potential issue (empty value)
        extraction_with_issue = ExtractionResult(
            entities={"name": "", "age": 30},  # Empty name
            sources={},
            confidence=0.5,
            metadata={},
        )

        detected = refiner._detect_issues(
            "Sample text about John", extraction_with_issue
        )

        # Should detect the missing name
        assert "name" in detected.lower() or "missing" in detected.lower()
