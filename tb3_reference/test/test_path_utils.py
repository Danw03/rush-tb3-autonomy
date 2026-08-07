"""Unit tests for reference-path geometry and cone feature validation."""

import math

import pytest

from tb3_reference.path_utils import (
    cone_features_are_valid,
    crop_polyline,
    heading_yaws,
    resample_polyline,
)


def test_accepts_new_cone_feature_layout() -> None:
    """Accept the current seven-value feature ordering."""
    features = [2.0, 0.4, 0.8, 0.0, 0.2, 0.3, 7.0]

    assert cone_features_are_valid(features, minimum_cone_count=1)


def test_rejects_old_path_valid_assumption_and_empty_detection() -> None:
    """Reject empty detections and feature vectors of the wrong length."""
    no_cones = [0.0, math.inf, 0.0, 0.0, 0.0, 0.0, 4.0]

    assert not cone_features_are_valid(no_cones, minimum_cone_count=1)
    assert not cone_features_are_valid([1.0, 0.4], minimum_cone_count=1)


def test_crops_and_resamples_straight_path() -> None:
    """Crop and uniformly resample a straight reference segment."""
    cropped = crop_polyline([(0.0, 0.0), (2.0, 0.0)], 0.5)
    sampled = resample_polyline(cropped, spacing=0.2)
    expected = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.5, 0.0)]

    assert len(sampled) == len(expected)
    for actual, wanted in zip(sampled, expected):
        assert actual == pytest.approx(wanted)


def test_heading_yaws_follow_curve_tangent() -> None:
    """Compute tangent headings along a right-angle polyline."""
    headings = heading_yaws([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])

    assert headings[0] == pytest.approx(0.0)
    assert headings[1] == pytest.approx(math.pi / 4.0)
    assert headings[2] == pytest.approx(math.pi / 2.0)
