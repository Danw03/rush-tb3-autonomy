"""Pure geometry helpers for cone reference-path generation."""

import math
from typing import List, Sequence, Tuple


Point2 = Tuple[float, float]


def _distance(first: Point2, second: Point2) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def cone_features_are_valid(
    features: Sequence[float],
    minimum_cone_count: int,
) -> bool:
    """Validate the seven-value cone feature contract used since 2026-08-06."""
    if len(features) != 7:
        return False

    cone_count = float(features[0])
    if not math.isfinite(cone_count):
        return False
    if cone_count < float(minimum_cone_count):
        return False

    return all(math.isfinite(float(value)) for value in features)


def remove_near_duplicates(
    points: Sequence[Point2],
    minimum_separation: float,
) -> List[Point2]:
    """Remove consecutive points that are too close to define a direction."""
    if not points:
        return []

    output = [(float(points[0][0]), float(points[0][1]))]
    for point in points[1:]:
        candidate = (float(point[0]), float(point[1]))
        if _distance(output[-1], candidate) >= minimum_separation:
            output.append(candidate)

    return output


def crop_polyline(
    points: Sequence[Point2],
    maximum_length: float,
) -> List[Point2]:
    """Crop a polyline at an arc length and interpolate the last point."""
    if len(points) < 2:
        return list(points)

    output = [points[0]]
    accumulated = 0.0

    for start, end in zip(points, points[1:]):
        segment_length = _distance(start, end)
        if segment_length <= 1.0e-12:
            continue

        remaining = maximum_length - accumulated
        if segment_length <= remaining:
            output.append(end)
            accumulated += segment_length
            continue

        ratio = remaining / segment_length
        output.append(
            (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
        )
        break

    return output


def smooth_polyline(
    points: Sequence[Point2],
    window: int,
) -> List[Point2]:
    """Apply a centered moving average while preserving both endpoints."""
    if len(points) < 3 or window <= 1:
        return list(points)

    radius = window // 2
    output = [points[0]]

    for index in range(1, len(points) - 1):
        first = max(0, index - radius)
        last = min(len(points), index + radius + 1)
        samples = points[first:last]
        output.append(
            (
                sum(point[0] for point in samples) / len(samples),
                sum(point[1] for point in samples) / len(samples),
            )
        )

    output.append(points[-1])
    return output


def resample_polyline(
    points: Sequence[Point2],
    spacing: float,
) -> List[Point2]:
    """Linearly resample a polyline at uniform arc-length spacing."""
    if len(points) < 2:
        return list(points)

    cumulative = [0.0]
    for start, end in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + _distance(start, end))

    total_length = cumulative[-1]
    if total_length <= 1.0e-12:
        return [points[0]]

    query_lengths = []
    query = 0.0
    while query < total_length:
        query_lengths.append(query)
        query += spacing
    query_lengths.append(total_length)

    output = []
    segment_index = 0
    for query in query_lengths:
        while (
            segment_index + 1 < len(cumulative)
            and cumulative[segment_index + 1] < query
        ):
            segment_index += 1

        if segment_index + 1 >= len(points):
            output.append(points[-1])
            continue

        start_length = cumulative[segment_index]
        end_length = cumulative[segment_index + 1]
        denominator = max(end_length - start_length, 1.0e-12)
        ratio = (query - start_length) / denominator
        start = points[segment_index]
        end = points[segment_index + 1]
        output.append(
            (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
        )

    return output


def heading_yaws(points: Sequence[Point2]) -> List[float]:
    """Compute forward/central/backward-difference tangent headings."""
    if len(points) < 2:
        raise ValueError('at least two points are required')

    headings = []
    for index in range(len(points)):
        if index == 0:
            start = points[0]
            end = points[1]
        elif index == len(points) - 1:
            start = points[-2]
            end = points[-1]
        else:
            start = points[index - 1]
            end = points[index + 1]

        headings.append(math.atan2(end[1] - start[1], end[0] - start[0]))

    return headings
