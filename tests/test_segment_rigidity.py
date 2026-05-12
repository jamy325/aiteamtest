import ast
from pathlib import Path

from core.types import Segment
from services.segment_rigidity import SegmentRigidityPolicy


def _segment(segment_id: str, segment_type: str, *, locked: bool = False) -> Segment:
    return Segment(
        segment_id=segment_id,
        path_id="path_1",
        type=segment_type,
        locked=locked,
    )


def test_segment_rigidity_policy_maps_segment_types_to_expected_levels() -> None:
    policy = SegmentRigidityPolicy()

    assert policy.rigidity_for_type("line") == "high"
    assert policy.rigidity_for_type("circle") == "high"
    assert policy.rigidity_for_type("arc") == "high"
    assert policy.rigidity_for_type("ellipse") == "medium_high"
    assert policy.rigidity_for_type("bezier") == "medium"
    assert policy.rigidity_for_type("bspline") == "low"
    assert policy.rigidity_for_type("polyline") == "low"


def test_segment_rigidity_policy_prefers_unlocked_segment_for_movement() -> None:
    policy = SegmentRigidityPolicy()
    locked_line = _segment("seg_locked", "line", locked=True)
    free_polyline = _segment("seg_free", "polyline")

    decision = policy.choose_segment_to_move(locked_line, free_polyline)

    assert decision.strategy == "move_right"
    assert decision.move_segment_id == "seg_free"
    assert decision.reference_segment_id == "seg_locked"
    assert decision.move_rigidity == "low"
    assert decision.reference_rigidity == "high"
    assert decision.reason == "left_locked_move_right"
    assert decision.blocked is False


def test_segment_rigidity_policy_moves_left_when_right_segment_is_locked() -> None:
    policy = SegmentRigidityPolicy()
    free_bezier = _segment("seg_left", "bezier")
    locked_arc = _segment("seg_right", "arc", locked=True)

    decision = policy.choose_segment_to_move(free_bezier, locked_arc)

    assert decision.strategy == "move_left"
    assert decision.move_segment_id == "seg_left"
    assert decision.reference_segment_id == "seg_right"
    assert decision.move_rigidity == "medium"
    assert decision.reference_rigidity == "high"
    assert decision.reason == "right_locked_move_left"
    assert decision.blocked is False


def test_segment_rigidity_policy_moves_lower_rigidity_segment() -> None:
    policy = SegmentRigidityPolicy()
    bezier = _segment("seg_bezier", "bezier")
    circle = _segment("seg_circle", "circle")

    decision = policy.choose_segment_to_move(bezier, circle)

    assert decision.strategy == "move_left"
    assert decision.move_segment_id == "seg_bezier"
    assert decision.reference_segment_id == "seg_circle"
    assert decision.move_rigidity == "medium"
    assert decision.reference_rigidity == "high"
    assert decision.reason == "move_less_rigid_left"


def test_segment_rigidity_policy_blocks_when_both_segments_are_locked() -> None:
    policy = SegmentRigidityPolicy()
    left = _segment("seg_left", "line", locked=True)
    right = _segment("seg_right", "bezier", locked=True)

    decision = policy.choose_segment_to_move(left, right)

    assert decision.strategy == "blocked"
    assert decision.blocked is True
    assert decision.move_segment_id is None
    assert decision.reference_segment_id is None
    assert decision.reason == "both_locked"


def test_segment_rigidity_policy_uses_midpoint_strategy_for_low_rigidity_pair() -> None:
    policy = SegmentRigidityPolicy()
    left = _segment("seg_left", "bspline")
    right = _segment("seg_right", "polyline")

    decision = policy.choose_segment_to_move(left, right)

    assert decision.strategy == "move_both_midpoint"
    assert decision.move_segment_id is None
    assert decision.reference_segment_id is None
    assert decision.move_rigidity == "low"
    assert decision.reference_rigidity == "low"
    assert decision.reason == "equal_low_rigidity_move_both_midpoint"


def test_segment_rigidity_policy_uses_minimal_strategy_for_high_rigidity_pair() -> None:
    policy = SegmentRigidityPolicy()
    left = _segment("seg_left", "line")
    right = _segment("seg_right", "arc")

    decision = policy.choose_segment_to_move(left, right)

    assert decision.strategy == "move_both_minimal"
    assert decision.move_segment_id is None
    assert decision.reference_segment_id is None
    assert decision.move_rigidity == "high"
    assert decision.reference_rigidity == "high"
    assert decision.reason == "equal_high_rigidity_move_both_minimal"


def test_segment_rigidity_policy_does_not_mutate_input_segments() -> None:
    policy = SegmentRigidityPolicy()
    left = _segment("seg_left", "ellipse")
    right = _segment("seg_right", "ellipse")
    original_left = left
    original_right = right

    decision = policy.choose_segment_to_move(left, right)

    assert decision.strategy == "move_both_minimal"
    assert decision.reason == "equal_medium_high_rigidity_move_both_minimal"
    assert left == original_left
    assert right == original_right


def test_segment_rigidity_service_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/segment_rigidity.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
