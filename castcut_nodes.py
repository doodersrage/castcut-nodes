"""
Castcut nodes for ComfyUI: the checks Castcut runs after a still lands, inside the job itself.

One file on purpose: ComfyUI-Manager's "copy" install drops single .py files into custom_nodes/,
and a symlinked / cloned folder loads it through this folder's __init__.py.

- CastcutPoseScore    DWPose keypoints (POSE_KEYPOINT) vs the pose guide's keypoints -> score,
                      posture and limb deltas per image. A port of src/lib/pose-score.ts,
                      pose-limb-score.ts and pose-posture.ts; tests/vectors/pose-score.json is
                      checked by both the TypeScript and the Python tests so they stay in step.
- CastcutPickBest     image batch + scores -> best image, the other take, its index, a report.
- CastcutFaceDistance cosine distance of each image's face to a reference (ComfyUI_FaceAnalysis'
                      ANALYSIS_MODELS; optional — needs that pack with insightface).
- CastcutMaskRepair   matte repair (src/lib/isolate-mask.ts) + the original pixels composited
                      onto a fill colour; tests/vectors/mask-repair.json keeps it in step.
- CastcutReport       writes the scores into the job's UI output (`castcut`) so the app reads them
                      from /history, and saves the alternate take next to the still.

Only numpy (and PIL for the alternate save) is needed beyond ComfyUI itself; torch, cv2,
folder_paths are imported lazily so the pure functions run (and are tested) without ComfyUI.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque

try:  # numpy ships with ComfyUI; the pose functions below don't need it.
    import numpy as np
except ImportError:  # pragma: no cover - only without numpy
    np = None

CASTCUT_VERSION = "1.1.0"

# Every node's object_info `description` ends with this marker, so the app can tell which version
# is installed without running anything (src/lib/castcut-nodes-setup.ts parses it). Keep the
# format. Bump CASTCUT_VERSION together with pyproject.toml and the app's
# CASTCUT_NODES_BUNDLED_VERSION whenever a node's inputs, outputs or results change.
VERSION_MARKER = f"[castcut-nodes {CASTCUT_VERSION}]"


def _describe(text):
    return f"{text} {VERSION_MARKER}"

# --------------------------------------------------------------------------------------------
# JS-compatible number helpers (the TS code runs on JS numbers)
# --------------------------------------------------------------------------------------------


def _js_number(value):
    """`Number(value)` for the JSON values a keypoint list can hold."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return 0.0
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0.0
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float("nan")


def _finite(value):
    return isinstance(value, float) and math.isfinite(value)


def _js_round(value):
    """`Math.round`: halves go up."""
    return math.floor(value + 0.5)


def _round2(value):
    return _js_round(value * 100) / 100


def _hypot(x, y):
    return math.hypot(x, y)


# --------------------------------------------------------------------------------------------
# pose-posture.ts
# --------------------------------------------------------------------------------------------

LYING_TILT_DEG = 50
SEATED_KNEE_DROP = 0.45
KNEELING_SHIN_DROP = 0.35

NOSE, NECK, R_SHOULDER, L_SHOULDER = 0, 1, 2, 5
R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE = 8, 9, 10, 11, 12, 13
R_EYE, L_EYE = 14, 15

INVERTED_TILT_DEG = 150
TILT_MARGIN_DEG = 10
FORESHORTENED_TORSO = 0.6
BEND_FEET_DROP = 0.9
BEND_LEG_FROM_VERTICAL_DEG = 35
ALL_FOURS_KNEE_DROP = 0.5
SIDE_SHOULDER_SHARE = 0.35
CROUCH_KNEE_ANGLE_DEG = 115
DROP_MARGIN = 0.12
FRAME_EDGE = 0.015

GROUP = {
    "standing": "upright",
    "bending": "bent",
    "sitting": "seated",
    "kneeling": "low",
    "crouching": "low",
    "all-fours": "low",
    "lying-back": "lying",
    "lying-side": "lying",
    "lying-front": "lying",
    "upside-down": "inverted",
    "unknown": "unknown",
}

POSTURE_WORD = {
    "standing": "standing",
    "sitting": "sitting",
    "kneeling": "kneeling",
    "crouching": "crouching",
    "lying-back": "lying on the back",
    "lying-side": "lying on the side",
    "lying-front": "lying on the front",
    "bending": "bending over",
    "all-fours": "on all fours",
    "upside-down": "upside down",
    "unknown": "unclear",
}


def _at(body, index):
    return body[index] if 0 <= index < len(body) else None


def without_edge_joints(body):
    return [
        point
        if point
        and FRAME_EDGE < point["x"] < 1 - FRAME_EDGE
        and FRAME_EDGE < point["y"] < 1 - FRAME_EDGE
        else None
        for point in body
    ]


def _mean(points):
    found = [p for p in points if p is not None]
    if not found:
        return None
    sx = 0.0
    sy = 0.0
    for p in found:
        sx += p["x"]
    for p in found:
        sy += p["y"]
    return {"x": sx / len(found), "y": sy / len(found)}


def _dist(a, b):
    return _hypot(a["x"] - b["x"], a["y"] - b["y"])


def _from_down_deg(a, b):
    length = _dist(a, b)
    if length < 1e-9:
        return 0.0
    return (math.acos(max(-1.0, min(1.0, (b["y"] - a["y"]) / length))) * 180) / math.pi


def _knee_angle_deg(hip, knee, ankle):
    a = {"x": hip["x"] - knee["x"], "y": hip["y"] - knee["y"]}
    b = {"x": ankle["x"] - knee["x"], "y": ankle["y"] - knee["y"]}
    lengths = _hypot(a["x"], a["y"]) * _hypot(b["x"], b["y"])
    if lengths == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (a["x"] * b["x"] + a["y"] * b["y"]) / lengths))
    return (math.acos(cos) * 180) / math.pi


def _scaled(body, aspect):
    def p(index):
        point = _at(body, index)
        return {"x": point["x"] * aspect, "y": point["y"]} if point else None

    return p


def body_reference_length(body, aspect):
    p = _scaled(body, aspect)
    neck = p(NECK)
    hip = _mean([p(R_HIP), p(L_HIP)])
    candidates = []
    if neck and hip:
        candidates.append(_dist(neck, hip))
    for h, k in ((R_HIP, R_KNEE), (L_HIP, L_KNEE)):
        a = p(h)
        b = p(k)
        if a and b:
            candidates.append(_dist(a, b) * 1.1)
    rs = p(R_SHOULDER)
    ls = p(L_SHOULDER)
    if rs and ls:
        candidates.append(_dist(rs, ls) * 1.4)
    return max(candidates) if candidates else 0.0


def _facing_of(body, aspect, ref):
    rs = _at(body, R_SHOULDER)
    ls = _at(body, L_SHOULDER)
    if not rs or not ls or ref <= 0:
        return "unknown"
    dx = (ls["x"] - rs["x"]) * aspect
    if abs(dx) < 0.35 * ref:
        return "unknown"
    face = bool(_at(body, NOSE) and (_at(body, R_EYE) or _at(body, L_EYE)))
    if dx > 0:
        return "camera" if face else "unknown"
    return "unknown" if face else "away"


def _read(posture, confident, facing, tilt_deg):
    return {
        "posture": posture,
        "group": GROUP[posture],
        "confident": posture != "unknown" and bool(confident),
        "facing": facing,
        "tiltDeg": None if tilt_deg is None else _js_round(tilt_deg),
    }


def _avg(values):
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def classify_posture(body, aspect):
    p = _scaled(body, aspect)
    ref = body_reference_length(body, aspect)
    facing = _facing_of(body, aspect, ref)
    neck = p(NECK) or p(NOSE)
    hip = _mean([p(R_HIP), p(L_HIP)])
    if not neck or not hip or ref <= 0:
        return _read("unknown", False, facing, None)
    torso = _dist(neck, hip)
    tilt = 180 - _from_down_deg(hip, neck)
    foreshortened = torso < FORESHORTENED_TORSO * ref
    legs = [
        {"hip": p(R_HIP) or hip, "knee": p(R_KNEE), "ankle": p(R_ANKLE)},
        {"hip": p(L_HIP) or hip, "knee": p(L_KNEE), "ankle": p(L_ANKLE)},
    ]
    with_knee = [leg for leg in legs if leg["knee"] is not None]
    full = [leg for leg in with_knee if leg["ankle"] is not None]

    if tilt >= INVERTED_TILT_DEG and not foreshortened:
        return _read("upside-down", tilt >= INVERTED_TILT_DEG + TILT_MARGIN_DEG, facing, tilt)

    if tilt > LYING_TILT_DEG and not foreshortened:
        confident_tilt = tilt > LYING_TILT_DEG + TILT_MARGIN_DEG
        if full:
            feet_drop = _avg([leg["ankle"]["y"] - hip["y"] for leg in full]) / ref
            leg_from_vertical = _avg([_from_down_deg(leg["hip"], leg["ankle"]) for leg in full])
            if feet_drop >= BEND_FEET_DROP and leg_from_vertical <= BEND_LEG_FROM_VERTICAL_DEG:
                return _read("bending", confident_tilt, facing, tilt)
            knee_drop = _avg([leg["knee"]["y"] - leg["hip"]["y"] for leg in full]) / ref
            shin_from_vertical = _avg([_from_down_deg(leg["knee"], leg["ankle"]) for leg in full])
            if knee_drop >= ALL_FOURS_KNEE_DROP and shin_from_vertical >= 55:
                return _read(
                    "all-fours",
                    confident_tilt and knee_drop >= ALL_FOURS_KNEE_DROP + DROP_MARGIN,
                    facing,
                    tilt,
                )
        return _read(_lying_kind(body, aspect, ref), confident_tilt, facing, tilt)

    if not with_knee:
        return _read("unknown" if foreshortened else "standing", False, facing, tilt)
    knee_drop = max(leg["knee"]["y"] - leg["hip"]["y"] for leg in with_knee) / ref
    support = None
    for leg in full:
        if support is None or leg["ankle"]["y"] > support["ankle"]["y"]:
            support = leg
    feet_drop = (support["ankle"]["y"] - hip["y"]) / ref if support else None
    if foreshortened:
        if feet_drop is not None and feet_drop >= 1.4 and knee_drop >= 0.6:
            return _read("standing", False, facing, tilt)
        return _read("unknown", False, facing, tilt)
    near_tilt = tilt > LYING_TILT_DEG - TILT_MARGIN_DEG
    if knee_drop < SEATED_KNEE_DROP:
        if support and feet_drop is not None:
            angle = _knee_angle_deg(support["hip"], support["knee"], support["ankle"])
            if 0.25 < feet_drop < 0.75 and angle < 80:
                return _read("crouching", False, facing, tilt)
        return _read(
            "sitting", (not near_tilt) and knee_drop < SEATED_KNEE_DROP - DROP_MARGIN, facing, tilt
        )
    if not support:
        return _read("bending" if tilt > 25 else "standing", False, facing, tilt)
    lowest_knee = max(leg["knee"]["y"] for leg in with_knee)
    shin_drop = (support["ankle"]["y"] - lowest_knee) / ref
    if shin_drop < KNEELING_SHIN_DROP:
        return _read(
            "kneeling",
            (not near_tilt) and shin_drop < KNEELING_SHIN_DROP - DROP_MARGIN,
            facing,
            tilt,
        )
    angle = _knee_angle_deg(support["hip"], support["knee"], support["ankle"])
    if angle < CROUCH_KNEE_ANGLE_DEG:
        return _read("crouching", angle < CROUCH_KNEE_ANGLE_DEG - 15, facing, tilt)
    confident_stand = (
        (not near_tilt)
        and knee_drop >= SEATED_KNEE_DROP + DROP_MARGIN
        and shin_drop >= KNEELING_SHIN_DROP + DROP_MARGIN
    )
    if tilt > 25:
        return _read("bending", False, facing, tilt)
    return _read("standing", confident_stand and tilt <= 20, facing, tilt)


def _lying_kind(body, aspect, ref):
    p = _scaled(body, aspect)
    rs = p(R_SHOULDER)
    ls = p(L_SHOULDER)
    neck = p(NECK)
    elbows = [point for point in (p(3), p(6)) if point is not None]
    shoulders = _mean([rs, ls])
    nose = p(NOSE)
    if (
        shoulders
        and neck
        and nose
        and len(elbows) == 2
        and all(elbow["y"] > shoulders["y"] + 0.2 * ref for elbow in elbows)
        and nose["y"] < neck["y"] - 0.1 * ref
    ):
        return "lying-front"
    if rs and ls and _dist(rs, ls) < SIDE_SHOULDER_SHARE * ref:
        return "lying-side"
    return "lying-back"


COMPATIBLE_GROUPS = {"low|seated"}


def posture_groups_agree(a, b):
    return a == b or "|".join(sorted([a, b])) in COMPATIBLE_GROUPS


def posture_mismatch(guide, still):
    if not guide["confident"] or not still["confident"]:
        return False
    return not posture_groups_agree(guide["group"], still["group"])


def posture_unsure(guide, still):
    if not guide["confident"] or still["confident"] or guide["group"] == "unknown":
        return False
    return not posture_groups_agree(guide["group"], still["group"])


# --------------------------------------------------------------------------------------------
# pose-limb-score.ts
# --------------------------------------------------------------------------------------------

SEGMENTS = (
    {"part": "torso", "from": "hip", "to": 1, "weight": 3},
    {"part": "shoulders", "from": 2, "to": 5, "weight": 1},
    {"part": "head", "from": 1, "to": 0, "weight": 0.5},
    {"part": "right upper arm", "from": 2, "to": 3, "weight": 1},
    {"part": "right forearm", "from": 3, "to": 4, "weight": 0.75},
    {"part": "left upper arm", "from": 5, "to": 6, "weight": 1},
    {"part": "left forearm", "from": 6, "to": 7, "weight": 0.75},
    {"part": "right thigh", "from": 8, "to": 9, "weight": 2},
    {"part": "right shin", "from": 9, "to": 10, "weight": 1.5},
    {"part": "left thigh", "from": 11, "to": 12, "weight": 2},
    {"part": "left shin", "from": 12, "to": 13, "weight": 1.5},
)

SWAP = (0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16)

LIMB_SIGMA_DEG = 40
LIMB_OFF_DEG = 45
FORESHORTENED = 0.22
NOT_FORESHORTENED = 0.45
FORESHORTEN_DELTA_DEG = 60
MIRROR_PENALTY = 0.25
MIN_SEGMENTS = 4
DEFINING_FLOOR = 0.3
DEFINING_FULL_DEG = 60
DEFINING_MISS_SHARE = 0.9
DEFINING_MISS_DEG = 110
GESTURE_MISS_COUNT = 3


def _segment_vectors(body, aspect):
    """Ordered list of (part, vec) — the TS Map in insertion (SEGMENTS) order."""
    p = _scaled(body, aspect)
    hips = [point for point in (p(8), p(11)) if point is not None]
    hip = None
    if hips:
        sx = 0.0
        sy = 0.0
        for point in hips:
            sx += point["x"]
        for point in hips:
            sy += point["y"]
        hip = {"x": sx / len(hips), "y": sy / len(hips)}
    out = {}
    for segment in SEGMENTS:
        a = hip if segment["from"] == "hip" else p(segment["from"])
        b = p(segment["to"])
        if not a or not b:
            continue
        x = b["x"] - a["x"]
        y = b["y"] - a["y"]
        out[segment["part"]] = {"x": x, "y": y, "len": _hypot(x, y)}
    return out


def _angle_between_deg(a, b):
    la = _hypot(a["x"], a["y"])
    lb = _hypot(b["x"], b["y"])
    if la < 1e-9 or lb < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, (a["x"] * b["x"] + a["y"] * b["y"]) / (la * lb)))
    return (math.acos(cos) * 180) / math.pi


def _relabel(body, mapping):
    length = max(18, len(body))
    out = []
    for i in range(length):
        source = mapping[i] if i < len(mapping) else i
        out.append(_at(body, source))
    return out


def _reflect(body):
    return [{"x": 1 - point["x"], "y": point["y"]} if point else None for point in body]


def _compare(guide, guide_ref, still, still_ref, weight_of=None):
    limbs = []
    for segment in SEGMENTS:
        g = guide.get(segment["part"])
        s = still.get(segment["part"])
        if not g or not s:
            continue
        weight = weight_of(segment) if weight_of else segment["weight"]
        g_share = g["len"] / guide_ref
        s_share = s["len"] / still_ref
        g_short = g_share < FORESHORTENED
        s_short = s_share < FORESHORTENED
        if g_short and s_short:
            continue
        if (g_short and s_share > NOT_FORESHORTENED) or (s_short and g_share > NOT_FORESHORTENED):
            delta = FORESHORTEN_DELTA_DEG
        else:
            delta = _angle_between_deg(g, s)
        limbs.append({"part": segment["part"], "deltaDeg": _js_round(delta), "weight": weight})
    if len(limbs) < MIN_SEGMENTS or not any(limb["part"] == "torso" for limb in limbs):
        return None
    total = 0.0
    weights = 0.0
    for limb in limbs:
        total += limb["weight"] * math.exp(-((limb["deltaDeg"] / LIMB_SIGMA_DEG) ** 2))
        weights += limb["weight"]
    return {"score": total / weights, "limbs": limbs}


def guide_asymmetry(guide, aspect):
    ref = body_reference_length(guide, aspect)
    if ref <= 0:
        return 1.0
    own = _segment_vectors(guide, aspect)
    mirror = _segment_vectors(_relabel(_reflect(guide), SWAP), aspect)
    torso = own.get("torso")
    mirror_torso = mirror.get("torso")
    if not torso or not mirror_torso:
        return 1.0

    def rotate(v, frm, to):
        angle = math.atan2(to["y"], to["x"]) - math.atan2(frm["y"], frm["x"])
        cos = math.cos(angle)
        sin = math.sin(angle)
        return {"x": v["x"] * cos - v["y"] * sin, "y": v["x"] * sin + v["y"] * cos, "len": v["len"]}

    aligned = {part: rotate(v, mirror_torso, torso) for part, v in mirror.items()}
    limb_parts = {}
    limb_aligned = {}
    for part, v in own.items():
        if "arm" in part or "thigh" in part or "shin" in part or part == "torso":
            limb_parts[part] = v
            m = aligned.get(part)
            if m:
                limb_aligned[part] = m
    self_match = _compare(limb_parts, ref, limb_aligned, ref)
    return max(0.0, min(1.0, 1 - self_match["score"])) if self_match else 1.0


def _is_posture(part):
    return part == "torso" or "thigh" in part or "shin" in part


def _defining_shares(guide):
    out = {}
    torso = guide.get("torso")
    if not torso:
        return out

    def share(deg):
        return max(0.0, min(1.0, deg / DEFINING_FULL_DEG))

    out["torso"] = share(_angle_between_deg(torso, {"x": 0, "y": -1}))
    down = {"x": -torso["x"], "y": -torso["y"]}
    for part, v in guide.items():
        if part == "torso":
            continue
        if part == "shoulders":
            out[part] = 0.5
        elif part == "head":
            out[part] = share(_angle_between_deg(v, torso))
        else:
            out[part] = share(_angle_between_deg(v, down))
    return out


def score_limb_angles(guide, detected, aspects):
    guide_ref = body_reference_length(guide, aspects["guide"])
    if guide_ref <= 0:
        return None
    guide_vectors = _segment_vectors(guide, aspects["guide"])
    asymmetry = guide_asymmetry(guide, aspects["guide"])
    defining = _defining_shares(guide_vectors)

    def weight_of(segment):
        if _is_posture(segment["part"]):
            return segment["weight"]
        return segment["weight"] * (
            DEFINING_FLOOR + (1 - DEFINING_FLOOR) * defining.get(segment["part"], 0)
        )

    best = None
    for mirrored in (False, True):
        for swapped in (False, True):
            body = _reflect(detected) if mirrored else detected
            if swapped:
                body = _relabel(body, SWAP)
            ref = body_reference_length(body, aspects["detected"])
            if ref <= 0:
                continue
            result = _compare(
                guide_vectors, guide_ref, _segment_vectors(body, aspects["detected"]), ref, weight_of
            )
            if not result:
                continue
            rank = result["score"] - (MIRROR_PENALTY * asymmetry if mirrored else 0)
            if best is None or rank > best["rank"]:
                off = [limb for limb in result["limbs"] if limb["deltaDeg"] >= LIMB_OFF_DEG]
                off.sort(key=lambda limb: -(limb["deltaDeg"] * limb["weight"]))
                best = {
                    "rank": rank,
                    "score": max(0.0, rank),
                    "limbs": result["limbs"],
                    "off": [limb["part"] for limb in off],
                    "mirrored": mirrored,
                }
    if best is None:
        return None
    limbs = [
        {**limb, "defining": _js_round(defining.get(limb["part"], 0) * 100) / 100}
        for limb in best["limbs"]
    ]
    defining_off = [
        limb
        for limb in limbs
        if limb["defining"] >= DEFINING_MISS_SHARE and limb["deltaDeg"] >= DEFINING_MISS_DEG
    ]
    defining_off.sort(key=lambda limb: -limb["deltaDeg"])
    return {
        "score": best["score"],
        "limbs": limbs,
        "off": best["off"],
        "mirrored": best["mirrored"],
        "definingOff": [limb["part"] for limb in defining_off],
        "gestureMiss": len(defining_off) >= GESTURE_MISS_COUNT,
    }


# --------------------------------------------------------------------------------------------
# pose-score.ts
# --------------------------------------------------------------------------------------------

SCORED_JOINTS = tuple(range(14))
SCORED_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13))
JOINT_SIGMA = 0.35
MIRROR_INDEX = (0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16)
MIN_SHARED_LIMBS = 4
MIN_SHARED_JOINTS = 6
HEADCOUNT_MIN_LIMBS = 2
HEADCOUNT_MIN_PROMINENCE = 0.3
LIMB_ANGLE_MIN_POSE_MATCH = 0.5
JOINT_DISTANCE_MIN_POSE_MATCH = 0.6
POSE_MISS_SCORE_CAP = _js_round(LIMB_ANGLE_MIN_POSE_MATCH * 0.8 * 100) / 100


def _parse_body(raw, width, height, normalized):
    if not isinstance(raw, list) or len(raw) < 18 * 3:
        return None
    body = []
    for i in range(18):
        x = _js_number(raw[i * 3])
        y = _js_number(raw[i * 3 + 1])
        c = _js_number(raw[i * 3 + 2])
        # `c <= 0` as in TS: a NaN confidence keeps the joint there too.
        if not _finite(x) or not _finite(y) or c <= 0 or x < 0 or y < 0 or (x == 0 and y == 0):
            body.append(None)
            continue
        body.append({"x": x, "y": y} if normalized else {"x": x / width, "y": y / height})
    return body if any(body) else None


def parse_openpose_json(raw):
    """controlnet_aux `openpose_json` / POSE_KEYPOINT frame -> {canvas, people} or None."""
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None
    width = _js_number(value.get("canvas_width"))
    height = _js_number(value.get("canvas_height"))
    width = width if _finite(width) and width != 0 else 0.0
    height = height if _finite(height) and height != 0 else 0.0
    rows = value.get("people") if isinstance(value.get("people"), list) else []
    coords = []
    for row in rows:
        points = row.get("pose_keypoints_2d") if isinstance(row, dict) else None
        if isinstance(points, list):
            coords.extend(points)
    xy = [_js_number(v) for i, v in enumerate(coords) if i % 3 != 2]
    xy = [n for n in xy if _finite(n)]
    normalized = len(xy) > 0 and all(n <= 1.5 for n in xy)
    if not normalized and not (width > 0 and height > 0):
        return None
    people = []
    for row in rows:
        points = row.get("pose_keypoints_2d") if isinstance(row, dict) else None
        body = _parse_body(points, width, height, normalized)
        if body:
            people.append(body)
    return {"canvas": {"width": width, "height": height}, "people": people}


def score_body_match(guide, detected, aspects):
    neck = _at(guide, 1)
    hips = [point for point in (_at(guide, 8), _at(guide, 11)) if point]
    if not neck or not hips:
        return None
    sx = 0.0
    sy = 0.0
    for point in hips:
        sx += point["x"]
    for point in hips:
        sy += point["y"]
    mid_hip = {"x": sx / len(hips), "y": sy / len(hips)}
    torso = _hypot((mid_hip["x"] - neck["x"]) * aspects["guide"], mid_hip["y"] - neck["y"])
    if torso < 1e-4:
        return None

    def score_with(mapping):
        pairs = []
        for index in SCORED_JOINTS:
            g = _at(guide, index)
            d = _at(detected, mapping[index])
            if g and d:
                pairs.append(
                    (
                        {"x": g["x"] * aspects["guide"], "y": g["y"]},
                        {"x": d["x"] * aspects["detected"], "y": d["y"]},
                    )
                )
        if len(pairs) < MIN_SHARED_JOINTS:
            return None

        def mean(points):
            tx = 0.0
            ty = 0.0
            for point in points:
                tx += point["x"]
            for point in points:
                ty += point["y"]
            return {"x": tx / len(points), "y": ty / len(points)}

        cg = mean([pair[0] for pair in pairs])
        cd = mean([pair[1] for pair in pairs])
        cross = 0.0
        norm = 0.0
        for g, d in pairs:
            cross += (g["x"] - cg["x"]) * (d["x"] - cd["x"]) + (g["y"] - cg["y"]) * (d["y"] - cd["y"])
            norm += (d["x"] - cd["x"]) ** 2 + (d["y"] - cd["y"]) ** 2
        scale = max(0.0, cross / norm) if norm > 1e-9 else 1.0
        total = 0.0
        for g, d in pairs:
            dx = g["x"] - cg["x"] - scale * (d["x"] - cd["x"])
            dy = g["y"] - cg["y"] - scale * (d["y"] - cd["y"])
            distance = _hypot(dx, dy) / torso
            total += math.exp(-((distance / JOINT_SIGMA) ** 2))
        return total / len(pairs)

    straight = score_with(tuple(range(18)))
    swapped = score_with(MIRROR_INDEX)
    if straight is None:
        return swapped
    if swapped is None:
        return straight
    return max(straight, swapped)


def _permutations(items, size):
    if size == 0:
        return [[]]
    out = []
    for index, item in enumerate(items):
        rest = items[:index] + items[index + 1 :]
        for tail in _permutations(rest, size - 1):
            out.append([item] + tail)
    return out


def _limb_count(body):
    return sum(1 for a, b in SCORED_LIMBS if _at(body, a) and _at(body, b))


def _prominence(body, aspect):
    points = [point for point in body if point]
    if len(points) < 3:
        return 0.0
    xs = [point["x"] * aspect for point in points]
    ys = [point["y"] for point in points]
    diagonal = _hypot(max(xs) - min(xs), max(ys) - min(ys))
    return diagonal if _limb_count(body) >= 2 else diagonal * 0.25


def count_prominent_people(detected):
    canvas = detected["canvas"]
    aspect = canvas["width"] / canvas["height"] if canvas["width"] > 0 and canvas["height"] > 0 else 1
    return sum(
        1
        for body in detected["people"]
        if _limb_count(body) >= HEADCOUNT_MIN_LIMBS
        and _prominence(body, aspect) >= HEADCOUNT_MIN_PROMINENCE
    )


def limb_verdict_score(limb_score, miss):
    limb = max(0.0, min(1.0, limb_score))
    if miss:
        return limb * POSE_MISS_SCORE_CAP
    return LIMB_ANGLE_MIN_POSE_MATCH + (1 - LIMB_ANGLE_MIN_POSE_MATCH) * limb


def score_pose_match(guide_people, guide_aspect, detected, method="limb-angle"):
    """Port of `scorePoseMatch`: the same PoseMatchResult shape (camelCase keys)."""
    canvas = detected["canvas"]
    detected_aspect = (
        canvas["width"] / canvas["height"]
        if canvas["width"] > 0 and canvas["height"] > 0
        else guide_aspect
    )
    aspects = {"guide": guide_aspect, "detected": detected_aspect}
    guide = list(guide_people[:3])
    candidates = [
        {"body": body, "index": index, "size": _prominence(body, detected_aspect)}
        for index, body in enumerate(detected["people"])
    ]
    candidates.sort(key=lambda c: -c["size"])
    candidates = candidates[: max(1, min(3, len(guide_people)))]
    joint = [[score_body_match(g, c["body"], aspects) for c in candidates] for g in guide]
    seen = [without_edge_joints(c["body"]) for c in candidates]
    limb = [[score_limb_angles(g, body, aspects) for body in seen] for g in guide]
    if method == "limb-angle":
        matrix = [[(match["score"] if match else None) for match in row] for row in limb]
    else:
        matrix = joint
    slots = min(len(guide), len(candidates))
    best_total = -1.0
    best_picks = []
    for guide_order in _permutations(list(range(len(guide))), slots):
        for cand_order in _permutations(list(range(len(candidates))), slots):
            total = 0
            picks = [-1 for _ in guide]
            for i, g in enumerate(guide_order):
                c = cand_order[i]
                value = matrix[g][c]
                total += value if value is not None else 0
                picks[g] = c
            if total > best_total:
                best_total = total
                best_picks = picks

    def pick(g):
        return best_picks[g] if g < len(best_picks) else -1

    def mean_over(values):
        if not guide:
            return 0.0
        total = 0.0
        for value in values:
            total += value if value is not None else 0
        return total / len(guide)

    joint_per = [joint[g][pick(g)] if pick(g) >= 0 else None for g in range(len(guide))]
    limb_per = [
        (limb[g][pick(g)]["score"] if limb[g][pick(g)] else None) if pick(g) >= 0 else None
        for g in range(len(guide))
    ]
    posture = []
    for g, body in enumerate(guide):
        guide_read = classify_posture(body, guide_aspect)
        c = pick(g)
        still_read = classify_posture(seen[c], detected_aspect) if c >= 0 else None
        posture.append(
            {
                "guide": guide_read,
                "still": still_read,
                "mismatch": posture_mismatch(guide_read, still_read) if still_read else False,
                "unsure": posture_unsure(guide_read, still_read) if still_read else False,
            }
        )
    posture_miss = any(pair["mismatch"] for pair in posture)
    gesture_miss = any(
        pick(g) >= 0 and bool(limb[g][pick(g)]) and limb[g][pick(g)]["gestureMiss"] is True
        for g in range(len(guide))
    )
    joint_score = mean_over(joint_per)
    read_limbs = [value for value in limb_per if value is not None]
    limb_score = 0.0
    if read_limbs:
        total = 0.0
        for value in read_limbs:
            total += value
        limb_score = total / len(read_limbs)
    anyone = any(pick(g) >= 0 for g in range(len(guide)))
    if method != "limb-angle":
        score = joint_score
    elif read_limbs:
        score = limb_verdict_score(limb_score, posture_miss)
    elif anyone and not posture_miss:
        score = LIMB_ANGLE_MIN_POSE_MATCH
    else:
        score = 0.0
    lead = limb[0][pick(0)] if guide and pick(0) >= 0 else None
    return {
        "score": _round2(score),
        "method": method,
        "jointScore": _round2(joint_score),
        "limbScore": _round2(limb_score),
        "gestureMiss": gesture_miss,
        "expectedPeople": len(guide),
        "detectedPeople": sum(
            1 for body in detected["people"] if _limb_count(body) >= MIN_SHARED_LIMBS
        ),
        "extraPeople": max(0, count_prominent_people(detected) - len(guide_people)),
        "perPerson": limb_per if method == "limb-angle" else joint_per,
        "assignment": [
            candidates[pick(g)]["index"] if pick(g) >= 0 else -1 for g in range(len(guide))
        ],
        "posture": posture,
        "postureMiss": posture_miss,
        "postureUnsure": (not posture_miss) and any(pair["unsure"] for pair in posture),
        "offLimbs": lead["off"] if lead else [],
        "limbDeltas": lead["limbs"] if lead else [],
    }


def parse_guide(raw):
    """Guide JSON from the app: {"guide": [[{x,y}|null ×18], …], "aspect": w/h}."""
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("Castcut guide must be a JSON object with guide and aspect.")
    people = value.get("guide")
    aspect = _js_number(value.get("aspect"))
    if not isinstance(people, list) or not _finite(aspect) or aspect <= 0:
        raise ValueError("Castcut guide needs a guide list and a positive aspect.")
    guide = []
    for body in people:
        if not isinstance(body, list):
            continue
        guide.append(
            [
                {"x": float(point["x"]), "y": float(point["y"])}
                if isinstance(point, dict) and "x" in point and "y" in point
                else None
                for point in body
            ]
        )
    return guide, aspect


# --------------------------------------------------------------------------------------------
# Best of two: day-best-of-two.ts pickBetterTake, generalised to a batch
# --------------------------------------------------------------------------------------------


def pick_best_index(scores):
    """Highest score wins; on a tie the later take (the TS rule: the second unless the first
    is strictly closer). A take without a score loses to one with a score."""
    best = None
    for index, score in enumerate(scores):
        usable = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
        if best is None:
            best = (index, score if usable else None)
            continue
        _, best_score = best
        if not usable:
            if best_score is None:
                best = (index, None)
            continue
        if best_score is None or score >= best_score:
            best = (index, score)
    return best[0] if best else 0


# --------------------------------------------------------------------------------------------
# isolate-mask.ts
# --------------------------------------------------------------------------------------------

SOLID_ALPHA = 200
SUBJECT_ALPHA = 128
SURE_BACKGROUND_ALPHA = 32
HOLE_NEAR_BACKDROP = 48
PLAIN_BACKDROP_SPREAD = 24
PLAIN_BACKDROP_SHARE = 0.8
FAR_FROM_BACKDROP = 96
RIM_RADIUS = 3


def _require_numpy():
    if np is None:
        raise RuntimeError("Castcut mask repair needs numpy (ComfyUI ships it).")


def _label_components(member):
    """4-connected components; labels (0 = not a member), count. cv2 when present, else BFS."""
    member = member.astype(np.uint8)
    try:
        import cv2  # noqa: PLC0415 - optional, fast path

        count, labels = cv2.connectedComponents(member, connectivity=4, ltype=cv2.CV_32S)
        return labels, int(count) - 1
    except ImportError:
        pass
    height, width = member.shape
    labels = np.zeros((height, width), dtype=np.int32)
    flat = member.reshape(-1)
    out = labels.reshape(-1)
    count = 0
    for start in np.flatnonzero(flat):
        if out[start]:
            continue
        count += 1
        out[start] = count
        queue = deque([int(start)])
        while queue:
            p = queue.popleft()
            x = p % width
            for n in (
                p - 1 if x > 0 else -1,
                p + 1 if x < width - 1 else -1,
                p - width if p >= width else -1,
                p + width if p < width * (height - 1) else -1,
            ):
                if n >= 0 and flat[n] and not out[n]:
                    out[n] = count
                    queue.append(n)
    return labels, count


def _dilate(member, radius):
    """Square dilation by `radius` (the TS separable max filter)."""
    height, width = member.shape
    rows = np.zeros_like(member, dtype=bool)
    src = member.astype(bool)
    for dx in range(-radius, radius + 1):
        if dx >= 0:
            rows[:, : width - dx] |= src[:, dx:]
        else:
            rows[:, -dx:] |= src[:, : width + dx]
    out = np.zeros_like(rows)
    for dy in range(-radius, radius + 1):
        if dy >= 0:
            out[: height - dy, :] |= rows[dy:, :]
        else:
            out[-dy:, :] |= rows[: height + dy, :]
    return out


def _color_distance(rgb, color):
    diff = rgb.astype(np.float64) - np.array([color["r"], color["g"], color["b"]], dtype=np.float64)
    return np.sqrt(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1] + diff[..., 2] * diff[..., 2])


def repair_subject_mask(alpha_in, rgb, regrow_edges=True):
    """Port of `repairSubjectMask`. alpha_in: uint8 [H,W]; rgb: uint8 [H,W,3+]."""
    _require_numpy()
    alpha_in = np.asarray(alpha_in, dtype=np.uint8)
    rgb = np.asarray(rgb, dtype=np.uint8)[..., :3]
    height, width = alpha_in.shape
    pixels = width * height
    alpha = np.where(alpha_in >= SOLID_ALPHA, 255, alpha_in).astype(np.uint8)
    empty = {
        "alpha": alpha,
        "filledHolePixels": 0,
        "regrownPixels": 0,
        "backdrop": None,
        "plainBackdrop": False,
    }
    if pixels < 16:
        return empty
    background = alpha < SUBJECT_ALPHA
    labels, count = _label_components(background)
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    outside_ids = np.unique(border[border > 0])
    is_outside = np.zeros(count + 1, dtype=bool)
    is_outside[outside_ids] = True
    outside = is_outside[labels]

    sure = (alpha < SURE_BACKGROUND_ALPHA) & outside
    samples = int(sure.sum())
    if samples == 0:
        return empty

    def median(channel):
        histogram = np.bincount(rgb[..., channel][sure], minlength=256)
        seen = 0
        for value in range(256):
            seen += int(histogram[value])
            if seen * 2 >= samples:
                return value
        return 255

    backdrop = {"r": median(0), "g": median(1), "b": median(2)}
    distance = _color_distance(rgb, backdrop)
    close = int(((distance < PLAIN_BACKDROP_SPREAD) & sure).sum())
    plain_backdrop = close / samples >= PLAIN_BACKDROP_SHARE

    # 1. Enclosed holes.
    hole = (labels > 0) & ~outside
    hole_size = np.bincount(labels[hole], minlength=count + 1)
    hole_near = np.bincount(labels[hole & (distance < HOLE_NEAR_BACKDROP)], minlength=count + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill_label = np.where(hole_size > 0, hole_near / np.maximum(hole_size, 1) < 0.5, False)
    fill = hole & fill_label[labels]
    alpha[fill] = 255
    filled_hole_pixels = int(fill.sum())

    # 2. Plain backdrop: regrow person-coloured "background" that touches the person.
    regrown_pixels = 0
    if plain_backdrop and regrow_edges:
        far = (alpha < 255) & (distance > FAR_FROM_BACKDROP)
        far_labels, far_count = _label_components(far)
        anchor = (alpha >= SUBJECT_ALPHA) & ~far
        neighbour = np.zeros_like(far)
        neighbour[:, 1:] |= anchor[:, :-1]
        neighbour[:, :-1] |= anchor[:, 1:]
        neighbour[1:, :] |= anchor[:-1, :]
        neighbour[:-1, :] |= anchor[1:, :]
        touching_ids = np.unique(far_labels[far & neighbour])
        touches = np.zeros(far_count + 1, dtype=bool)
        touches[touching_ids[touching_ids > 0]] = True
        regrow = (far_labels > 0) & touches[far_labels]
        alpha[regrow] = 255
        regrown_pixels = int(regrow.sum())

    if filled_hole_pixels + regrown_pixels > 0:
        restored = (alpha == 255) & (alpha_in < SUBJECT_ALPHA)
        near = _dilate(restored, RIM_RADIUS)
        firm = near & (alpha < 255) & (distance >= HOLE_NEAR_BACKDROP)
        alpha[firm] = 255

    return {
        "alpha": alpha,
        "filledHolePixels": filled_hole_pixels,
        "regrownPixels": regrown_pixels,
        "backdrop": backdrop,
        "plainBackdrop": bool(plain_backdrop),
    }


def composite_through_mask(rgba, alpha, fill):
    """Port of `compositeThroughMask`: uint8 RGBA [H,W,4] (or RGB) -> uint8 RGBA."""
    _require_numpy()
    rgba = np.asarray(rgba, dtype=np.uint8)
    height, width = rgba.shape[:2]
    source_alpha = (
        rgba[..., 3].astype(np.float64) if rgba.shape[-1] >= 4 else np.full((height, width), 255.0)
    )
    a = (np.asarray(alpha, dtype=np.float64) / 255) * (source_alpha / 255)
    inv = 1 - a
    out = np.empty((height, width, 4), dtype=np.uint8)
    for channel, key in enumerate(("r", "g", "b")):
        value = rgba[..., channel].astype(np.float64) * a + fill[key] * inv
        out[..., channel] = np.clip(np.floor(value + 0.5), 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def mask_subject_share(alpha):
    _require_numpy()
    alpha = np.asarray(alpha)
    if alpha.size == 0:
        return 0.0
    return float((alpha >= SUBJECT_ALPHA).sum()) / alpha.size


def cutout_looks_isolated(alpha):
    """isolate-subject.ts `cutoutLooksIsolated` on the mask alone."""
    _require_numpy()
    alpha = np.asarray(alpha)
    pixels = alpha.size
    if pixels < 16:
        return False
    background = int((alpha < 48).sum())
    subject = int((alpha > 160).sum())
    return background >= pixels * 0.02 and subject >= pixels * 0.02


def parse_fill(text):
    value = (text or "").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Castcut fill must be #rrggbb, got {text!r}.")
    return {"r": int(value[0:2], 16), "g": int(value[2:4], 16), "b": int(value[4:6], 16)}


# --------------------------------------------------------------------------------------------
# Tensor plumbing (torch only inside ComfyUI; plain numpy arrays work for tests)
# --------------------------------------------------------------------------------------------


def _to_numpy(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _like(source, array):
    """Return `array` as the same kind of container as `source` (torch tensor or numpy)."""
    if hasattr(source, "detach"):
        import torch  # noqa: PLC0415 - only inside ComfyUI

        return torch.from_numpy(np.ascontiguousarray(array))
    return array


def _image_uint8(image):
    """ComfyUI IMAGE float [B,H,W,C] in 0-1 -> uint8 (exact for 8-bit sources)."""
    return np.clip(np.floor(_to_numpy(image).astype(np.float64) * 255 + 0.5), 0, 255).astype(np.uint8)


def _mask_uint8(mask):
    """MASK float [B,H,W] -> uint8 the way SaveImage writes it (what the app's matte read sees)."""
    values = _to_numpy(mask).astype(np.float32)
    if values.ndim == 2:
        values = values[None]
    return np.clip(np.float32(255.0) * values, 0, 255).astype(np.uint8)


def _resize_mask(mask, height, width):
    if mask.shape == (height, width):
        return mask
    try:
        import cv2  # noqa: PLC0415

        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        ys = (np.arange(height) * mask.shape[0] / height).astype(int)
        xs = (np.arange(width) * mask.shape[1] / width).astype(int)
        return mask[ys][:, xs]


def _pose_frames(pose_keypoint):
    if isinstance(pose_keypoint, str):
        try:
            pose_keypoint = json.loads(pose_keypoint)
        except ValueError:
            return []
    if isinstance(pose_keypoint, dict):
        return [pose_keypoint]
    if isinstance(pose_keypoint, list):
        return pose_keypoint
    return []


# --------------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------------

CATEGORY = "Castcut"


class CastcutPoseScore:
    """DWPose keypoints of each image in a batch scored against the pose guide."""

    DESCRIPTION = _describe(
        "Scores each image's DWPose keypoints against Castcut's pose guide (limb angles + posture)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "guide_json": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "method": (["limb-angle", "joint-distance"], {"default": "limb-angle"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("scores_json", "first_score")
    FUNCTION = "score"
    CATEGORY = CATEGORY

    def score(self, pose_keypoint, guide_json, method="limb-angle"):
        guide, aspect = parse_guide(guide_json)
        results = []
        for frame in _pose_frames(pose_keypoint):
            detected = parse_openpose_json(frame)
            if detected is None:
                results.append(None)
                continue
            results.append(score_pose_match(guide, aspect, detected, method))
        first = next((r["score"] for r in results if r), 0.0)
        return (json.dumps({"version": CASTCUT_VERSION, "results": results}), float(first))


class CastcutPickBest:
    """Keep the take whose score is highest (ties: the later take)."""

    DESCRIPTION = _describe(
        "Keeps the take with the higher pose score; the other take goes to Castcut Report."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scores_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("best", "other", "best_index", "report_json")
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, images, scores_json):
        parsed = json.loads(scores_json) if isinstance(scores_json, str) else scores_json
        results = parsed.get("results", []) if isinstance(parsed, dict) else parsed
        count = int(images.shape[0])
        scores = []
        for index in range(count):
            entry = results[index] if index < len(results) else None
            if isinstance(entry, dict):
                scores.append(entry.get("score"))
            elif isinstance(entry, (int, float)):
                scores.append(entry)
            else:
                scores.append(None)
        best = pick_best_index(scores) if count > 0 else 0
        other = 1 - best if count == 2 else (0 if best != 0 else min(1, count - 1))
        report = {
            "version": CASTCUT_VERSION,
            "kind": "pick-best",
            "bestIndex": best,
            "otherIndex": other,
            "scores": scores,
            "results": results,
        }
        return (images[best : best + 1], images[other : other + 1], best, json.dumps(report))


class CastcutFaceDistance:
    """Cosine distance of each image's largest face to the reference (100 = no face)."""

    DESCRIPTION = _describe(
        "Cosine distance of each image's largest face to a reference (needs ComfyUI_FaceAnalysis)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS",),
                "reference": ("IMAGE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("distances_json", "first_distance")
    FUNCTION = "measure"
    CATEGORY = CATEGORY

    def measure(self, analysis_models, reference, image):
        if not hasattr(analysis_models, "get_embeds"):
            raise RuntimeError(
                "CastcutFaceDistance needs ComfyUI_FaceAnalysis (FaceAnalysisModels) with insightface."
            )

        def embeds(batch):
            out = []
            for frame in np.clip(_to_numpy(batch) * 255, 0, 255).astype(np.uint8):
                out.append(analysis_models.get_embeds(frame[..., :3]))
            return out

        refs = [e for e in embeds(reference) if e is not None]
        if not refs:
            raise RuntimeError("No face detected in the reference image.")
        ref = np.mean(np.stack(refs), axis=0)
        distances = []
        for emb in embeds(image):
            if emb is None:
                distances.append(100.0)
            elif np.array_equal(ref, emb):
                distances.append(0.0)
            else:
                distances.append(
                    float(1 - np.dot(ref, emb) / (np.linalg.norm(ref) * np.linalg.norm(emb)))
                )
        payload = {"version": CASTCUT_VERSION, "metric": "cosine", "distances": distances}
        return (json.dumps(payload), float(distances[0]) if distances else 100.0)


class CastcutMaskRepair:
    """Repair a matte from the photo's colours and composite the original pixels onto a fill."""

    DESCRIPTION = _describe(
        "Repairs a background-removal matte from the photo's colours and composites onto a fill."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill": ("STRING", {"default": "#ffffff"}),
                "regrow_edges": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # LoadImage's MASK (1 = transparent): a photo with its own transparency keeps it.
                "source_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("composite", "mask", "report_json")
    FUNCTION = "repair"
    CATEGORY = CATEGORY

    def repair(self, image, mask, fill="#ffffff", regrow_edges=False, source_mask=None):
        fill_rgb = parse_fill(fill)
        pixels = _image_uint8(image)
        masks = _mask_uint8(mask)
        sources = _to_numpy(source_mask).astype(np.float64) if source_mask is not None else None
        if sources is not None and sources.ndim == 2:
            sources = sources[None]
        composites = []
        repaired_masks = []
        reports = []
        for index in range(pixels.shape[0]):
            rgb = pixels[index][..., :3]
            height, width = rgb.shape[:2]
            alpha_in = _resize_mask(masks[min(index, masks.shape[0] - 1)], height, width)
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = rgb
            if sources is not None and sources[min(index, sources.shape[0] - 1)].shape == (height, width):
                transparent = sources[min(index, sources.shape[0] - 1)]
                rgba[..., 3] = np.clip(np.floor((1 - transparent) * 255 + 0.5), 0, 255).astype(np.uint8)
            else:
                rgba[..., 3] = 255
            result = repair_subject_mask(alpha_in, rgb, regrow_edges=bool(regrow_edges))
            composites.append(composite_through_mask(rgba, result["alpha"], fill_rgb)[..., :3])
            repaired_masks.append(result["alpha"])
            reports.append(
                {
                    "filledHolePixels": result["filledHolePixels"],
                    "regrownPixels": result["regrownPixels"],
                    "backdrop": result["backdrop"],
                    "plainBackdrop": result["plainBackdrop"],
                    "subjectShare": round(mask_subject_share(result["alpha"]), 4),
                    "looksIsolated": cutout_looks_isolated(result["alpha"]),
                }
            )
        composite = np.stack(composites).astype(np.float32) / np.float32(255)
        repaired = np.stack(repaired_masks).astype(np.float32) / np.float32(255)
        report = {"version": CASTCUT_VERSION, "kind": "mask-repair", "fill": fill_rgb, "images": reports}
        return (_like(image, composite), _like(mask, repaired), json.dumps(report))


class CastcutReport:
    """Put the scores in the job's history (`outputs[id].castcut`) and save the other take."""

    DESCRIPTION = _describe(
        "Writes Castcut's check results into the job's history and saves the alternate take."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report_json": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "alternate": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "castcut-alt"}),
                "extra_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def report(
        self,
        report_json,
        alternate=None,
        filename_prefix="castcut-alt",
        extra_json=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        try:
            report = json.loads(report_json) if isinstance(report_json, str) else dict(report_json)
        except ValueError:
            report = {"raw": str(report_json)}
        if extra_json:
            try:
                report["extra"] = json.loads(extra_json)
            except ValueError:
                report["extra"] = extra_json
        if alternate is not None:
            report["alternates"] = save_images(alternate, filename_prefix, prompt, extra_pnginfo)
        # Not under "images": the app takes those as the job's still.
        return {"ui": {"castcut": [report]}}


def save_images(images, filename_prefix, prompt=None, extra_pnginfo=None):
    """SaveImage's file naming, into the output folder; returns view refs."""
    import folder_paths  # noqa: PLC0415 - ComfyUI only
    from PIL import Image  # noqa: PLC0415
    from PIL.PngImagePlugin import PngInfo  # noqa: PLC0415

    array = _to_numpy(images)
    output_dir = folder_paths.get_output_directory()
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, array.shape[2], array.shape[1]
    )
    refs = []
    for index, frame in enumerate(array):
        img = Image.fromarray(np.clip(255.0 * frame, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        for key, value in (extra_pnginfo or {}).items():
            metadata.add_text(key, json.dumps(value))
        name = f"{filename.replace('%batch_num%', str(index))}_{counter:05}_.png"
        img.save(os.path.join(full_folder, name), pnginfo=metadata, compress_level=4)
        refs.append({"filename": name, "subfolder": subfolder, "type": "output"})
        counter += 1
    return refs


NODE_CLASS_MAPPINGS = {
    "CastcutPoseScore": CastcutPoseScore,
    "CastcutPickBest": CastcutPickBest,
    "CastcutFaceDistance": CastcutFaceDistance,
    "CastcutMaskRepair": CastcutMaskRepair,
    "CastcutReport": CastcutReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CastcutPoseScore": "Castcut Pose Score",
    "CastcutPickBest": "Castcut Pick Best",
    "CastcutFaceDistance": "Castcut Face Distance",
    "CastcutMaskRepair": "Castcut Mask Repair",
    "CastcutReport": "Castcut Report",
}
