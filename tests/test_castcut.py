"""
Castcut node pack tests — plain Python, no ComfyUI or torch needed:

    python3 -m unittest discover -s comfyui-nodes/castcut/tests
    python3 -m pytest comfyui-nodes/castcut/tests

The vectors in tests/vectors/ are written by scripts/castcut-test-vectors.mts from the TypeScript
code the app runs; src/lib/castcut-vectors.test.ts checks the TS side still gives them.
"""

import base64
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
VECTORS = os.path.join(HERE, "vectors")

spec = importlib.util.spec_from_file_location("castcut_nodes", os.path.join(PACK, "castcut_nodes.py"))
castcut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(castcut)


def load(name):
    with open(os.path.join(VECTORS, name), encoding="utf8") as handle:
        return json.load(handle)["cases"]


def b64(text):
    return np.frombuffer(base64.b64decode(text), dtype=np.uint8)


# Rounded TS numbers (Math.round(x * 100) / 100, Math.round of degrees) can land one step apart
# when V8 and libm disagree in the last bit of a hypot / acos; anything more is a real drift.
ROUNDED_KEYS = {"score", "jointScore", "limbScore", "defining"}


def assert_close(test, expected, actual, path="$"):
    if isinstance(expected, dict):
        test.assertIsInstance(actual, dict, path)
        test.assertEqual(sorted(expected.keys()), sorted(actual.keys()), path)
        for key in expected:
            assert_close(test, expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        test.assertIsInstance(actual, list, path)
        test.assertEqual(len(expected), len(actual), path)
        for index, (e, a) in enumerate(zip(expected, actual)):
            assert_close(test, e, a, f"{path}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        test.assertEqual(expected, actual, path)
    elif isinstance(expected, (int, float)):
        test.assertIsInstance(actual, (int, float), path)
        key = path.rsplit(".", 1)[-1].split("[")[0]
        if key in ROUNDED_KEYS:
            test.assertLessEqual(abs(expected - actual), 0.0101, path)
        elif key in {"deltaDeg", "tiltDeg"}:
            test.assertLessEqual(abs(expected - actual), 1, path)
        else:
            test.assertTrue(math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9), f"{path}: {expected} != {actual}")
    else:
        test.fail(f"{path}: unexpected {expected!r}")


def same(expected, actual):
    """Equal up to the last bits of an unrounded float (V8 and libm differ there): every rounded
    score, posture word and verdict must be identical."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(
            same(expected[k], actual[k]) for k in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            same(e, a) for e, a in zip(expected, actual)
        )
    if isinstance(expected, float) and isinstance(actual, float):
        return math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-15)
    return expected == actual and type(expected) is not bool or expected is actual


class PoseScoreVectors(unittest.TestCase):
    def test_matches_typescript(self):
        cases = load("pose-score.json")
        self.assertGreater(len(cases), 40)
        exact = 0
        for case in cases:
            with self.subTest(case["id"]):
                detected = castcut.parse_openpose_json(case["openpose"])
                if case["expected"] is None:
                    self.assertIsNone(detected)
                    continue
                guide = [
                    [{"x": p["x"], "y": p["y"]} if p else None for p in body]
                    for body in case["guide"]["people"]
                ]
                actual = castcut.score_pose_match(
                    guide, case["guide"]["aspect"], detected, case.get("method", "limb-angle")
                )
                assert_close(self, case["expected"], actual)
                exact += same(case["expected"], actual)
        # How many match to 1e-12 (the rest within one rounding step of a rounded score).
        print(f"\npose vectors: {exact}/{len(cases)} identical to the TS output (floats to 1e-12)")

    def test_parse_handles_strings_lists_and_garbage(self):
        frame = {"canvas_width": 100, "canvas_height": 200, "people": [{"pose_keypoints_2d": [10, 20, 1] * 18}]}
        self.assertEqual(castcut.parse_openpose_json(json.dumps([frame]))["people"][0][0], {"x": 0.1, "y": 0.1})
        self.assertIsNone(castcut.parse_openpose_json("not json"))
        self.assertIsNone(castcut.parse_openpose_json([]))
        # Pixels without a canvas can't be normalized.
        self.assertIsNone(castcut.parse_openpose_json({"people": frame["people"]}))


class PickBest(unittest.TestCase):
    def test_follows_pick_better_take(self):
        self.assertEqual(castcut.pick_best_index([0.7, 0.6]), 0)
        self.assertEqual(castcut.pick_best_index([0.6, 0.7]), 1)
        self.assertEqual(castcut.pick_best_index([0.6, 0.6]), 1)  # tie: the later take
        self.assertEqual(castcut.pick_best_index([None, 0.1]), 1)
        self.assertEqual(castcut.pick_best_index([0.1, None]), 0)
        self.assertEqual(castcut.pick_best_index([None, None]), 1)
        self.assertEqual(castcut.pick_best_index([0.2, 0.9, 0.5]), 1)

    def test_node_splits_the_batch(self):
        images = np.stack([np.full((4, 4, 3), 0.1, np.float32), np.full((4, 4, 3), 0.9, np.float32)])
        scores = json.dumps({"results": [{"score": 0.81}, {"score": 0.42}]})
        best, other, index, report = castcut.CastcutPickBest().pick(images, scores)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(float(best[0, 0, 0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(other[0, 0, 0, 0]), 0.9, places=5)
        parsed = json.loads(report)
        self.assertEqual(parsed["scores"], [0.81, 0.42])
        self.assertEqual((parsed["bestIndex"], parsed["otherIndex"]), (0, 1))


class PoseScoreNode(unittest.TestCase):
    def test_scores_each_frame(self):
        case = next(c for c in load("pose-score.json") if c["id"].startswith("real:") and c["expected"])
        guide = json.dumps({"guide": case["guide"]["people"], "aspect": case["guide"]["aspect"]})
        scores_json, first = castcut.CastcutPoseScore().score([case["openpose"], {"people": []}], guide)
        results = json.loads(scores_json)["results"]
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["score"], case["expected"]["score"], delta=0.0101)
        self.assertEqual(first, results[0]["score"])
        # An empty frame (no canvas, nobody) can't be read.
        self.assertIsNone(results[1])

    def test_bad_guide_fails_clearly(self):
        with self.assertRaises(ValueError):
            castcut.CastcutPoseScore().score([], json.dumps({"guide": []}))


class MaskRepairVectors(unittest.TestCase):
    def test_matches_typescript_exactly(self):
        for case in load("mask-repair.json"):
            with self.subTest(case["id"]):
                w, h = case["width"], case["height"]
                rgba = b64(case["rgba"]).reshape(h, w, 4)
                alpha = b64(case["alpha"]).reshape(h, w)
                result = castcut.repair_subject_mask(alpha, rgba[..., :3], regrow_edges=case["regrowEdges"])
                expected = case["expected"]
                self.assertEqual(result["filledHolePixels"], expected["filledHolePixels"])
                self.assertEqual(result["regrownPixels"], expected["regrownPixels"])
                self.assertEqual(result["backdrop"], expected["backdrop"])
                self.assertEqual(result["plainBackdrop"], expected["plainBackdrop"])
                np.testing.assert_array_equal(result["alpha"].reshape(-1), b64(expected["alpha"]))
                composite = castcut.composite_through_mask(rgba, result["alpha"], case["fill"])
                np.testing.assert_array_equal(composite.reshape(-1), b64(expected["composite"]))

    def test_pure_python_labelling_matches_cv2(self):
        real_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = None  # import cv2 -> ImportError: the BFS fallback
        try:
            for case in load("mask-repair.json")[:3]:
                w, h = case["width"], case["height"]
                rgba = b64(case["rgba"]).reshape(h, w, 4)
                result = castcut.repair_subject_mask(
                    b64(case["alpha"]).reshape(h, w), rgba[..., :3], regrow_edges=case["regrowEdges"]
                )
                np.testing.assert_array_equal(result["alpha"].reshape(-1), b64(case["expected"]["alpha"]))
        finally:
            if real_cv2 is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = real_cv2

    def test_node_round_trips_comfy_tensors(self):
        case = next(c for c in load("mask-repair.json") if c["id"] == "synthetic:cape-hole-and-arm-gap")
        w, h = case["width"], case["height"]
        rgba = b64(case["rgba"]).reshape(h, w, 4)
        image = (rgba[..., :3].astype(np.float32) / np.float32(255))[None]
        mask = (b64(case["alpha"]).reshape(h, w).astype(np.float32) / np.float32(255))[None]
        composite, repaired, report = castcut.CastcutMaskRepair().repair(image, mask, "#FFFFFF", False)
        expected = b64(case["expected"]["composite"]).reshape(h, w, 4)[..., :3]
        # What SaveImage writes from the composite is the TS composite, byte for byte.
        saved = np.clip(np.float32(255.0) * composite[0], 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(saved, expected)
        self.assertEqual(json.loads(report)["images"][0]["filledHolePixels"], 96)
        self.assertEqual(repaired.shape, (1, h, w))

    def test_fill_parse(self):
        self.assertEqual(castcut.parse_fill("#00b140"), {"r": 0, "g": 177, "b": 64})
        self.assertEqual(castcut.parse_fill("fff"), {"r": 255, "g": 255, "b": 255})
        with self.assertRaises(ValueError):
            castcut.parse_fill("white")


class FaceDistance(unittest.TestCase):
    def test_cosine_per_image_and_no_face(self):
        class Models:
            def get_embeds(self, frame):
                value = int(frame[0, 0, 0])
                return None if value == 0 else np.array([1.0, value / 255.0], dtype=np.float32)

        reference = np.full((1, 2, 2, 3), 1.0, np.float32)
        image = np.stack([np.full((2, 2, 3), 1.0, np.float32), np.zeros((2, 2, 3), np.float32)])
        distances_json, first = castcut.CastcutFaceDistance().measure(Models(), reference, image)
        distances = json.loads(distances_json)["distances"]
        self.assertEqual(distances, [0.0, 100.0])
        self.assertEqual(first, 0.0)

    def test_without_face_analysis_fails_clearly(self):
        with self.assertRaises(RuntimeError):
            castcut.CastcutFaceDistance().measure(object(), np.zeros((1, 2, 2, 3)), np.zeros((1, 2, 2, 3)))


class Report(unittest.TestCase):
    def test_ui_output_and_alternate_save(self):
        with tempfile.TemporaryDirectory() as out_dir:
            fake = types.ModuleType("folder_paths")
            fake.get_output_directory = lambda: out_dir
            fake.get_save_image_path = lambda prefix, base, w, h: (out_dir, prefix, 7, "", prefix)
            sys.modules["folder_paths"] = fake
            try:
                alternate = np.full((1, 3, 2, 3), 0.5, np.float32)
                result = castcut.CastcutReport().report(
                    json.dumps({"kind": "pick-best", "bestIndex": 1}), alternate, "castcut-alt", json.dumps({"a": 1})
                )
            finally:
                del sys.modules["folder_paths"]
            report = result["ui"]["castcut"][0]
            self.assertNotIn("images", result["ui"])
            self.assertEqual(report["bestIndex"], 1)
            self.assertEqual(report["extra"], {"a": 1})
            self.assertEqual(report["alternates"], [{"filename": "castcut-alt_00007_.png", "subfolder": "", "type": "output"}])
            self.assertTrue(os.path.exists(os.path.join(out_dir, "castcut-alt_00007_.png")))


class Registration(unittest.TestCase):
    def test_every_node_declares_the_comfy_api(self):
        self.assertEqual(set(castcut.NODE_CLASS_MAPPINGS), set(castcut.NODE_DISPLAY_NAME_MAPPINGS))
        for name, cls in castcut.NODE_CLASS_MAPPINGS.items():
            with self.subTest(name):
                self.assertIn("required", cls.INPUT_TYPES())
                self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))
                self.assertEqual(cls.CATEGORY, "Castcut")

    def test_every_description_carries_the_version_marker(self):
        # The app reads the installed version from object_info's `description`.
        marker = f"[castcut-nodes {castcut.CASTCUT_VERSION}]"
        for name, cls in castcut.NODE_CLASS_MAPPINGS.items():
            with self.subTest(name):
                self.assertTrue(cls.DESCRIPTION.endswith(marker), cls.DESCRIPTION)

    def test_versions_agree(self):
        # pyproject.toml (Comfy Registry), castcut_nodes.py and the app's bundled version.
        import re  # noqa: PLC0415

        with open(os.path.join(PACK, "pyproject.toml"), encoding="utf8") as handle:
            pyproject = handle.read()
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), castcut.CASTCUT_VERSION)
        app_constant = os.path.join(PACK, "..", "..", "src", "lib", "castcut-nodes-setup.ts")
        if os.path.exists(app_constant):  # absent in the standalone repo
            with open(app_constant, encoding="utf8") as handle:
                app = handle.read()
            self.assertIn(f"CASTCUT_NODES_BUNDLED_VERSION = '{castcut.CASTCUT_VERSION}'", app)

    def test_package_init_exports_the_mappings(self):
        # ComfyUI imports a custom_nodes folder as a package (`castcut/__init__.py`).
        spec_pkg = importlib.util.spec_from_file_location(
            "castcut_pkg", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK]
        )
        package = importlib.util.module_from_spec(spec_pkg)
        sys.modules["castcut_pkg"] = package
        try:
            spec_pkg.loader.exec_module(package)
        finally:
            del sys.modules["castcut_pkg"]
        self.assertEqual(set(package.NODE_CLASS_MAPPINGS), set(castcut.NODE_CLASS_MAPPINGS))
        self.assertEqual(package.__version__, castcut.CASTCUT_VERSION)


if __name__ == "__main__":
    unittest.main()
