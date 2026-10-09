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
import io


def _png_bytes():
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


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



class RouteHelperTests(unittest.TestCase):
    """The pure parts of the HTTP routes (castcut_nodes.py, "HTTP routes")."""

    def test_routes_are_not_registered_outside_comfyui(self):
        self.assertFalse(castcut.ROUTES_REGISTERED)
        self.assertEqual(castcut.info_payload()["version"], castcut.CASTCUT_VERSION)

    def test_safe_ref_path_stays_inside_the_folder(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "day"))
            self.assertEqual(
                castcut.safe_ref_path(root, "a.png", "day"),
                os.path.join(os.path.realpath(root), "day", "a.png"),
            )
            for name, sub in [
                ("../secret.png", ""),
                ("a.png", "../.."),
                ("a.png", "day/../../etc"),
                ("..", ""),
                ("", ""),
                ("a\x00.png", ""),
                ("sub\\a.png", ""),
            ]:
                with self.assertRaises(ValueError, msg=(name, sub)):
                    castcut.safe_ref_path(root, name, sub)
            # A symlink out of the folder is refused too.
            outside = tempfile.mkdtemp()
            try:
                os.symlink(outside, os.path.join(root, "link"))
                with self.assertRaises(ValueError):
                    castcut.safe_ref_path(root, "x.png", "link")
            finally:
                os.rmdir(outside)

    def test_rotation_matches_comfy_image_rotate(self):
        # torch.rot90(image, k, dims=[2, 1]) on B×H×W×C: 90 degrees turns clockwise.
        image = np.arange(6).reshape(2, 3, 1)
        self.assertEqual(castcut.rotate_like_comfy(image, "none").tolist(), image.tolist())
        self.assertEqual(
            castcut.rotate_like_comfy(image, "90 degrees")[..., 0].tolist(), [[3, 0], [4, 1], [5, 2]]
        )
        self.assertEqual(
            castcut.rotate_like_comfy(image, "270 degrees")[..., 0].tolist(), [[2, 5], [1, 4], [0, 3]]
        )
        with self.assertRaises(ValueError):
            castcut.rotation_turns("45 degrees")

    def test_content_input_name_matches_the_app(self):
        # src/lib/comfy-input-name.ts contentAddressedInputName: "<prefix>-<16 hex><ext>".
        self.assertEqual(
            castcut.content_input_name("face-check", ".png", "ABCDEF0123456789ffff"),
            "face-check-abcdef0123456789.png",
        )
        for bad in [("a/b", ".png", "0" * 64), ("a", "png", "0" * 64), ("a", ".png", "xyz")]:
            with self.assertRaises(ValueError, msg=bad):
                castcut.content_input_name(*bad)

    def test_cosine_face_distance_is_face_embed_distance(self):
        self.assertEqual(castcut.cosine_face_distance([1, 0], None), 100.0)
        self.assertEqual(castcut.cosine_face_distance([1, 2], [1, 2]), 0.0)
        self.assertAlmostEqual(castcut.cosine_face_distance([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(castcut.cosine_face_distance([1, 0], [1, 1]), 1 - 1 / math.sqrt(2))

    def test_face_boxes_are_largest_first_and_clamped(self):
        faces = [{"bbox": [10, 10, 20, 20]}, {"bbox": [-5, 50, 105, 130]}]
        self.assertEqual(
            castcut.face_boxes(faces, 100, 120),
            [
                {"x": 0, "y": 50, "width": 100, "height": 70},
                {"x": 10, "y": 10, "width": 10, "height": 10},
            ],
        )

    def test_fingerprint_changes_with_nodes_or_models_only(self):
        base = castcut.object_info_fingerprint(["A", "B"], {"loras": ["x.safetensors"]})
        self.assertEqual(base, castcut.object_info_fingerprint(["B", "A"], {"loras": ["x.safetensors"]}))
        self.assertNotEqual(base, castcut.object_info_fingerprint(["A"], {"loras": ["x.safetensors"]}))
        self.assertNotEqual(base, castcut.object_info_fingerprint(["A", "B"], {"loras": ["y.safetensors"]}))

    def test_analyze_request_with_a_fake_analyzer(self):
        class Fake:
            def __init__(self):
                self.calls = []

            def embedding(self, rgb, provider):
                self.calls.append(provider)
                return None if rgb.mean() < 10 else np.array([1.0, float(rgb.mean() > 200)])

            def faces(self, rgb, provider):
                return [{"bbox": [1, 2, 3, 5]}] if rgb.shape[0] > rgb.shape[1] else []

        def png(value, size=(4, 4)):
            import io  # noqa: PLC0415

            from PIL import Image  # noqa: PLC0415

            buffer = io.BytesIO()
            Image.new("RGB", size, (value, value, value)).save(buffer, "PNG")
            return {"data": base64.b64encode(buffer.getvalue()).decode()}

        fake = Fake()
        result = castcut.analyze_request(
            {"op": "face-distance", "reference": png(100), "images": [png(100), png(250), png(0)]},
            fake,
        )
        self.assertEqual(result["distances"][0], 0.0)
        self.assertEqual(result["distances"][2], 100.0)
        self.assertEqual(fake.calls[0], "CPU")
        self.assertEqual(
            castcut.analyze_request({"op": "face-distance", "reference": png(0), "images": []}, fake),
            {"op": "face-distance", "error": "no-face-in-reference"},
        )
        # Wide picture: no face as is, found after a quarter turn; the search stops there.
        boxes = castcut.analyze_request(
            {
                "op": "face-boxes",
                "image": png(100, (8, 4)),
                "rotations": ["none", "90 degrees", "270 degrees"],
            },
            fake,
        )
        self.assertEqual([r["rotation"] for r in boxes["results"]], ["none", "90 degrees"])
        self.assertEqual(boxes["results"][1]["boxes"], [{"x": 1, "y": 2, "width": 2, "height": 3}])
        for bad in [{"op": "nope"}, {"op": "face-boxes", "provider": "TPU", "image": png(1)}]:
            with self.assertRaises(ValueError):
                castcut.analyze_request(bad, fake)


class RouteHelperTests13(unittest.TestCase):
    """1.3.0: face probe, input delete, health (castcut_nodes.py, "HTTP routes")."""

    def test_probe_faces_follows_face_bounding_box_indexes(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        big = {"bbox": [100, 10, 140, 50]}
        small = {"bbox": [10, 10, 30, 30]}
        seen = []

        def embed(crop):
            seen.append(crop.shape[:2])
            return np.array([1.0, 0.0]) if crop.shape[1] > 50 else np.array([0.0, 1.0])

        faces = castcut.probe_faces(rgb, [small, big], embed, np.array([1.0, 0.0]))
        # Largest first, padded by 30% of the box: 100 - 12 = 88; 10 - 6 = 4.
        self.assertEqual([f["x"] for f in faces], [88, 4])
        self.assertEqual(faces[0]["distance"], 0.0)
        self.assertAlmostEqual(faces[1]["distance"], 1.0)
        self.assertEqual(seen[0], (62, 64))  # top clamps to 0: 0..62
        # One face answers both indexes; none is an empty probe.
        self.assertEqual(len({f["x"] for f in castcut.probe_faces(rgb, [big], embed, [1, 0])}), 1)
        self.assertEqual(castcut.probe_faces(rgb, [], embed, [1, 0]), [])

    def test_plan_input_delete_keeps_new_queued_and_strange_names(self):
        import time  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as root:
            now = time.time()
            for name, age in [("old-a.png", 10), ("old-b.png", 10), ("queued.png", 10), ("new.png", 0.5)]:
                path = os.path.join(root, name)
                with open(path, "wb") as handle:
                    handle.write(b"x")
                os.utime(path, (now - age * 86_400, now - age * 86_400))
            delete, skipped = castcut.plan_input_delete(
                ["old-a.png", "old-a.png", "queued.png", "new.png", "gone.png", "../x.png", "old-b.png"],
                root,
                json.dumps({"1": {"inputs": {"image": "queued.png"}}}),
                now,
                0,  # asked for no minimum: the one-day floor still holds
            )
            self.assertEqual(sorted(os.path.basename(p) for p in delete), ["old-a.png", "old-b.png"])
            self.assertEqual(
                {s["name"]: s["reason"] for s in skipped},
                {
                    "queued.png": "in-queue",
                    "new.png": "too-new",
                    "gone.png": "missing",
                    "../x.png": "invalid-name",
                },
            )

    def test_health_and_info_without_comfyui(self):
        health = castcut.health_payload({"running": 0, "pending": 2})
        self.assertEqual(health["queue"], {"running": 0, "pending": 2})
        self.assertFalse(health["dwpose"])
        info = castcut.info_payload()
        self.assertIn("input-delete", info["routes"])
        self.assertEqual(info["analyze"]["ops"], list(castcut.ANALYZE_OPS))

    def test_pose_op_without_dwpose_says_so(self):
        class NoPose:
            def available(self):
                return False

        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, "PNG")
        image = {"data": base64.b64encode(buffer.getvalue()).decode()}
        self.assertEqual(
            castcut.analyze_request({"op": "pose", "image": image}, None, NoPose()),
            {"op": "pose", "error": "no-dwpose"},
        )


class RouteHelperTests14(unittest.TestCase):
    """1.4.0: usage counts, png text, person poses' guards."""

    def setUp(self):
        castcut.USAGE.clear()

    def test_usage_counts_served_errors_and_average(self):
        castcut.record_usage("analyze:pose", True, 300)
        castcut.record_usage("analyze:pose", True, 500)
        castcut.record_usage("analyze:pose", False, 100)
        castcut.record_usage("stage", True, 4)
        routes = castcut.usage_payload()["routes"]
        self.assertEqual(routes["analyze:pose"], {"served": 2, "errors": 1, "avgMs": 300.0})
        self.assertEqual(list(routes), ["analyze:pose", "stage"])

    def test_png_text_reads_chunks_not_pixels(self):
        from PIL import Image  # noqa: PLC0415
        from PIL.PngImagePlugin import PngInfo  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as root:
            meta = PngInfo()
            meta.add_text("prompt", json.dumps({"1": {"class_type": "KSampler"}}))
            meta.add_text("workflow", "{}")
            path = os.path.join(root, "still.png")
            Image.new("RGB", (8, 8)).save(path, pnginfo=meta)
            chunks = castcut.png_text(path)
            self.assertEqual(json.loads(chunks["prompt"])["1"]["class_type"], "KSampler")
            self.assertEqual(chunks["workflow"], "{}")
            bare = os.path.join(root, "bare.png")
            Image.new("RGB", (8, 8)).save(bare)
            self.assertEqual(castcut.png_text(bare), {})

    def test_person_poses_needs_its_packs_and_a_plain_model_name(self):
        class Yes:
            def available(self):
                return True

        class No:
            def available(self):
                return False

        image = {"filename": "x.png", "type": "output"}
        self.assertEqual(
            castcut.analyze_request({"op": "person-poses", "image": image}, None, Yes(), No()),
            {"op": "person-poses", "error": "no-person-read"},
        )
        for bad in ["segm/../../etc/x.pt", "segm/a/b.pt"]:
            with self.assertRaises(ValueError, msg=bad):
                castcut.analyze_request(
                    {"op": "person-poses", "image": image, "model": bad}, None, Yes(), Yes()
                )


    def test_duo_counts_needs_its_nodes_checks_models_and_counts(self):
        class Pose:
            def available(self):
                return True

            def openpose_json(self, rgb, hands, body, face):
                assert (hands, body, face) == (False, True, False)
                return "[]"

        class Reader:
            def __init__(self, ok=True):
                self.ok = ok
                self.asked = None

            def available(self):
                return self.ok

            def counts_available(self):
                return self.ok

            def counts(self, rgb, models, threshold):
                self.asked = (models, threshold)
                return {key: 2 for key in models}

        image = {"data": base64.b64encode(_png_bytes()).decode()}
        self.assertEqual(
            castcut.analyze_request({"op": "duo-counts", "image": image}, None, Pose(), Reader(False)),
            {"op": "duo-counts", "error": "no-duo-counts"},
        )
        reader = Reader()
        reply = castcut.analyze_request(
            {
                "op": "duo-counts",
                "image": image,
                "threshold": 0.5,
                "models": {
                    "faces": {"model": "bbox/face_yolov8m.pt", "segm": False},
                    "penises": {"model": "segm/nsfw-seg-penis-s.pt", "segm": True},
                },
            },
            None,
            Pose(),
            reader,
        )
        self.assertEqual(reply, {"op": "duo-counts", "counts": {"faces": 2, "penises": 2}, "openpose_json": "[]"})
        self.assertEqual(
            reader.asked,
            ({"faces": ("bbox/face_yolov8m.pt", False), "penises": ("segm/nsfw-seg-penis-s.pt", True)}, 0.5),
        )
        for models in ({"faces": {"model": "bbox/../x.pt"}}, {"elbows": {"model": "bbox/a.pt"}}, {"faces": {}}):
            with self.assertRaises(ValueError, msg=str(models)):
                castcut.analyze_request(
                    {"op": "duo-counts", "image": image, "models": models}, None, Pose(), Reader()
                )



class EditorWorkflowTests(unittest.TestCase):
    """1.5.0: "Open in ComfyUI" writes the still's graph where the editor can open it by URL."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = (castcut.PACK_DIR, castcut.EDITOR_WORKFLOWS_DIR, castcut.EDITOR_WORKFLOWS_AT_START)
        pack = os.path.join(self.tmp, "castcut-nodes")
        os.makedirs(os.path.join(pack, "example_workflows"))
        castcut.PACK_DIR = pack
        castcut.EDITOR_WORKFLOWS_DIR = os.path.join(pack, "example_workflows")
        castcut.EDITOR_WORKFLOWS_AT_START = True

    def tearDown(self):
        castcut.PACK_DIR, castcut.EDITOR_WORKFLOWS_DIR, castcut.EDITOR_WORKFLOWS_AT_START = self.saved

    def test_single_file_install_has_no_source(self):
        self.assertEqual(castcut.editor_workflow_request({"name": "x", "workflow": {"nodes": []}}),
                         {"source": None, "reason": "single-file"})

    def test_writes_template_and_keeps_newest(self):
        open(os.path.join(castcut.PACK_DIR, "__init__.py"), "w").close()
        result = castcut.editor_workflow_request({"name": "day-2026-10-07-ab12", "workflow": {"nodes": [1]}})
        self.assertEqual(result, {"source": "castcut-nodes", "template": "castcut-day-2026-10-07-ab12"})
        with open(os.path.join(castcut.EDITOR_WORKFLOWS_DIR, "castcut-day-2026-10-07-ab12.json")) as handle:
            self.assertEqual(json.load(handle), {"nodes": [1]})
        for index in range(castcut.EDITOR_WORKFLOW_KEEP + 3):
            castcut.editor_workflow_request({"name": f"n{index}", "workflow": {"nodes": []}})
        names = [n for n in os.listdir(castcut.EDITOR_WORKFLOWS_DIR) if n.startswith("castcut-")]
        self.assertEqual(len(names), castcut.EDITOR_WORKFLOW_KEEP)

    def test_rejects_bad_names_and_needs_a_restart_for_a_new_folder(self):
        open(os.path.join(castcut.PACK_DIR, "__init__.py"), "w").close()
        for bad in ["../x", "a b", ""]:
            with self.assertRaises(ValueError, msg=bad):
                castcut.editor_workflow_request({"name": bad, "workflow": {"nodes": []}})
        castcut.EDITOR_WORKFLOWS_AT_START = False
        self.assertEqual(castcut.editor_workflow_request({"name": "x", "workflow": {"nodes": []}})["reason"], "restart")


if __name__ == "__main__":
    unittest.main()
