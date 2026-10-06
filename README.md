# Castcut nodes for ComfyUI

Optional. Castcut runs a few checks after a still lands — the pose check (DWPose + limb angles +
posture), Best of two for hard poses, the cut-out's matte repair. Without this pack each is a
separate ComfyUI call (and Best of two is two jobs, queued one after the other). With it they run
**inside the job**:

- **Best of two in one job** (Day → Best of two for hard poses): the sampler renders both takes as
  a batch (one model load, one encode), DWPose reads both, `CastcutPoseScore` scores them against
  the guide, `CastcutPickBest` hands the closer take to the job's SaveImage and `CastcutReport`
  saves the other take and writes both scores into the job's history, where the app picks them up.
- **Cut-out in one job** (Isolate on white / plates): BiRefNet matte → `CastcutMaskRepair` (hole
  fill from the photo's colours) → the original pixels composited onto the fill — no mask round
  trip.

The app checks ComfyUI's `object_info`; when the nodes are missing it queues exactly the graphs it
always did.

## Nodes

| Node | In | Out |
| --- | --- | --- |
| `CastcutPoseScore` | `POSE_KEYPOINT` (DWPose), `guide_json` | per-image results JSON (the app's `PoseMatchResult`), first score |
| `CastcutPickBest` | `IMAGE` batch, scores JSON | best image, other image, best index, report JSON |
| `CastcutFaceDistance` | `ANALYSIS_MODELS` (ComfyUI_FaceAnalysis), reference, images | cosine distance per image (100 = no face) |
| `CastcutMaskRepair` | image, mask, fill `#rrggbb`, regrow edges, optional LoadImage mask | composite, repaired mask, report JSON |
| `CastcutReport` | report JSON, optional alternate image | UI output `castcut` (read from `/history`), alternate saved to output |

## Routes (1.2.0, more in 1.3.0 and 1.4.0)

The pack also adds HTTP routes to ComfyUI's server (`castcut_nodes.py`, "HTTP routes"). The app
asks `GET /castcut/info` and falls back to queued graphs when a route is missing or fails.

| Route | Does |
| --- | --- |
| `GET /castcut/info` | version, routes, whether InsightFace is importable |
| `POST /castcut/analyze` | `face-distance` (reference + images → cosine distances, 100 = no face) and `face-boxes` (image + ImageRotate turns → boxes, stops at the first turn with a face); images are `{filename, subfolder, type}` read in place, or `{data: <base64>}`; `provider` `CPU` (default) or `CUDA` |
| `POST /castcut/stage` | copy an output/temp file into `input/` as `<prefix>-<sha256[:16]><ext>`, the app's upload name for the same bytes |
| `GET /castcut/object-info-fingerprint` | a hash of the node list and the model file lists (not the input folder) |
| `POST /castcut/analyze` `face-probe` (1.3.0) | the Face finish probe: the `count` largest faces, each cropped with `paddingPercent` (FaceBoundingBox) and compared with a reference → `[{x, distance}]` |
| `POST /castcut/analyze` `pose` (1.3.0) | DWPose `openpose_json` (body + hands, no face; yolox_l.onnx, dw-ll_ucoco_384.onnx, 512) from comfyui_controlnet_aux, built once with **CPU** onnxruntime sessions |
| `POST /castcut/input-delete` (1.3.0) | delete the named files from `input/`: plain top-level names only, at least `minAgeSeconds` old (never under a day), not named by a running or pending job → deleted / skipped with reasons |
| `GET /castcut/health` (1.3.0) | queue counts, free VRAM, loaded model classes, which analyzers are installed and loaded; from 1.4.0 `usage` — per route / op, how many calls were served or failed and their average time since ComfyUI started |
| `POST /castcut/analyze` `person-poses` (1.4.0) | the app's two-person pose read: the Impact Pack's person segmentation (`SegmDetectorSEGS` with the YOLO model pinned to the CPU), each of the `count` largest people alone on grey (`ImpactSEGSOrderedFilter`, `SegsToCombinedMask`, ComfyUI's composite), DWPose on each → one `openpose_json` per person |
| `GET /castcut/png-text` (1.4.0) | `?filename&subfolder&type&keys=prompt` — a PNG's text chunks (the graph it was made with) without sending the picture |

The analyzer is InsightFace buffalo_l loaded once per provider, with ComfyUI_FaceAnalysis'
detection (sizes 640 → 320 until a face shows, largest first) and FaceEmbedDistance's cosine.
Paths are resolved inside ComfyUI's input / output / temp folders only. The pose copy is DWPose's
own detector and annotator call (progress bar off), so its keypoints are the node's; it runs on
the CPU so it never takes GPU memory from a render (~0.3 s a still).

`CastcutPoseScore` and `CastcutMaskRepair` are ports of `src/lib/pose-score.ts`,
`pose-limb-score.ts`, `pose-posture.ts` and `src/lib/isolate-mask.ts`. They stay in step through
shared test vectors (`tests/vectors/*.json`) that both test suites read.

Needs only what ComfyUI already has (numpy, PIL; cv2 is used when present). Best of two needs
`comfyui_controlnet_aux` (DWPose); `CastcutFaceDistance` needs `ComfyUI_FaceAnalysis` with
insightface and fails with a clear message without it.

## Install

The pack is one file, `castcut_nodes.py` (plus this folder's `__init__.py` when installed as a
folder). Pick one way and restart ComfyUI afterwards.

**From the Castcut app** (easiest): Settings → ComfyUI → **Castcut nodes** shows whether ComfyUI
has the pack and which version, installs it with ComfyUI-Manager when it can, restarts ComfyUI
(after asking, and only when its queue is empty), or hands you a copy-paste command for your
setup. See [docs/castcut-nodes.md](https://github.com/doodersrage/castcut/blob/main/docs/castcut-nodes.md).

1. **ComfyUI-Manager by name** (from the Comfy Registry): Manager → Custom Nodes
   Manager → search *Castcut nodes* → Install; or `comfy node install castcut-nodes`.

2. **ComfyUI-Manager → Install via Git URL**: `https://github.com/doodersrage/castcut-nodes`. Manager
   3.41+ allows Git-URL installs only with `allow_git_url_install = true` under `[default]` in
   `ComfyUI/user/__manager/config.ini` and ComfyUI listening on 127.0.0.1. The repository holds
   just this pack.

3. **Copy the file** (smallest; works when the ComfyUI service user can't read your home folder):

   ```bash
   sudo install -m 644 comfyui-nodes/castcut/castcut_nodes.py \
     /opt/comfyui/custom_nodes/castcut_nodes.py
   sudo systemctl restart comfyui
   ```

   (Use your ComfyUI path; a desktop ComfyUI: copy into `ComfyUI/custom_nodes/`; Windows
   portable: `ComfyUI_windows_portable\ComfyUI\custom_nodes\`. A running Castcut serves the file at
   `/api/castcut-nodes/file`.)

4. **Symlink the folder** (ComfyUI must be able to read the repo):

   ```bash
   ln -s "$PWD/comfyui-nodes/castcut" /path/to/ComfyUI/custom_nodes/castcut
   ```

Check `http://127.0.0.1:8188/object_info/CastcutPoseScore` returns the node; its `description`
ends with `[castcut-nodes 1.4.0]`, which is how the app reads the installed version. To update,
copy / pull / update again and restart. Install it one way only — a `castcut_nodes.py` file and a
`castcut` folder side by side both load.

**Without the pack** the app checks `object_info`, finds the nodes missing and queues exactly the
graphs it always did: Best of two as two jobs, cut-outs repaired in the app.

## Versions and publishing

`CASTCUT_VERSION` in `castcut_nodes.py`, `version` in `pyproject.toml` and the app's
`CASTCUT_NODES_BUNDLED_VERSION` (`src/lib/castcut-nodes-setup.ts`) move together; the tests check
it. `pyproject.toml` carries the Comfy Registry metadata (`[tool.comfy]`, `PublisherId` is a
placeholder until a publisher exists). `scripts/castcut-nodes-release.sh <dir>` copies this folder,
the licence and a standalone README note into `<dir>`, ready to push as its own repository and
publish with `comfy node publish`.

Publishing: push a `v<version>` tag to the standalone repository and
`.github/workflows/publish.yml` runs the tests, checks the tag matches `pyproject.toml` and
`CASTCUT_VERSION`, and publishes with Comfy's publish action (repository secret
`REGISTRY_ACCESS_TOKEN`, a publisher API key from registry.comfy.org).

## Tests

Plain Python, no ComfyUI or torch:

```bash
python3 -m pytest comfyui-nodes/castcut/tests      # or
python3 -m unittest discover -s comfyui-nodes/castcut/tests
```

The vectors come from the TypeScript code. After changing the pose check or the mask repair:

```bash
node --import tsx scripts/castcut-test-vectors.mts   # keeps the real rows, redoes the synthetic ones
npm test                                             # src/lib/castcut-vectors.test.ts
python3 -m pytest comfyui-nodes/castcut/tests        # the port follows?
```

`--dataset <pose-calibration.json> --every 6` and `--real-mask <still.png>:<matte.png>` refresh the
real rows (see the script header).
