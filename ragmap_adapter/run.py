"""``ragmap-run``: RoboHop's segment graphs, and localization across trajectories.

    ragmap-run --input /input --output /output [mapper.KEY=VALUE] [localizer.KEY=VALUE]
               [adapter.localizer_state=per_query|sequential]

For each trajectory, builds the release's topological segment map
(``libs/mapper/map_topo.py``'s ``MapTopological.create_map_topo``, configured
exactly as ``scripts/create_maps_hm3d.py`` configures it). Then, for each
requested (source, target) pair, localizes the requested source map images
against the target map with the release's ``LocalizeTopological``
(``libs/localizer/loc_topo.py``), configured from the release's own RoboHop
config (``configs/defaults.yaml``, ``goal_gen``). Upstream code is imported,
never edited.

INPUT (``/input``, read-only)::

    meta.json    {"schema": "ragmap.segment_graph/v1",
                  "trajectories": [{"id", "images": "<dir>", "frames": "<dir>/frames.jsonl",
                                    "graph": "<dir>/graph.pickle" (optional)}],
                  "pairs": [{"source", "target", "queries": [map_index, ...]}]}
    <dir>/frames.jsonl   {"map_index", "frame_index", "frame_id", "image"} per map image
    <dir>/images/000000.jpg ...   the map images, in map order (``map_index``)
    <dir>/graph.pickle   optional: a graph pickle this adapter wrote for exactly
                         these images and mapper settings (see 6. below)

OUTPUT (``/output``)::

    nodes.jsonl  {"id", "trajectory_id", "node", "map_index", "frame_id", "segment",
                  "mask": {"size": [H, W], "counts": [...]}, "centroid": [x, y], "area"}
    edges.jsonl  {"kind": "intra_image" | "temporal" | "cross_trajectory",
                  "source", "target", ...}
    localizations.jsonl  one record per query: the localized map image and the
                  per-map-image vote counts
    run.json     {"schema", "status": "ok" | "failed", "error", "counts", "timings",
                  "versions", "config"}
    <trajectory id>/nodes_fast_sam_graphObject_4_lightglue.pickle   upstream's own graph

Pixel coordinates (masks, centroids, keypoints) are in the release's working
frame: every image resized to ``W x H`` = 320 x 240 (``run.json``
``mask_frame``). ``mask`` is upstream's own uncompressed, column-major RLE
(``libs.common.utils.mask_to_rle_numpy``).

Edge kinds:

- ``intra_image``: the mapper's Delaunay edges between one image's segments.
- ``temporal``: the mapper's inter-image edges within one trajectory
  (``get_robust_DA_edges``, SuperPoint + LightGlue over a window of 3;
  upstream ``edgeType: "da"``).
- ``cross_trajectory``: from the localizer. ``source`` is a source-trajectory
  node (the query segment), ``target`` the target-trajectory node it was
  matched to *in the map image the localizer localized the query to*.
  ``keypoints`` lists the LightGlue matches ``[x_s, y_s, x_t, y_t]`` that fall
  inside both masks -- exactly the matches whose count is the pair's vote in
  ``matchPair_imgWithMask`` (``votes``).

What the adapter does, and all it does:

1. **Stubs** the simulator modules ``libs/common/utils.py`` imports
   (``habitat_sim``, ``quaternion``, ``curses``) when they are absent
   (``ragmap_adapter.stubs``).
2. **The localizer's window covers the whole target map.** There is no pose
   prior across trajectories, so ``loc_radius`` is set to the target map's
   image count, which makes ``getRefImgInds`` return every map image. That is
   a configuration value, not a code change; every other localizer value is
   the release's. Relocalization never fires as a consequence: its window can
   only widen what already covers everything.
3. **No oracle.** The release's RoboHop config sets
   ``use_gt_localization: True``, i.e. localization by simulator ground truth.
   Across two real trajectories there is none, so it is ``False`` here, which
   is the release's inferred-localization setting (``configs/tango.yaml``).
4. **Localizer state.** Upstream localizes consecutive frames of one robot
   walk and smooths the decision over the last 8 queries
   (``matchedRefNodeIndsHistory``). The queries here are a strided sweep over
   another trajectory, so by default (``adapter.localizer_state=per_query``)
   the history and the last localized index are cleared before each query, as
   if each were the first. ``sequential`` keeps upstream's running state.
5. **Keypoints are recorded, not recomputed.** A subclass of ``MatchLightGlue``
   forwards every call unchanged and keeps the matched keypoints
   ``matchPair_imgWithMask`` computed from the same features and matches.
6. **Precomputed maps.** A trajectory whose record names a ``graph`` is not
   re-mapped: that pickle is loaded exactly as upstream's own
   ``MapTopological.load_graph`` loads a precomputed graph (``pickle.load``),
   and the image list is derived as ``MapTopological.__init__`` derives it.
   The mapper pickles ``G4`` and uses that same object afterwards, so a map
   loaded back is the graph a fresh run would localize against. RAGMAP uses
   this to cache maps across runs (the caller keys the pickle by image sha,
   mapper overrides and the exact frames); the segmentor is loaded only when
   some trajectory still has to be mapped. Its rows in ``nodes.jsonl`` and
   ``edges.jsonl`` are emitted from the loaded graph as for a fresh map.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("ragmap_adapter")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ragmap.segment_graph/v1"
#: The release's working resolution (``scripts/create_maps_hm3d.py``, and the
#: simulator's 320 x 240 in ``configs/defaults.yaml``).
WIDTH, HEIGHT = 320, 240

#: ``scripts/create_maps_hm3d.py``'s mapper configuration, verbatim.
MAPPER_CFG: dict[str, Any] = {
    "W": WIDTH, "H": HEIGHT, "device": "cuda", "segmentor_name": "fast_sam", "modelPath": None,
    "force_recompute_masks": True, "match_area": True,
    "matcher_name": "lightglue",
    "remove_h5": True, "precompute_path_lengths": False, "edge_weight_str": None,
}

LOCALIZER_STATES = ("per_query", "sequential")


def _parse_overrides(items: list[str]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {"mapper": {}, "localizer": {}, "adapter": {}}
    for item in items:
        key, sep, raw = item.partition("=")
        section, dot, name = key.partition(".")
        if not sep or not dot or section not in sections or not name:
            raise SystemExit(f"Override {item!r} must be mapper.KEY=VALUE, localizer.KEY=VALUE or adapter.KEY=VALUE")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        sections[section][name] = value
    return sections


def _release_localizer_cfg() -> dict[str, Any]:
    """The release's RoboHop localizer and matcher settings (``configs/defaults.yaml``)."""

    import yaml

    with (REPO_ROOT / "configs" / "defaults.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    goal_gen = dict(config["goal_gen"])
    keys = ("matcher_name", "match_area", "loc_radius", "subsample_ref", "reloc_rad_add",
            "reloc_rad_max", "min_num_matches")
    return {key: goal_gen[key] for key in keys}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _node_id(trajectory_id: str, node: int) -> str:
    return f"{trajectory_id}/node/{int(node)}"


class _Trajectory:
    def __init__(self, record: dict[str, Any], input_dir: Path, output_dir: Path) -> None:
        self.id = str(record["id"])
        self.input_images = input_dir / record["images"]
        self.frames = [
            json.loads(line)
            for line in (input_dir / record["frames"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.frames.sort(key=lambda frame: int(frame["map_index"]))
        self.precomputed_graph = input_dir / record["graph"] if record.get("graph") else None
        self.work = output_dir / self.id
        self.images = self.work / "images"
        self.graph = None
        self.node_image = None  # node index -> map index
        self.image_paths: list[str] = []


def _build_map(trajectory: _Trajectory, segmentor: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from libs.mapper.map_topo import MapTopological

    trajectory.work.mkdir(parents=True, exist_ok=True)
    # The localizer writes its feature cache next to the image directory
    # (``Path(imgDir).parent``), and /input is read-only: link the directory in.
    if not trajectory.images.exists():
        os.symlink(trajectory.input_images, trajectory.images, target_is_directory=True)
    started = time.monotonic()
    mapper = MapTopological(str(trajectory.images), outDir=str(trajectory.work), cfg=dict(cfg), segmentor=segmentor)
    if len(mapper.imgNames) != len(trajectory.frames):
        raise RuntimeError(
            f"{trajectory.id}: {len(mapper.imgNames)} images but {len(trajectory.frames)} frames listed"
        )
    mapper.create_map_topo()
    trajectory.graph = mapper.G4
    trajectory.image_paths = list(mapper.imgNames)
    trajectory.node_image = np.array([trajectory.graph.nodes[n]["map"][0] for n in trajectory.graph.nodes()])
    return {"seconds": time.monotonic() - started, "graph": mapper.graphPath}


def _load_map(trajectory: _Trajectory, cfg: dict[str, Any]) -> dict[str, Any]:
    """A precomputed map: upstream's graph pickle, and the image list the mapper would list."""

    import pickle
    import shutil

    import numpy as np
    from libs.mapper.map_topo import MapTopological
    from natsort import natsorted

    trajectory.work.mkdir(parents=True, exist_ok=True)
    if not trajectory.images.exists():
        os.symlink(trajectory.input_images, trajectory.images, target_is_directory=True)
    started = time.monotonic()
    # `MapTopological.__init__`'s image list and subsampling, without loading
    # its segmentor and matcher (neither is used once the graph exists).
    settings = {**MapTopological.default_config(None), **cfg}
    names = [f"{trajectory.images}/{name}" for name in natsorted(os.listdir(f"{trajectory.images}"))]
    names = names[settings["subsample_si"]:settings["subsample_ei"]:settings["subsample_step"]]
    if len(names) != len(trajectory.frames):
        raise RuntimeError(f"{trajectory.id}: {len(names)} images but {len(trajectory.frames)} frames listed")
    # Where `MapTopological` writes it (`h5FullPath`, `graphPath`), so the
    # output tree is the one a fresh map leaves.
    h5 = f"{trajectory.work}/nodes_{settings['segmentor_name']}.h5"
    if len(settings["textLabels"]) > 0:
        h5 = f"{h5[:-3]}_filteredByText.h5"
    graph_path = f"{h5[:-3]}_graphObject_4_{settings['matcher_name']}.pickle"
    shutil.copyfile(trajectory.precomputed_graph, graph_path)
    with open(graph_path, "rb") as handle:  # `MapTopological.load_graph`
        trajectory.graph = pickle.load(handle)
    trajectory.image_paths = names
    trajectory.node_image = np.array([trajectory.graph.nodes[n]["map"][0] for n in trajectory.graph.nodes()])
    if len(trajectory.node_image) and int(trajectory.node_image.max()) >= len(names):
        raise RuntimeError(f"{trajectory.id}: the precomputed graph names images beyond the {len(names)} listed")
    return {"seconds": time.monotonic() - started, "graph": graph_path}


def _graph_rows(trajectory: _Trajectory) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    from libs.common.utils import rle_to_mask

    frames = {int(frame["map_index"]): frame for frame in trajectory.frames}
    nodes: list[dict[str, Any]] = []
    for node, data in trajectory.graph.nodes(data=True):
        image, segment = (int(value) for value in data["map"])
        rle = data["segmentation"]
        mask = rle_to_mask(rle)
        ys, xs = np.nonzero(mask)
        nodes.append({
            "id": _node_id(trajectory.id, node),
            "trajectory_id": trajectory.id,
            "node": int(node),
            "map_index": image,
            "frame_id": frames[image]["frame_id"],
            "segment": segment,
            "mask": {"size": [int(v) for v in rle["size"]], "counts": [int(v) for v in rle["counts"]]},
            # The mapper's own centroid formula (``create_graph_intra_img``).
            "centroid": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
            "area": int(data.get("area", int(mask.sum()))),
        })
    edges: list[dict[str, Any]] = []
    for u, v, data in trajectory.graph.edges(data=True):
        kind = "temporal" if data.get("edgeType") == "da" else "intra_image"
        row = {"kind": kind, "source": _node_id(trajectory.id, u), "target": _node_id(trajectory.id, v)}
        if kind == "temporal":
            row["upstream_edge_type"] = "da"
        edges.append(row)
    return nodes, edges


def _recording_matcher(inner: Any) -> Any:
    """``inner``, re-classed so each ``matchPair_imgWithMask`` call keeps its keypoints."""

    from libs.matcher import lightglue as matcher_lg
    from libs.matcher.LightGlue.lightglue.utils import rbd

    class RecordingMatchLightGlue(matcher_lg.MatchLightGlue):
        def __init__(self, wrapped: Any) -> None:  # noqa: D401 - no weights reloaded
            self.__dict__.update(wrapped.__dict__)
            self.records: dict[str, tuple[Any, Any]] = {}

        def matchPair_imgWithMask(self, imSrc, imTgt, nodesSrc, nodesTgt, visualize=False, matcher=None,
                                  extractor=None, ftSrc=None, ftTgt=None, lmatches=None):
            import numpy as np

            result = super().matchPair_imgWithMask(
                imSrc, imTgt, nodesSrc, nodesTgt, visualize, matcher, extractor, ftSrc, ftTgt, lmatches,
            )
            # The same arrays upstream derives from the same inputs (its lines
            # `kp1, kp2, matches = ...; mkp1, mkp2 = ...`). The localizer always
            # passes precomputed features and batched matches.
            if lmatches is not None and ftSrc is not None and ftTgt is not None and lmatches["matches"].shape[0]:
                src, tgt, matches = rbd(ftSrc), rbd(ftTgt), lmatches["matches"]
                mkp1 = src["keypoints"][matches[..., 0]].detach().cpu().numpy()
                mkp2 = tgt["keypoints"][matches[..., 1]].detach().cpu().numpy()
            else:
                mkp1 = mkp2 = np.zeros((0, 2), dtype=np.float32)
            self.records[str(imTgt)] = (mkp1, mkp2)
            return result

    return RecordingMatchLightGlue(inner)


def _localize_pair(
    source: _Trajectory, target: _Trajectory, queries: list[int], cfg: dict[str, Any], state: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    from libs.common.utils import rle_to_mask
    from libs.localizer.loc_topo import LocalizeTopological

    started = time.monotonic()
    localizer = LocalizeTopological(str(target.images), target.graph, WIDTH, HEIGHT, cfg=dict(cfg))
    recorder = _recording_matcher(localizer.matcher)
    localizer.matcher = recorder
    target_frames = {int(frame["map_index"]): frame for frame in target.frames}
    source_frames = {int(frame["map_index"]): frame for frame in source.frames}
    target_masks: dict[int, Any] = {}

    def target_mask(node: int):
        if node not in target_masks:
            target_masks[node] = rle_to_mask(target.graph.nodes[node]["segmentation"])
        return target_masks[node]

    edges: list[dict[str, Any]] = []
    localizations: list[dict[str, Any]] = []
    for query in queries:
        query_nodes = np.argwhere(source.node_image == query).flatten()
        record = {
            "source_trajectory_id": source.id, "target_trajectory_id": target.id,
            "query_map_index": int(query), "query_frame_id": source_frames[query]["frame_id"],
            "query_segments": int(len(query_nodes)),
        }
        if not len(query_nodes):
            localizations.append({**record, "lost": True, "localized_map_index": None, "reason": "no segments"})
            continue
        qry_nodes = [source.graph.nodes[int(n)] for n in query_nodes]
        # `Goal_Gen.get_goal_mask` clears `lost` before every observation.
        localizer.lost = False
        if state == "per_query":
            localizer.matchedRefNodeIndsHistory = []
            localizer.localizedImgIdx = 0
        recorder.records = {}
        match_pairs = np.asarray(localizer.localize(source.image_paths[query], qry_nodes))
        lost = bool(localizer.lost)
        votes: dict[int, int] = {}
        if match_pairs.size:
            for image in target.node_image[match_pairs[:, 1].astype(int)]:
                votes[int(image)] = votes.get(int(image), 0) + 1
        localized = None if lost else int(localizer.localizedImgIdx)
        localizations.append({
            **record, "lost": lost, "localized_map_index": localized,
            "localized_frame_id": target_frames[localized]["frame_id"] if localized is not None else None,
            "matched_pairs": int(len(match_pairs)) if match_pairs.size else 0,
            "votes_per_map_image": {str(key): value for key, value in sorted(votes.items())},
        })
        if localized is None or not match_pairs.size:
            continue
        mkp1, mkp2 = recorder.records.get(localizer.imgNames[localized], (np.zeros((0, 2)), np.zeros((0, 2))))
        for local, target_node in match_pairs.astype(int):
            if int(target.node_image[target_node]) != localized:
                continue
            source_node = int(query_nodes[local])
            source_mask = rle_to_mask(source.graph.nodes[source_node]["segmentation"])
            mask_t = target_mask(int(target_node))
            # `MatchLightGlue.map_node2kp`'s own pixel lookup (`astype(int)`).
            inside = np.zeros(len(mkp1), dtype=bool)
            if len(mkp1):
                inside = (source_mask[mkp1[:, 1].astype(int), mkp1[:, 0].astype(int)]
                          & mask_t[mkp2[:, 1].astype(int), mkp2[:, 0].astype(int)])
            keypoints = np.column_stack([mkp1[inside], mkp2[inside]]) if inside.any() else np.zeros((0, 4))
            edges.append({
                "kind": "cross_trajectory",
                "source": _node_id(source.id, source_node),
                "target": _node_id(target.id, int(target_node)),
                "source_frame_id": source_frames[query]["frame_id"],
                "target_frame_id": target_frames[localized]["frame_id"],
                "votes": int(inside.sum()),
                "keypoints": [[round(float(value), 2) for value in row] for row in keypoints],
            })
    summary = {
        "seconds": time.monotonic() - started,
        "queries": len(queries),
        "localized": sum(1 for row in localizations if row.get("localized_map_index") is not None),
        "cross_edges": len(edges),
        "loc_radius": localizer.reloc_dia_default // 2,
        "window_images": int(len(localizer.getRefImgInds())),
        "map_images": len(localizer.imgNames),
    }
    return edges, localizations, summary


def _versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version(),
                                "object_rel_nav_git_sha": os.environ.get("OBJECT_REL_NAV_GIT_SHA", "unknown")}
    for name in ("torch", "ultralytics", "kornia", "numpy", "networkx", "cv2", "scipy", "h5py"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - reported, never fatal
            versions[name] = f"unavailable: {exc}"
    try:
        import torch

        versions["cuda"] = torch.version.cuda
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover
        pass
    return versions


def run(input_dir: Path, output_dir: Path, overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from ragmap_adapter import stubs

    stubbed = stubs.install()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)  # upstream resolves `model_weights/` from the repo root
    from libs.experiments import model_loader

    meta = json.loads((input_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise RuntimeError(f"meta.json schema is {meta.get('schema')!r}, expected {SCHEMA!r}")
    state = str(overrides["adapter"].get("localizer_state", "per_query"))
    if state not in LOCALIZER_STATES:
        raise RuntimeError(f"adapter.localizer_state must be one of {LOCALIZER_STATES}, got {state!r}")
    mapper_cfg = {**MAPPER_CFG, **overrides["mapper"]}
    release_localizer = _release_localizer_cfg()
    timings: dict[str, Any] = {}
    started = time.monotonic()
    trajectories = {record["id"]: _Trajectory(record, input_dir, output_dir) for record in meta["trajectories"]}
    segmentor = None
    if any(trajectory.precomputed_graph is None for trajectory in trajectories.values()):
        # `create_maps_hm3d.py` builds one segmentor and hands it to every map.
        segmentor = model_loader.get_segmentor(
            mapper_cfg["segmentor_name"], mapper_cfg["W"], mapper_cfg["H"], mapper_cfg["device"],
            path_models=mapper_cfg.get("modelPath"),
        )
        timings["segmentor_load"] = time.monotonic() - started
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    maps: dict[str, Any] = {}
    for trajectory in trajectories.values():
        if trajectory.precomputed_graph is not None:
            info = _load_map(trajectory, mapper_cfg)
        else:
            info = _build_map(trajectory, segmentor, mapper_cfg)
        trajectory_nodes, trajectory_edges = _graph_rows(trajectory)
        nodes += trajectory_nodes
        edges += trajectory_edges
        maps[trajectory.id] = {
            "images": len(trajectory.frames), "nodes": len(trajectory_nodes),
            "intra_image_edges": sum(1 for edge in trajectory_edges if edge["kind"] == "intra_image"),
            "temporal_edges": sum(1 for edge in trajectory_edges if edge["kind"] == "temporal"),
            "graph_pickle": str(Path(info["graph"]).relative_to(output_dir)),
            "precomputed": trajectory.precomputed_graph is not None,
        }
        timings[f"map:{trajectory.id}"] = info["seconds"]
    localizations: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    localizer_cfgs: dict[str, Any] = {}
    for pair in meta.get("pairs", ()):
        source, target = trajectories[pair["source"]], trajectories[pair["target"]]
        cfg = {**release_localizer, "use_gt_localization": False}
        # The window: every image of the target map (see the module docstring).
        cfg["loc_radius"] = len(target.frames)
        cfg.update(overrides["localizer"])
        localizer_cfgs[f"{source.id}->{target.id}"] = cfg
        pair_edges, pair_localizations, summary = _localize_pair(
            source, target, [int(value) for value in pair["queries"]], cfg, state,
        )
        edges += pair_edges
        localizations += pair_localizations
        pairs.append({"source": source.id, "target": target.id, **summary})
        timings[f"localize:{source.id}->{target.id}"] = summary["seconds"]
    _write_jsonl(output_dir / "nodes.jsonl", nodes)
    _write_jsonl(output_dir / "edges.jsonl", edges)
    _write_jsonl(output_dir / "localizations.jsonl", localizations)
    timings["total"] = time.monotonic() - started
    return {
        "counts": {
            "trajectories": len(trajectories), "nodes": len(nodes),
            "edges": {kind: sum(1 for edge in edges if edge["kind"] == kind)
                      for kind in ("intra_image", "temporal", "cross_trajectory")},
            "queries": len(localizations),
            "localized": sum(1 for row in localizations if row.get("localized_map_index") is not None),
            "maps": maps, "pairs": pairs,
        },
        "timings": timings,
        "mask_frame": {"width": WIDTH, "height": HEIGHT},
        "config": {
            "mapper": mapper_cfg, "localizer_release": release_localizer, "localizer": localizer_cfgs,
            "localizer_state": state, "stubbed_modules": stubbed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragmap-run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="mapper.KEY=VALUE, localizer.KEY=VALUE, adapter.KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    overrides = _parse_overrides(args.overrides)
    args.output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema": SCHEMA, "status": "failed", "error": None, "versions": _versions()}
    try:
        payload.update(run(args.input.resolve(), args.output.resolve(), overrides))
        payload["status"] = "ok"
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        traceback.print_exc()
    (args.output / "run.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: payload.get(key) for key in ("status", "error", "counts")}, default=str)[:4000])
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
