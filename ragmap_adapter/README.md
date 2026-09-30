# RAGMAP adapter

`ragmap-run --input /input --output /output [mapper.KEY=VALUE] [localizer.KEY=VALUE] [adapter.localizer_state=per_query|sequential]`

Runs this release's RoboHop segment mapper (`libs/mapper/map_topo.py`, configured
as `scripts/create_maps_hm3d.py` does) on each input trajectory, then localizes
chosen images of one trajectory against another's map with
`libs/localizer/loc_topo.py`'s `LocalizeTopological` (the release's RoboHop
`configs/defaults.yaml` values, with the window widened to the whole target map
and no ground-truth oracle). It writes the segment graphs and the
cross-trajectory segment associations, with the LightGlue keypoints supporting
each one. RAGMAP's `robohop_alignment` scene-alignment module lifts those to 3-D
and fits a planar transform between the trajectories.

A trajectory may name a `graph` pickle this adapter wrote earlier for exactly the
same images and mapper settings; it is then loaded instead of re-mapped (RAGMAP
caches maps this way), and FastSAM is loaded only when some trajectory still
needs mapping.

Upstream code is not modified. The contract (`ragmap.segment_graph/v1`) and
every adapter decision are documented in `ragmap_adapter/run.py`'s docstring.

Image: `ghcr.io/velythyl/object-rel-nav:sha-<short>`, built by
`.github/workflows/ghcr.yml`. FastSAM-s and SuperPoint/LightGlue weights are
baked in. Needs one CUDA GPU (upstream hard-codes `cuda`).

Licensing: the upstream repository has no LICENSE file at its root, and
`libs/segmentor/fast_sam_module.py` is marked GPL-3.0. This fork is for
research reproduction only.
