# RoboHop / object-rel-nav (original release) + RAGMAP adapter, as the
# correspondence engine of RAGMAP's `robohop_alignment` scene-alignment module.
#
# The container builds RoboHop's topological segment map per trajectory
# (libs/mapper/map_topo.py: FastSAM-s + Delaunay + SuperPoint/LightGlue) and
# localizes one trajectory's images against another's map with the release's
# LocalizeTopological. Upstream code is copied unmodified; see ragmap_adapter/.
#
# habitat-sim is deliberately NOT installed: libs/common/utils.py imports it at
# module level, but nothing the mapper/localizer calls uses it, so the adapter
# registers loud stubs for it (ragmap_adapter/stubs.py).
#
# Weights are baked in at build time: FastSAM-s (ultralytics assets) at
# model_weights/FastSAM-s.pt, where fast_sam_module.py looks for it, and the
# SuperPoint / LightGlue checkpoints in the torch hub cache.
FROM python:3.10-slim

ARG OBJECT_REL_NAV_GIT_SHA=unknown
ARG FASTSAM_URL=https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt
ENV OBJECT_REL_NAV_ROOT=/opt/object-rel-nav \
    OBJECT_REL_NAV_GIT_SHA=${OBJECT_REL_NAV_GIT_SHA} \
    PYTHONPATH=/opt/object-rel-nav \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch-hub \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ragmap_adapter/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
        torch==2.3.1 torchvision==0.18.1 \
    && printf 'torch==2.3.1\ntorchvision==0.18.1\n' > /tmp/constraints.txt \
    && pip install --constraint /tmp/constraints.txt -r /tmp/requirements.txt

WORKDIR /opt/object-rel-nav
COPY . /opt/object-rel-nav
RUN mkdir -p model_weights \
    && curl --fail --location --silent --show-error -o model_weights/FastSAM-s.pt "${FASTSAM_URL}" \
    && printf '#!/bin/sh\nexec python -m ragmap_adapter.run "$@"\n' > /usr/local/bin/ragmap-run \
    && chmod +x /usr/local/bin/ragmap-run \
    && python -c "import torch; assert torch.version.cuda == '12.1', torch.version.cuda; \
from ragmap_adapter import stubs; print('stubbed', stubs.install()); \
from libs.mapper.map_topo import MapTopological; from libs.localizer.loc_topo import LocalizeTopological; \
from libs.matcher.LightGlue.lightglue import SuperPoint, LightGlue; \
SuperPoint(max_num_keypoints=2048, detection_threshold=0.0); LightGlue(features='superpoint'); \
from libs.segmentor.fast_sam_module import FastSamClass; \
FastSamClass({'width': 320, 'height': 240, 'mask_height': 240, 'mask_width': 320, 'conf': 0.5, \
'model': 'FastSAM-s.pt', 'imgsz': 480}, device='cpu'); \
from ragmap_adapter.run import main; print('imports and weights ok')" \
    && ls -la "${TORCH_HOME}/hub/checkpoints" model_weights \
    && ragmap-run --help >/dev/null

WORKDIR /work
ENTRYPOINT ["ragmap-run"]
CMD ["--help"]
