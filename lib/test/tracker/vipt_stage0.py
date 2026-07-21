from dataclasses import dataclass
import math

import numpy as np
import torch

from lib.test.tracker.vipt import ViPTTrack
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


@dataclass(frozen=True)
class TemplateSnapshot:
    template_id: str
    source: str
    source_frame: int
    source_bbox_xywh: tuple
    resize_factor: float
    z_patch_arr: np.ndarray
    z_tensor: torch.Tensor
    box_mask_z: object

    def clone(self):
        return TemplateSnapshot(
            template_id=self.template_id,
            source=self.source,
            source_frame=self.source_frame,
            source_bbox_xywh=self.source_bbox_xywh,
            resize_factor=self.resize_factor,
            z_patch_arr=self.z_patch_arr.copy(),
            z_tensor=self.z_tensor.clone(),
            box_mask_z=self.box_mask_z.clone() if self.box_mask_z is not None else None,
        )


class ViPTStage0Track(ViPTTrack):
    def initialize(self, image, info):
        output = super().initialize(image, info)
        bbox = list(info["init_bbox"][:4])
        metadata_bbox = self._validate_bbox(bbox)
        crop_size = math.ceil(math.sqrt(bbox[2] * bbox[3]) * self.params.template_factor)
        snapshot = TemplateSnapshot(
            template_id=self._template_id("initial", 0, metadata_bbox),
            source="initial",
            source_frame=0,
            source_bbox_xywh=metadata_bbox,
            resize_factor=float(self.params.template_size / crop_size),
            z_patch_arr=self.z_patch_arr.copy(),
            z_tensor=self.z_tensor.clone(),
            box_mask_z=self.box_mask_z.clone() if self.box_mask_z is not None else None,
        )
        self.initial_template_snapshot = snapshot
        self.commit_template(snapshot)
        return output

    @staticmethod
    def _validate_bbox(bbox):
        metadata_bbox = tuple(float(value) for value in bbox)
        if (len(bbox) != 4 or not np.isfinite(metadata_bbox).all()
                or metadata_bbox[2] <= 0 or metadata_bbox[3] <= 0):
            raise ValueError(f"Invalid template bbox: {metadata_bbox}")
        return metadata_bbox

    @staticmethod
    def _template_id(source, source_frame, bbox):
        return f"{source}:{int(source_frame)}:" + ",".join(
            f"{value:.6f}" for value in bbox)

    def build_template_snapshot(self, image, bbox, source, source_frame):
        crop_bbox = list(bbox[:4])
        metadata_bbox = self._validate_bbox(crop_bbox)
        z_patch_arr, resize_factor, _ = sample_target(
            image, crop_bbox, self.params.template_factor, output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                crop_bbox, resize_factor, template.device).squeeze(1)
            box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
        return TemplateSnapshot(
            template_id=self._template_id(source, source_frame, metadata_bbox),
            source=str(source),
            source_frame=int(source_frame),
            source_bbox_xywh=metadata_bbox,
            resize_factor=float(resize_factor),
            z_patch_arr=z_patch_arr.copy(),
            z_tensor=template,
            box_mask_z=box_mask_z,
        )

    def commit_template(self, snapshot):
        if not isinstance(snapshot, TemplateSnapshot):
            raise TypeError("snapshot must be a TemplateSnapshot")
        active = snapshot.clone()
        self.active_template_snapshot = active
        self.z_patch_arr = active.z_patch_arr
        self.z_tensor = active.z_tensor
        self.box_mask_z = active.box_mask_z

    def rollback_to_initial(self):
        if not hasattr(self, "initial_template_snapshot"):
            raise RuntimeError("Tracker must be initialized before rollback")
        self.commit_template(self.initial_template_snapshot)

    def predict_with_context(self, image, search_anchor, template_snapshot):
        if not isinstance(template_snapshot, TemplateSnapshot):
            raise TypeError("template_snapshot must be a TemplateSnapshot")
        height, width, _ = image.shape
        anchor = [float(value) for value in search_anchor[:4]]
        x_patch_arr, resize_factor, _ = sample_target(
            image, anchor, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            output = self.network.forward(
                template=template_snapshot.z_tensor,
                search=search,
                ce_template_mask=template_snapshot.box_mask_z,
            )
        response = self.output_window * output["score_map"]
        pred_boxes, best_score = self.network.box_head.cal_bbox(
            response, output["size_map"], output["offset_map"], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        target_bbox = clip_box(
            self.map_box_back_from_anchor(pred_box, resize_factor, anchor),
            height,
            width,
            margin=10,
        )
        return {
            "target_bbox": target_bbox,
            "best_score": best_score[0][0].item(),
            "state_before": anchor,
            "search_anchor": anchor,
            "template_id": template_snapshot.template_id,
        }

    def map_box_back_from_anchor(self, pred_box, resize_factor, search_anchor):
        cx_prev = search_anchor[0] + 0.5 * search_anchor[2]
        cy_prev = search_anchor[1] + 0.5 * search_anchor[3]
        cx, cy, width, height = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + cx_prev - half_side
        cy_real = cy + cy_prev - half_side
        return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]


def get_tracker_class():
    return ViPTStage0Track
