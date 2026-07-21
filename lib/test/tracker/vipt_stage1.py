import math

import torch

from lib.test.tracker.vipt_stage0 import ViPTStage0Track
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class ViPTStage1Track(ViPTStage0Track):
    @staticmethod
    def response_statistics(response, pred_boxes, topk=20):
        values = response.detach().float().flatten(1)
        probabilities = values.clamp_min(0)
        totals = probabilities.sum(dim=1, keepdim=True)
        probabilities = torch.where(
            totals > 0,
            probabilities / totals.clamp_min(torch.finfo(probabilities.dtype).eps),
            torch.full_like(probabilities, 1.0 / probabilities.shape[1]),
        )
        entropy = -(probabilities * probabilities.clamp_min(
            torch.finfo(probabilities.dtype).eps).log()).sum(dim=1)
        entropy = entropy / math.log(probabilities.shape[1])
        top_scores, _ = torch.topk(values, k=min(int(topk), values.shape[1]), dim=1)
        margin = (top_scores[:, 0] - top_scores[:, 1]
                  if top_scores.shape[1] > 1 else top_scores[:, 0])
        centers = pred_boxes[..., :2]
        center_mean = centers.mean(dim=1, keepdim=True)
        box_dispersion = torch.sqrt(
            ((centers - center_mean) ** 2).sum(dim=-1).mean(dim=1)
        )
        diagnostics = torch.stack((
            values.max(dim=1).values,
            entropy,
            margin,
            top_scores.std(dim=1, unbiased=False),
            box_dispersion,
        ), dim=1)[0].detach().cpu().tolist()
        return dict(zip((
            "response_peak",
            "response_entropy",
            "response_margin",
            "response_topk_score_std",
            "response_topk_box_dispersion",
        ), (float(value) for value in diagnostics)))

    def predict_with_context(self, image, search_anchor, template_snapshot):
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
        topk_boxes, _ = self.network.box_head.cal_bbox_topk(
            response, output["size_map"], output["offset_map"], k=20, return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        target_bbox = clip_box(
            self.map_box_back_from_anchor(pred_box, resize_factor, anchor),
            height,
            width,
            margin=10,
        )
        diagnostics = self.response_statistics(response, topk_boxes)
        return {
            "target_bbox": target_bbox,
            "best_score": best_score[0][0].item(),
            "state_before": anchor,
            "search_anchor": anchor,
            "template_id": template_snapshot.template_id,
            **diagnostics,
        }


def get_tracker_class():
    return ViPTStage1Track
