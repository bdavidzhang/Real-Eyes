"""
Open-set object detection wrapper for VGGT-SLAM 2.0.

Composes existing PE-Core CLIP + SAM3 functions into a single class.
Uses VGGT-SLAM 2.0 APIs (submap.get_points_in_mask(frame_idx, mask, graph),
submap.get_all_semantic_vectors(), etc.).
"""

import base64
import cv2
import numpy as np
import torch
import open3d as o3d
from PIL import Image

from vggt_slam.slam_utils import compute_text_embeddings, compute_obb_from_points, overlay_masks


class ObjectDetector:
    """
    Encapsulates PE-Core CLIP + SAM3 for open-set 3D object detection.

    Usage:
        od = ObjectDetector(device="cuda")
        text_emb = od.encode_text("chair")
        masks, boxes, scores = od.segment(image_pil, "chair")
        bbox = od.compute_3d_bbox(submap, frame_idx, mask, graph, scene_center)
    """

    def __init__(self, device="cuda", clip_model_name="PE-Core-L14-336",
                 sam3_confidence_threshold=0.30):
        self.device = device

        # Load PE-Core CLIP
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        self.clip_model = pe.CLIP.from_config(clip_model_name, pretrained=True)
        self.clip_model.eval()
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = pe_transforms.get_text_tokenizer(self.clip_model.context_length)
        self.clip_preprocess = pe_transforms.get_image_transform(self.clip_model.image_size)

        # Load SAM3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(sam3_model, confidence_threshold=sam3_confidence_threshold)

    def encode_text(self, query):
        """Encode a text query into a CLIP embedding. Returns (1, D) numpy array."""
        return compute_text_embeddings(self.clip_model, self.clip_tokenizer, query)

    @torch.no_grad()
    def encode_text_vector(self, query):
        """Encode a text query into a (D,) CPU tensor for dot-product matching."""
        text_tokens = self.clip_tokenizer([query]).to(self.device)
        text_emb = self.clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.float().cpu().squeeze(0)

    def segment(self, image_pil, query):
        """Run SAM3 text-prompted segmentation on an image.

        Args:
            image_pil: PIL Image
            query: text prompt string

        Returns:
            (masks, boxes, scores) tensors, or (None, None, None) if nothing detected.
        """
        with torch.no_grad():
            inference_state = self.sam3_processor.set_image(image_pil)
            output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=query)
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or len(masks) == 0:
            return None, None, None
        return masks, boxes, scores

    def segment_all(self, image_pil, query):
        """Run SAM3 and return list of (mask_2d, box_2d, score) tuples."""
        masks, boxes, scores = self.segment(image_pil, query)
        if masks is None:
            return []
        results = []
        for i in range(len(scores)):
            mask_2d = masks[i, 0].cpu().numpy() if masks.dim() == 4 else masks[i].cpu().numpy()
            box_2d = boxes[i].cpu().numpy()
            score = scores[i].item()
            results.append((mask_2d, box_2d, score))
        return results

    def compute_3d_bbox(self, submap, frame_idx, mask, graph, scene_center):
        """Compute a 3D oriented bounding box from a 2D mask.

        Uses submap.get_points_in_mask(frame_idx, mask, graph) to get world-frame points,
        then computes OBB via compute_obb_from_points().

        Args:
            submap: Submap object
            frame_idx: frame index within the submap
            mask: 2D boolean mask (H, W)
            graph: PoseGraph instance
            scene_center: (3,) array for recentering

        Returns:
            dict with center, extent, rotation, corners (all recentered), or None.
        """
        points_world = submap.get_points_in_mask(frame_idx, mask, graph)
        if points_world is None or len(points_world) < 10:
            return None

        # Recenter to match viewer coordinates
        points_recentered = points_world - scene_center

        # Remove outliers
        if len(points_recentered) > 50:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_recentered)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            points_recentered = np.asarray(pcd.points)

        if len(points_recentered) < 10:
            return None

        try:
            center, extent, rotation = compute_obb_from_points(points_recentered)
        except ValueError:
            return None

        # Compute 8 corner points
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
        ])
        corners_world = (rotation @ corners_local.T).T + center

        return {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "rotation": rotation.tolist(),
            "corners": corners_world.tolist(),
        }

    @staticmethod
    def deduplicate_detections(detections):
        """Remove overlapping OBB duplicates for the same query.

        Keeps the higher-confidence detection when two OBBs for the same
        query overlap (half-extent sum test along each axis).
        """
        if len(detections) <= 1:
            return [d for d in detections if d.get("success") and d.get("bounding_box")]

        keep = []
        for det in sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True):
            if not det.get("success") or not det.get("bounding_box"):
                continue
            center = np.array(det["bounding_box"]["center"])
            half_ext = np.array(det["bounding_box"]["extent"]) / 2.0

            is_dup = False
            for kept in keep:
                if kept["query"] != det["query"]:
                    continue
                kept_center = np.array(kept["bounding_box"]["center"])
                kept_half = np.array(kept["bounding_box"]["extent"]) / 2.0
                diff = np.abs(center - kept_center)
                if np.all(diff < (half_ext + kept_half)):
                    is_dup = True
                    break
            if not is_dup:
                keep.append(det)
        return keep

    @staticmethod
    def image_to_base64(image_np):
        """Convert RGB numpy image to base64-encoded JPEG string."""
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def mask_overlay_to_base64(image_np, mask):
        """Create a mask overlay visualization and return as base64 PNG."""
        overlay = image_np.copy()
        color = np.array([0, 255, 100], dtype=np.uint8)
        overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 100), 2)
        _, buffer = cv2.imencode('.png', overlay_bgr)
        return base64.b64encode(buffer).decode('utf-8')
