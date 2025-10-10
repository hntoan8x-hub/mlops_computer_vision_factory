# cv_factory/shared_libs/ml_core/evaluator/metrics/detection_metrics.py

import numpy as np
import logging
from typing import Dict, Any, Union, List, Tuple, Optional

# Import Base Abstraction
from ..base.base_metric import BaseMetric, MetricValue, InputData

logger = logging.getLogger(__name__)

# Utility function for calculating Intersection over Union (IoU)
def calculate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculates IoU between two bounding boxes (xyxy format)."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # Compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the intersection over union
    iou = inter_area / (box_a_area + box_b_area - inter_area)
    return iou

class mAPMetric(BaseMetric):
    """
    A stateful metric class for calculating mean Average Precision (mAP) 
    over a set of Intersection over Union (IoU) thresholds.

    Accumulates predictions and ground truths across all batches for a global 
    precision-recall curve calculation.
    """
    
    def __init__(self, name: str = 'mAP', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the mAP Metric.

        Config requires 'iou_thresholds' (e.g., [0.5, 0.75]) and 'num_classes'.
        """
        super().__init__(name, config)
        self.iou_thresholds = self.config.get('iou_thresholds', [0.5]) # Default mAP@0.5
        self.num_classes = self.config.get('num_classes', 1)
        self.reset()

    def reset(self) -> None:
        """
        Resets the accumulated state. 
        State accumulates all predictions and ground truths for global calculation.
        """
        self._internal_state = {
            'all_preds': [],  # List of dicts: [{'score': float, 'box': np.ndarray, 'class': int, 'match': bool}, ...]
            'all_gts': [],    # List of dicts: [{'box': np.ndarray, 'class': int, 'used': bool}, ...]
            'total_gt_count': 0 # Total number of ground truths across all classes
        }
        self._is_initialized = True
        logger.debug(f"Metric '{self.name}' state reset. IoU thresholds: {self.iou_thresholds}")

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Accumulates predictions and ground truths for a single batch/image.

        Args:
            predictions (List[Dict]): List of prediction dicts: [{'box': np.ndarray, 'score': float, 'class': int}]
            targets (List[Dict]): List of ground truth dicts: [{'box': np.ndarray, 'class': int}]
        """
        if not self._is_initialized:
            self.reset()
            
        # 1. Accumulate Predictions
        for pred in predictions:
            # Standardize pred format before storage
            self._internal_state['all_preds'].append({
                'score': float(pred['score']),
                'box': np.array(pred['box']),
                'class': int(pred['class']),
                'match': False  # Placeholder for matching status during calculation
            })

        # 2. Accumulate Ground Truths
        for target in targets:
            self._internal_state['all_gts'].append({
                'box': np.array(target['box']),
                'class': int(target['class']),
                'used': False # Placeholder to ensure each GT is matched once
            })
            
        self._internal_state['total_gt_count'] += len(targets)


    def _calculate_ap_single_class(self, class_id: int, iou_thresh: float) -> float:
        """
        Calculates Average Precision (AP) for a single class and IoU threshold.
        """
        # Filter predictions and ground truths for the specific class
        class_preds = sorted([p for p in self._internal_state['all_preds'] if p['class'] == class_id], 
                             key=lambda x: x['score'], reverse=True)
        class_gts = [g for g in self._internal_state['all_gts'] if g['class'] == class_id]
        
        # Reset GT 'used' status for this class and threshold run
        for gt in class_gts:
            gt['used'] = False
            
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        # 1. Match Predictions to GTs
        for i, pred in enumerate(class_preds):
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find the best matching, unused GT
            for gt_idx, gt in enumerate(class_gts):
                if not gt['used']:
                    iou = calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # 2. Assign TP or FP
            if best_iou >= iou_thresh and best_gt_idx != -1:
                # True Positive: Found a match above IoU threshold
                tp[i] = 1
                class_gts[best_gt_idx]['used'] = True # Mark GT as used
            else:
                # False Positive: No match, or match was below threshold
                fp[i] = 1

        # 3. Calculate Cumulative TP, FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Total number of ground truths for this class
        num_gt = len(class_gts)
        if num_gt == 0:
            return 0.0

        # 4. Calculate Precision and Recall
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 5. Calculate Average Precision (AP) using the 11-point or all-point method
        # Using the standard all-point interpolation method (PASCAL VOC/COCO style)
        ap = 0.0
        # Insert (0, 1) and (1, 0) points for interpolation boundaries
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Find where recall changes (i.e., TP occurs)
        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Sum areas: sum (mrec[i+1] - mrec[i]) * mpre[i+1]
        ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
        
        return ap

    def compute(self) -> MetricValue:
        """
        Calculates the final mAP value averaged over all classes and configured IoU thresholds.
        """
        if self._internal_state['total_gt_count'] == 0:
            logger.warning(f"Attempted to compute mAP with zero ground truths.")
            return 0.0
        
        results: Dict[str, float] = {}
        all_aps: List[float] = []

        # Iterate over all defined IoU thresholds
        for iou_thresh in self.iou_thresholds:
            class_aps: List[float] = []
            
            # Iterate over all classes
            for class_id in range(self.num_classes):
                ap = self._calculate_ap_single_class(class_id, iou_thresh)
                class_aps.append(ap)
                
            # Calculate mAP for the specific threshold
            map_at_thresh = np.mean(class_aps)
            results[f"mAP@{iou_thresh:.2f}"] = float(map_at_thresh)
            all_aps.append(map_at_thresh)

        # Calculate the final mAP (average over all thresholds, if multiple are defined)
        if len(all_aps) > 0:
            results[self.name] = float(np.mean(all_aps))
        
        # NOTE: If COCO-style mAP is required (mAP@[.5:.05:.95]), 
        # the iou_thresholds must be set accordingly in config.
            
        return results