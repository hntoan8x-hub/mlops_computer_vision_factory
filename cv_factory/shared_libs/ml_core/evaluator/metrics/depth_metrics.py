# cv_factory/shared_libs/ml_core/evaluator/metrics/depth_metrics.py

import numpy as np
import logging
from typing import Dict, Any, Union, List, Optional

# Import Base Abstraction
from ..base.base_metric import BaseMetric, MetricValue, InputData

logger = logging.getLogger(__name__)

# --- Utility Functions for Core Depth Metrics ---

def calculate_depth_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculates standard depth estimation metrics (AbsRel, SqRel, RMSE) for a batch.
    
    Args:
        pred (np.ndarray): Predicted depth map (masked).
        target (np.ndarray): Ground truth depth map (masked).
        mask (np.ndarray): Mask of valid pixels (where target > 0).

    Returns:
        Dict: Dictionary of per-sample/batch metrics.
    """
    diff = pred[mask] - target[mask]
    abs_diff = np.abs(diff)
    
    # 1. Absolute Relative Difference (AbsRel)
    abs_rel = np.mean(abs_diff / target[mask])
    
    # 2. Squared Relative Difference (SqRel)
    sq_rel = np.mean(diff**2 / target[mask]**2)
    
    # 3. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(diff**2))

    return {
        "abs_rel_sum": abs_rel * np.sum(mask), # Sum weighted by valid pixels
        "sq_rel_sum": sq_rel * np.sum(mask),
        "rmse_sum": rmse * np.sum(mask),
        "valid_pixels": np.sum(mask)
    }

def calculate_threshold_accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float) -> int:
    """
    Calculates the number of accurate pixels for a given threshold (delta_n).
    
    Accuracy is met if: max(y/y_gt, y_gt/y) < threshold
    
    Args:
        pred (np.ndarray): Predicted depth map.
        target (np.ndarray): Ground truth depth map.
        mask (np.ndarray): Mask of valid pixels.
        threshold (float): The delta value (e.g., 1.25, 1.25^2, 1.25^3).
        
    Returns:
        int: Total number of pixels satisfying the threshold condition.
    """
    # Chỉ tính trên các pixel hợp lệ
    ratio = np.maximum(pred[mask] / target[mask], target[mask] / pred[mask])
    
    correct_pixels = np.sum(ratio < threshold)
    return correct_pixels


# --- Stateful Metric Classes ---

class AbsRelMetric(BaseMetric):
    """
    Calculates the Mean Absolute Relative Difference (AbsRel), SqRel, and RMSE.
    Accumulates the SUM of weighted metrics and total valid pixels.
    """
    
    def __init__(self, name: str = 'AbsRel', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.reset()

    def reset(self) -> None:
        """Resets accumulated sums and pixel count."""
        self._internal_state = {
            'abs_rel_sum_total': 0.0,
            'sq_rel_sum_total': 0.0,
            'rmse_sum_total': 0.0,
            'total_valid_pixels': 0.0
        }
        self._is_initialized = True

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state with a batch of predicted and ground truth depth maps.

        Args:
            predictions (np.ndarray): Predicted depth map(s) [B, H, W] or [B, 1, H, W].
            targets (np.ndarray): Ground truth depth map(s) [B, H, W] or [B, 1, H, W].
        """
        if not self._is_initialized: self.reset()
            
        pred_batch = np.array(predictions).squeeze() # [B, 1, H, W] -> [B, H, W]
        target_batch = np.array(targets).squeeze()
        
        # Xử lý từng mẫu trong batch
        for pred, target in zip(pred_batch, target_batch):
            # Mask: Loại bỏ các pixel không hợp lệ (thường là target=0 hoặc target=inf)
            mask = target > 1e-3
            
            if np.sum(mask) == 0:
                continue
                
            metrics_batch = calculate_depth_metrics(pred, target, mask)
            
            self._internal_state['abs_rel_sum_total'] += metrics_batch['abs_rel_sum']
            self._internal_state['sq_rel_sum_total'] += metrics_batch['sq_rel_sum']
            self._internal_state['rmse_sum_total'] += metrics_batch['rmse_sum']
            self._internal_state['total_valid_pixels'] += metrics_batch['valid_pixels']

    def compute(self) -> MetricValue:
        """Calculates the final mean metrics across all accumulated samples."""
        total_pixels = self._internal_state['total_valid_pixels']
        
        if total_pixels == 0:
            return 0.0
            
        final_abs_rel = self._internal_state['abs_rel_sum_total'] / total_pixels
        final_sq_rel = self._internal_state['sq_rel_sum_total'] / total_pixels
        final_rmse = self._internal_state['rmse_sum_total'] / total_pixels
        
        return {
            self.name: float(final_abs_rel),
            "SqRel": float(final_sq_rel),
            "RMSE": float(final_rmse) # Lưu RMSE vào đây để dễ tracking
        }

class ThresholdAccuracyMetric(BaseMetric):
    """
    Calculates threshold accuracy metrics (Delta_1, Delta_2, Delta_3).
    """
    
    def __init__(self, name: str = 'Delta_Accuracy', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        # Các ngưỡng tiêu chuẩn
        self.thresholds = self.config.get('thresholds', [1.25, 1.25**2, 1.25**3]) 
        self.reset()

    def reset(self) -> None:
        """Resets accumulated correct pixels and total valid pixels for all thresholds."""
        self._internal_state = {
            f'correct_pixels_delta_{i+1}': 0 for i in range(len(self.thresholds))
        }
        self._internal_state['total_valid_pixels'] = 0.0
        self._is_initialized = True

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state with a batch of depth maps.
        """
        if not self._is_initialized: self.reset()
            
        pred_batch = np.array(predictions).squeeze()
        target_batch = np.array(targets).squeeze()
        
        # Xử lý từng mẫu trong batch
        for pred, target in zip(pred_batch, target_batch):
            mask = target > 1e-3
            
            if np.sum(mask) == 0:
                continue
                
            # Cập nhật tổng số pixel hợp lệ
            self._internal_state['total_valid_pixels'] += np.sum(mask)
            
            # Tính toán và tích lũy correct pixels cho từng ngưỡng
            for i, thresh in enumerate(self.thresholds):
                correct_pixels = calculate_threshold_accuracy(pred, target, mask, thresh)
                self._internal_state[f'correct_pixels_delta_{i+1}'] += correct_pixels

    def compute(self) -> MetricValue:
        """Calculates the final Delta Accuracy scores."""
        total_pixels = self._internal_state['total_valid_pixels']
        
        if total_pixels == 0:
            return 0.0
            
        results: Dict[str, float] = {}
        for i in range(len(self.thresholds)):
            key_count = f'correct_pixels_delta_{i+1}'
            key_result = f'Delta_{i+1}'
            
            accuracy = self._internal_state[key_count] / total_pixels
            results[key_result] = float(accuracy)
            
            if i == 0:
                 # Đặt Delta_1 làm giá trị chính cho MetricValue
                 results[self.name] = float(accuracy) 

        return results