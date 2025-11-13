# shared_libs/ml_core/retraining/tmr_facade.py (NEW FILE)

import logging
from typing import Dict, Any, List, Tuple
from shared_libs.ml_core.monitoring.orchestrator.monitoring_orchestrator import MonitoringOrchestrator
from shared_libs.ml_core.retraining.orchestrator.retrain_orchestrator import RetrainOrchestrator

logger = logging.getLogger(__name__)

class TMRFacade:
    """
    Façade (Glue Layer) chịu trách nhiệm điều phối luồng Trigger-Monitoring-Retraining (TMR).
    
    Nó thực thi: Monitoring -> Thu thập Report -> Truyền Report cho Retrain Orchestrator.
    """

    def __init__(self, 
                 monitoring_orchestrator: MonitoringOrchestrator, 
                 retrain_orchestrator: RetrainOrchestrator):
        
        self.monitor = monitoring_orchestrator
        self.retrain = retrain_orchestrator
        logger.info("TMRFacade (Glue) initialized with Monitor and Retrain Orchestrators.")

    def run_tmr_workflow(self, reference_data: Any, current_data: Any) -> Tuple[bool, List[str]]:
        """
        Thực hiện toàn bộ luồng TMR:
        1. Chạy Monitoring.
        2. Trích xuất các báo cáo cần thiết (Drift, Metrics).
        3. Chạy Retrain Orchestrator với các báo cáo này.
        """
        logger.info("Starting TMR Workflow: Monitoring Phase.")
        
        # 1. MONITORING: Chạy tất cả các Monitor
        full_report = self.monitor.run_check(reference_data, current_data)
        
        # 2. GLUE: Trích xuất các báo cáo cần thiết cho Triggers
        # Trích xuất báo cáo Drift và Metrics cho Retrain Orchestrator
        
        trigger_kwargs: Dict[str, Any] = {}
        drift_report = next((r for r in full_report if 'drift_detected' in r), None)
        
        # Lấy metrics/scores từ các monitor (giả định MetricMonitor)
        metric_reports = [r for r in full_report if 'score' in r and 'metric' in r.get('monitor', '').lower()]
        
        # Định dạng lại metrics cho PerformanceTrigger
        current_metrics = {}
        for report in metric_reports:
             # Ví dụ: metric_monitor_accuracy -> accuracy
             metric_name = report.get('monitor', '').split('_')[-1] 
             if metric_name:
                  current_metrics[metric_name] = report.get('score')

        trigger_kwargs['drift_report'] = drift_report
        trigger_kwargs['current_metrics'] = current_metrics
        
        # 3. RETRAINING: Chạy luồng kiểm tra Triggers
        logger.info("Starting TMR Workflow: Retraining Trigger Check Phase.")
        
        # Retrain Orchestrator sẽ tự quyết định submit job nếu trigger fire
        self.retrain.run(**trigger_kwargs)
        
        # NOTE: Giả định Retrain Orchestrator đã log trạng thái.
        
        # Trả về kết quả kiểm tra (tùy chọn)
        is_triggered = self.retrain.log_payload.get('status') == 'submitted'
        reasons = self.retrain.log_payload.get('reasons', [])
        
        return is_triggered, reasons