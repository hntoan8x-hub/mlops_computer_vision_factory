# cv_factory/shared_libs/ml_core/monitoring/orchestrator/monitoring_orchestrator.py

import logging
import time
import importlib
from typing import Dict, Any, List, Union, Type, Optional

# --- Import Contracts and Utilities ---
from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor
from shared_libs.ml_core.monitoring.base.base_reporter import BaseReporter
from shared_libs.ml_core.monitoring.configs.monitoring_config_schema import MonitoringConfig, MonitorConfig, ReporterConfig
from shared_libs.core_utils.exceptions import ConfigurationError, PersistenceError 

# Import specific Reporter classes for type checking (isinstance and issubclass)
from shared_libs.ml_core.monitoring.reporters.alert_reporter import AlertReporter
# Import Contract mới
from shared_libs.infra.monitoring.base_event_emitter import BaseEventEmitter 

logger = logging.getLogger(__name__)

class MonitoringOrchestrator:
    """
    The master orchestrator for the MLOps monitoring pipeline.
    """
    
    # Thêm Dependency Injection cho EventEmitter
    def __init__(self, config: Dict[str, Any], event_emitter: BaseEventEmitter):
        
        # 1. Validate and parse configuration
        try:
            self.validated_config = MonitoringConfig(**config)
        except Exception as e:
            raise ConfigurationError(f"Monitoring configuration validation failed: {e}") from e
            
        self.emitter = event_emitter # NEW: Store injected emitter
        self.monitors: List[BaseMonitor] = self._instantiate_monitors()
        self.reporters: List[BaseReporter] = self._instantiate_reporters()
        
        logger.info(f"Monitoring Orchestrator initialized. Active Monitors: {len(self.monitors)}, Active Reporters: {len(self.reporters)}")

    @staticmethod
    def _get_class_by_convention(type_name: str, component_type: str) -> Type[Any]:
        """
        Dynamically loads a class object based on its standardized name and type, 
        using a strict convention: shared_libs.ml_core.monitoring.<type>s.<type>_<component_type>.py
        """
        # Ex: 'feature_drift' -> 'FeatureDriftMonitor'
        class_name = "".join(word.capitalize() for word in type_name.split("_")) + component_type.capitalize()
        # Ex: 'monitor' -> 'monitors', 'reporter' -> 'reporters'
        module_folder = component_type + "s" 
        module_path = f"shared_libs.ml_core.monitoring.{module_folder}.{type_name}_{component_type}"
        
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            raise ImportError(f"Cannot load {component_type} class '{class_name}' from {module_path}. Check naming convention. Error: {e}")

    def _instantiate_monitors(self) -> List[BaseMonitor]:
        """Instantiates all monitors from the configuration dynamically."""
        monitors = []
        
        for monitor_config in self.validated_config.monitors:
            try:
                MonitorCls = self._get_class_by_convention(monitor_config.type, 'monitor')
                monitors.append(MonitorCls(monitor_config.params))
            except Exception as e:
                logger.error(f"Failed to instantiate monitor '{monitor_config.type}': {e}")
                continue
        return monitors

    def _instantiate_reporters(self) -> List[BaseReporter]:
        """Instantiates all reporters from the configuration dynamically and injects the Emitter."""
        reporters = []
        
        for reporter_config in self.validated_config.reporters:
            try:
                # 1. Dynamically get the Class object
                ReporterCls = self._get_class_by_convention(reporter_config.type, 'reporter')
                
                # 2. Handle special configuration injection for AlertReporter
                params = reporter_config.params
                if issubclass(ReporterCls, AlertReporter):
                    # AlertReporter needs the global alert configuration
                    params = self.validated_config.alerts.dict()
                
                # NEW: Thêm Emitter vào kwargs nếu Reporter chấp nhận nó
                kwargs = {}
                # Dùng introspect để kiểm tra nếu init method có parameter 'emitter'
                if 'emitter' in ReporterCls.__init__.__code__.co_varnames:
                     kwargs['emitter'] = self.emitter
                
                reporters.append(ReporterCls(params, **kwargs)) # CẬP NHẬT THÊM KWARGS
            except Exception as e:
                logger.error(f"Failed to instantiate reporter '{reporter_config.type}': {e}")
                continue
        return reporters

    def run_check(self, reference_data: Any, current_data: Any, **kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes all active monitors and dispatches reports/alerts based on outcomes.
        """
        full_report: List[Dict[str, Any]] = []
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        for monitor in self.monitors:
            monitor_name = monitor.__class__.__name__
            logger.info(f"Executing check: {monitor_name}")
            
            # 1. Execute Check
            report_data = monitor.check(reference_data, current_data, **kwargs)
            alert_status = monitor.get_alert_status(report_data)
            report_message = monitor.get_report_message(report_data)
            
            # Add metadata to report
            report_data['message'] = report_message
            report_data['alert'] = alert_status
            report_data['monitor'] = monitor_name
            report_data['timestamp'] = timestamp
            full_report.append(report_data)

            # 2. Dispatch Reports/Alerts
            for reporter in self.reporters:
                # Create a metric-friendly data payload (only numbers)
                report_metrics = {k: v for k, v in report_data.items() if isinstance(v, (int, float))}

                if alert_status and isinstance(reporter, AlertReporter):
                    # AlertReporter (sends external notification)
                    reporter.report(f"ALERT_{monitor_name.upper()}", report_data)
                else:
                    # Log/Prometheus/Grafana Reporters (reports metrics/logs)
                    reporter.report(f"monitor_{monitor_name.lower()}", report_metrics)
        
        logger.info(f"Monitoring run completed. Total checks: {len(self.monitors)}")
        return full_report