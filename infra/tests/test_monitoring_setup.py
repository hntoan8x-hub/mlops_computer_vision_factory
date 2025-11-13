import subprocess
import pytest
import yaml

def test_prometheus_config():
    """
    Validates the Prometheus configuration file syntax.
    """
    try:
        # Use Prometheus's own tool to check config syntax (requires prometheus binary)
        # command = ["promtool", "check", "config", "../monitoring/prometheus.yaml"]
        # subprocess.run(command, check=True, capture_output=True, text=True)
        # Alternative: simple YAML parse check
        with open("../monitoring/prometheus.yaml", "r") as f:
            yaml.safe_load(f)
        assert True
    except (subprocess.CalledProcessError, FileNotFoundError, yaml.YAMLError) as e:
        pytest.fail(f"Prometheus config is invalid: {e}")

def test_alertmanager_config():
    """
    Validates the Alertmanager configuration file syntax.
    """
    try:
        # Use Alertmanager's own tool (requires alertmanager binary)
        # command = ["amtool", "check-config", "../monitoring/alertmanager.yaml"]
        # subprocess.run(command, check=True, capture_output=True, text=True)
        # Alternative: simple YAML parse check
        with open("../monitoring/alertmanager.yaml", "r") as f:
            yaml.safe_load(f)
        assert True
    except (subprocess.CalledProcessError, FileNotFoundError, yaml.YAMLError) as e:
        pytest.fail(f"Alertmanager config is invalid: {e}")
