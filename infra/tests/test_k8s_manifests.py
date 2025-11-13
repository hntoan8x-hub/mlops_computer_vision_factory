import subprocess
import pytest
import os

def validate_yaml_file(file_path):
    """
    Validates a Kubernetes manifest YAML file using `kubectl dry-run --validate`.
    """
    try:
        command = [
            "kubectl",
            "apply",
            "-f",
            file_path,
            "--server-side",
            "--dry-run=client"
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        return f"Validation failed for {file_path}: {e.stderr}"

def test_k8s_manifests_validation():
    """
    Iterates through all Kubernetes manifest files and validates them.
    """
    k8s_dir = "../k8s/"
    validation_errors = []

    for root, _, files in os.walk(k8s_dir):
        for file_name in files:
            if file_name.endswith(('.yaml', '.yml')):
                file_path = os.path.join(root, file_name)
                result = validate_yaml_file(file_path)
                if result is not True:
                    validation_errors.append(result)

    if validation_errors:
        pytest.fail("The following Kubernetes manifest files have validation errors:\n" + "\n".join(validation_errors))
