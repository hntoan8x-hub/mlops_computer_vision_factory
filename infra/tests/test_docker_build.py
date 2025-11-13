import subprocess
import pytest

def test_docker_assistant_build():
    """
    Checks if the Docker image for the assistant service can be built successfully.
    """
    try:
        subprocess.run(
            ["docker", "build", "-f", "../docker/Dockerfile.assistant", "-t", "test-assistant-image", "."],
            check=True,
            capture_output=True,
            text=True
        )
        assert True
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed: {e.stderr}")
        pytest.fail(f"Docker build for assistant failed: {e.stderr}")

def test_docker_trainer_build():
    """
    Checks if the Docker image for the trainer job can be built successfully.
    """
    try:
        subprocess.run(
            ["docker", "build", "-f", "../docker/Dockerfile.trainer", "-t", "test-trainer-image", "."],
            check=True,
            capture_output=True,
            text=True
        )
        assert True
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed: {e.stderr}")
        pytest.fail(f"Docker build for trainer failed: {e.stderr}")
