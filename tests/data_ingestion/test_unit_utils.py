# tests/data_ingestion/test_unit_utils.py (Excerpt for file_utils)

import pytest
import os
from cv_factory.shared_libs.data_ingestion.utils.file_utils import get_file_paths, is_valid_file

@pytest.fixture
def mock_directory_structure(tmp_path):
    """
    Creates a temporary directory structure for recursive file testing.
    - tmp_path/img/1.jpg
    - tmp_path/img/sub/2.png
    - tmp_path/doc/3.txt (unsupported)
    """
    img_dir = tmp_path / "img"
    sub_dir = img_dir / "sub"
    doc_dir = tmp_path / "doc"
    
    os.makedirs(sub_dir)
    os.makedirs(doc_dir)

    (img_dir / "1.jpg").write_text("image content")
    (sub_dir / "2.png").write_text("image content")
    (doc_dir / "3.txt").write_text("text content")
    
    return str(tmp_path)

def test_get_file_paths_recursive(mock_directory_structure):
    """
    Tests if get_file_paths correctly finds files recursively based on extensions.
    """
    supported = [".jpg", ".png"]
    found_paths = get_file_paths(os.path.join(mock_directory_structure, "img"), supported)
    
    assert len(found_paths) == 2
    assert any(p.endswith("1.jpg") for p in found_paths)
    assert any(p.endswith("2.png") for p in found_paths)
    assert not any(p.endswith("3.txt") for p in found_paths)

def test_is_valid_file_checks_permissions(tmp_path, monkeypatch):
    """
    Tests if is_valid_file correctly checks for file existence and read permissions.
    """
    test_file = tmp_path / "test.dat"
    test_file.write_text("data")
    
    # 1. Test valid file
    assert is_valid_file(str(test_file)) is True
    
    # 2. Test non-existent file
    assert is_valid_file("/non/existent/path") is False
    
    # 3. Test lack of read permission (Mocking os.access)
    def mock_access(path, mode):
        if mode == os.R_OK:
            return False
        return True
        
    monkeypatch.setattr(os, 'access', mock_access)
    assert is_valid_file(str(test_file)) is False