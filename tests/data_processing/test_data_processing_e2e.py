import pytest
import numpy as np
from unittest.mock import Mock, patch
from shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator import CVPreprocessingOrchestrator
from shared_libs.data_processing.configs.preprocessing_config_schema import ProcessingConfig
from shared_libs.data_processing._utils.data_type_utils import is_video_data # Utility đã tách ra

# --- DỮ LIỆU GIẢ (STUBS) ---
IMAGE_RGB_3D = np.zeros((100, 100, 3), dtype=np.uint8)
VIDEO_4D = np.zeros((10, 100, 100, 3), dtype=np.uint8) # 10 frames
METADATA_RGB = {"color_format": "RGB", "resolution": 512, "source_quality": "high"}

# --- CONFIG SSOT (Được tối giản cho test) ---
# NOTE: Trong thực tế, cần tải config YAML đầy đủ
VALID_CONFIG_DICT = {
    "enabled": True,
    "cleaning": {
        "enabled": True,
        "policy_mode": "conditional_metadata",
        "steps": [
            {"type": "resizer", "enabled": True, "params": {"width": 256, "height": 256}},
            {"type": "color_space", "enabled": True, "params": {"conversion_code": "BGR2RGB"}},
            {"type": "normalizer", "enabled": True, "params": {"mean": [0.5], "std": [0.5]}}
        ]
    },
    "augmentation": {
        "enabled": True,
        "policy_mode": "randaugment",
        "n_select": 2,
        "magnitude": 0.8,
        "steps": [
            {"type": "flip_rotate", "enabled": True},
            {"type": "noise_injection", "enabled": True},
        ]
    },
    "feature_engineering": {
        "enabled": True,
        "components": [
            {"type": "cnn_embedder", "enabled": True, "params": {"model_name": "resnet18"}},
            {"type": "dim_reducer", "enabled": True, "params": {"method": "pca", "n_components": 128}}
        ]
    },
    "video_processing": {"enabled": False} # Mặc định tắt video flow
}

# --- Fixture for Orchestrator Instance (Mocking Sub-Orchestrators) ---
@pytest.fixture
def mock_sub_orchestrators(mocker):
    # Mocking các Orchestrator cấp dưới để cô lập test
    mocker.patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.ImageCleanerOrchestrator', autospec=True)
    mocker.patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.ImageAugmenterOrchestrator', autospec=True)
    mocker.patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.ImageFeatureExtractorOrchestrator', autospec=True)
    mocker.patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.VideoProcessingOrchestrator', autospec=True)
    
    # Giả lập output của feature processor là một vector 128D
    mock_feature_instance = mocker.MagicMock()
    mock_feature_instance.transform.return_value = np.zeros(128)
    
    mocker.patch.object(CVPreprocessingOrchestrator, '_determine_feature_processor', return_value=mocker.MagicMock(return_value=mock_feature_instance))
    
    return CVPreprocessingOrchestrator(config=VALID_CONFIG_DICT, context='training')
  
# Test_1.1 - 1.4: Architectural & Lifecycle Tests
def test_architecture_initialization(mock_sub_orchestrators):
    """Test_1.1 & Test_1.2: Verifies Pydantic validation and sub-orchestrator initialization (Decoupling)."""
    orchestrator = mock_sub_orchestrators
    
    # 1. Verification of Pydantic Validation (Success is no exception raised)
    assert isinstance(orchestrator.validated_config, ProcessingConfig)
    
    # 2. Verification of Composition (Sub-Orchestrators are created)
    assert orchestrator.image_cleaner is not None
    assert orchestrator.image_augmenter is not None
    assert orchestrator.feature_processor is not None
    
    # 3. Verification of Context
    assert orchestrator.context == 'training'
    assert orchestrator.processor_type == 'embedding' # Based on config dict stub
    
def test_governance_multiple_embedders_fails():
    """Test_1.4: Verifies the Pydantic governance rule (only one embedder) is enforced."""
    bad_config = VALID_CONFIG_DICT.copy()
    bad_config['feature_engineering'] = {
        "enabled": True,
        "components": [
            {"type": "cnn_embedder", "enabled": True},
            {"type": "vit_embedder", "enabled": True} # Two active embedders
        ]
    }
    
    with pytest.raises(ValueError, match="Only one deep learning embedder"):
        CVPreprocessingOrchestrator(config=bad_config, context='inference')

def test_state_management_save_load(tmpdir, mock_sub_orchestrators):
    """Test_1.3: Verifies that save/load delegates correctly to all sub-orchestrators."""
    save_path = str(tmpdir.mkdir("state"))
    orchestrator = mock_sub_orchestrators
    
    # 1. Run Save and verify delegation paths
    orchestrator.save(save_path)
    
    orchestrator.image_cleaner.save.assert_called_once_with(os.path.join(save_path, "cleaner"))
    orchestrator.image_augmenter.save.assert_called_once_with(os.path.join(save_path, "augmenter"))
    orchestrator.feature_processor.save.assert_called_once_with(os.path.join(save_path, "processor"))

    # 2. Run Load (on a new instance, assuming factories work)
    orchestrator.image_cleaner.load.assert_called_once_with(os.path.join(save_path, "cleaner"))
    orchestrator.image_augmenter.load.assert_called_once_with(os.path.join(save_path, "augmenter"))
    orchestrator.feature_processor.load.assert_called_once_with(os.path.join(save_path, "processor"))
    
# Test_3.1, 3.2, 2.1: Policy and Sequencing Tests
@patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.ImageAugmenterOrchestrator', autospec=True)
@patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.ImageCleanerOrchestrator', autospec=True)
def test_sequencing_and_policy_flow(MockCleanerOrchestrator, MockAugmenterOrchestrator):
    """Test_2.1, 3.2: Verifies the main flow sequencing (Clean -> Augment -> Feature)."""
    
    # Stub the feature processor output
    mock_feature_processor = Mock()
    mock_feature_processor.transform.return_value = np.array([0.1]*128)
    
    # Create orchestrator with a patched feature processor instance
    orchestrator = CVPreprocessingOrchestrator(config=VALID_CONFIG_DICT, context='training')
    orchestrator.feature_processor = mock_feature_processor
    
    # Define intermediate data flows
    MOCK_CLEANED_DATA = np.zeros((256, 256, 3))
    MOCK_AUGMENTED_DATA = np.ones((256, 256, 3))
    
    # Configure Mocks' side effects to simulate data transformation
    MockCleanerOrchestrator.return_value.transform.return_value = MOCK_CLEANED_DATA
    MockAugmenterOrchestrator.return_value.transform.return_value = MOCK_AUGMENTED_DATA
    
    # Run the orchestrator with mock labels for Augmentation
    result = orchestrator.run(IMAGE_RGB_3D, labels=[1])

    # 1. Verification of Sequencing (CRITICAL)
    # Check if Cleaner transform was called first
    MockCleanerOrchestrator.return_value.transform.assert_called_once()
    
    # Check if Augmenter transform was called second (takes output of Cleaner)
    MockAugmenterOrchestrator.return_value.transform.assert_called_once()
    assert np.array_equal(MockAugmenterOrchestrator.return_value.transform.call_args[0][0], MOCK_CLEANED_DATA)
    
    # Check if Feature Processor was called last (takes output of Augmenter)
    mock_feature_processor.transform.assert_called_once()
    assert np.array_equal(mock_feature_processor.transform.call_args[0][0], MOCK_AUGMENTED_DATA)
    
    # 2. Verification of Augmentation Policy (Test_3.2 - RandAugment)
    # Augmenter must receive labels/kwargs for Mixup/CutMix logic
    MockAugmenterOrchestrator.return_value.transform.assert_called_with(MOCK_CLEANED_DATA, labels=[1])
    
    # 3. Verification of Final Output
    assert np.array_equal(result, np.array([0.1]*128))
    
def test_augmentation_toggle_off():
    """Test_2.3: Verifies Augmentation is skipped in 'inference' context."""
    orchestrator = CVPreprocessingOrchestrator(config=VALID_CONFIG_DICT, context='inference')
    
    # Run the orchestrator
    orchestrator.run(IMAGE_RGB_3D)

    # Augmenter should NOT be called in inference context
    orchestrator.image_augmenter.transform.assert_not_called()

def test_adaptive_cleaning_metadata(mocker):
    """Test_3.1: Verifies metadata is passed and used for conditional cleaning."""
    
    # Mock the policy controller's decision to verify the flow
    mock_cleaner_instance = mocker.MagicMock()
    
    # Create the orchestrator instance
    orchestrator = CVPreprocessingOrchestrator(config=VALID_CONFIG_DICT, context='inference')
    orchestrator.image_cleaner = mock_cleaner_instance
    
    # Run transform, passing METADATA_RGB
    orchestrator.run(IMAGE_RGB_3D, metadata=METADATA_RGB)
    
    # Cleaner Orchestrator must receive metadata for its internal Policy Controller to act
    mock_cleaner_instance.transform.assert_called_once()
    
    # Check that metadata was passed correctly
    call_args, call_kwargs = mock_cleaner_instance.transform.call_args
    assert call_kwargs['metadata'] == METADATA_RGB
    
# Test_4.1 - 4.3: Video Bridge Tests
@pytest.fixture
def setup_video_orchestrator(mocker):
    # 1. Setup Video Config
    video_config = VALID_CONFIG_DICT.copy()
    video_config['video_processing'] = {
        "enabled": True,
        "cleaners": [], # Minimal setup
        "samplers": [{"type": "uniform_sampler", "params": {"target_frames": 10}}]
    }
    
    # 2. Mock Video Orchestrator
    mocker.patch('shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator.VideoProcessingOrchestrator', autospec=True)
    
    # 3. Mock Video Orchestrator's transform output: 10 frames (3D array)
    MOCK_FRAME_LIST = [np.zeros((100, 100, 3)) + i for i in range(10)]
    mocker.patch.object(VideoProcessingOrchestrator, 'transform', return_value=MOCK_FRAME_LIST)

    # 4. Mock the internal _run_image_pipeline helper method
    mock_run_image_pipeline = mocker.patch.object(CVPreprocessingOrchestrator, '_run_image_pipeline', return_value=np.zeros(128))
    
    orchestrator = CVPreprocessingOrchestrator(config=video_config, context='inference')
    
    return orchestrator, MOCK_FRAME_LIST, mock_run_image_pipeline

def test_video_flow_and_bridge_execution(setup_video_orchestrator):
    """Test_4.1 & 4.2: Verifies video detection and frame processing loop (The Bridge)."""
    orchestrator, MOCK_FRAME_LIST, mock_run_image_pipeline = setup_video_orchestrator
    
    # 1. Verification of Video Detection and Orchestrator Call (Test_4.1)
    result = orchestrator.run(VIDEO_4D)
    
    # Video Orchestrator transform must be called once
    orchestrator.video_processor.transform.assert_called_once() 
    
    # 2. Verification of Image Pipeline Loop (Test_4.2)
    # The internal image pipeline helper must be called once for EACH frame (10 times)
    assert mock_run_image_pipeline.call_count == 10
    
    # Check the input to the image pipeline (should be the first frame)
    assert np.array_equal(mock_run_image_pipeline.call_args_list[0][0][0], MOCK_FRAME_LIST[0])

    # 3. Verification of Final Output (Test_4.3)
    # Final result must be a list containing 10 feature vectors
    assert isinstance(result, list)
    assert len(result) == 10
    assert result[0].shape == (128,)
    
