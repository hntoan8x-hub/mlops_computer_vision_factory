# shared_libs/data_labeling/base_labeler.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional
from pydantic import ValidationError
from torch import Tensor

# Import config schema để validate đầu vào
from ..data_labeling.configs.labeler_config_schema import LabelerConfig 

# Giả định chúng ta sẽ cần Connector Factory để đọc file nhãn từ S3/GCS
# Dù Labeler không trực tiếp đọc ảnh, nó cần đọc file nhãn (csv, json)
from ..data_ingestion.factories.connector_factory import ConnectorFactory 

class BaseLabeler(ABC):
    """
    Abstract Base Class (ABC) cho tất cả các Labeler trong CV_Factory.
    Định nghĩa hợp đồng cho việc tải, kiểm tra, và chuẩn hóa nhãn.
    """
    
    def __init__(self, connector_id: str, config: Dict[str, Any]):
        """
        Khởi tạo BaseLabeler.

        Args:
            connector_id (str): ID duy nhất cho Labeler (thường lấy từ tên task).
            config (Dict[str, Any]): Cấu hình thô, sẽ được validate bằng Pydantic.
        """
        self.labeler_id = connector_id
        self.raw_config = config
        self.validated_config: Optional[LabelerConfig] = None
        
        # Kết nối tới Connector Factory để chuẩn bị tải file nhãn
        self.connector_factory = ConnectorFactory
        self._validate_and_parse_config()
        
        # Danh sách nhãn thô đã tải (List[Dict])
        self.raw_labels: List[Dict[str, Any]] = []

    def _validate_and_parse_config(self) -> None:
        """
        Sử dụng Pydantic Schema để kiểm tra tính hợp lệ của cấu hình.
        """
        try:
            # Dùng LabelerConfig để validate cấu trúc chung
            self.validated_config = LabelerConfig(**self.raw_config)
            print(f"[{self.labeler_id}] Config validated successfully.")
        except ValidationError as e:
            raise ValueError(f"Labeler configuration failed Pydantic validation for {self.labeler_id}:\n{e}")

    # --- Hợp đồng Bắt buộc (Interface Methods) ---

    @abstractmethod
    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Tải dữ liệu nhãn thô từ nguồn (CSV, JSON, XML,...) và trả về List[Dict].
        Phương thức này nên sử dụng ConnectorFactory để thực hiện việc đọc file.
        
        Returns:
            List[Dict[str, Any]]: Danh sách các mẫu nhãn đã tải.
        """
        # Lưu ý: Các lớp con sẽ triển khai logic gọi Connector.read() ở đây
        raise NotImplementedError

    @abstractmethod
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Kiểm tra tính hợp lệ của một mẫu nhãn đã tải (ví dụ: kiểm tra đường dẫn file tồn tại, 
        giá trị BBox hợp lệ).
        
        Args:
            sample (Dict[str, Any]): Một mẫu nhãn.
            
        Returns:
            bool: True nếu mẫu hợp lệ.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_tensor(self, label_data: Any) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Chuẩn hóa dữ liệu nhãn đã qua xử lý thành PyTorch Tensor(s) sẵn sàng cho DataLoader.
        
        Args:
            label_data (Any): Dữ liệu nhãn đã được chuẩn hóa.
            
        Returns:
            Union[Tensor, Dict[str, Tensor]]: Tensor hoặc Dictionary Tensor.
        """
        raise NotImplementedError
        
    # --- Phương thức Chung/Helper ---

    def get_source_connector(self):
        """
        Tạo và trả về Data Connector thích hợp để đọc file nhãn.
        """
        if not self.validated_config:
             raise RuntimeError("Config not initialized.")
        
        # Lấy URI từ config cụ thể (giả định mọi config đều có 'label_source_uri')
        source_uri = self.validated_config.params.label_source_uri 
        
        # Logic đơn giản để xác định loại connector cần dùng để đọc file nhãn
        if source_uri.startswith("s3://") or source_uri.startswith("gs://"):
            connector_type = "api" # Giả định API/S3/GCS connector có thể fetch file
        else:
            connector_type = "image" # Giả định ImageConnector có thể đọc file local
            
        # Tên connector/ID có thể được điều chỉnh tùy theo nhu cầu
        return self.connector_factory.get_connector(
            connector_type=connector_type,
            connector_config={"source": source_uri}, 
            connector_id=f"{self.labeler_id}_label_reader"
        )