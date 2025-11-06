# shared_libs/data_labeling/manual_annotation/detection_parser.py

import logging
import json
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Union
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import DetectionLabel, DetectionObject
from ....data_labeling.configs.labeler_config_schema import DetectionLabelerConfig

logger = logging.getLogger(__name__)

class DetectionParser(BaseManualAnnotator):
    """
    Parser chuyên biệt cho Object Detection: Xử lý COCO JSON và VOC XML.
    Chuẩn hóa nhãn thô sang định dạng Pydantic DetectionLabel.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Sử dụng config thô để tạo đối tượng config đã được validate (cho params)
        self.parser_config = DetectionLabelerConfig(**config) 

        # Ánh xạ Category ID (trong COCO) sang Tên lớp (cho ví dụ)
        # Trong thực tế, class_map cần được tải từ file riêng
        self.category_id_to_name: Dict[int, str] = {1: "person", 2: "car", 3: "dog"} 

    def parse(self, raw_input: Union[Dict[str, Any], str]) -> List[StandardLabel]:
        """
        Phân tích dữ liệu thô (nội dung file JSON/XML) và chuẩn hóa.

        Args:
            raw_input (Dict | str): Dữ liệu thô. Nếu là JSON, là Dict. Nếu là XML, là string/file path.
        """
        input_format = self.parser_config.input_format
        
        if input_format == "coco_json":
            standardized_list = self._parse_coco_format(raw_input)
        elif input_format == "voc_xml":
            standardized_list = self._parse_voc_format(raw_input)
        else:
            raise ValueError(f"Unsupported detection format: {input_format}")
            
        # Validate và tạo các đối tượng DetectionLabel
        annotated_labels: List[DetectionLabel] = []
        for item in standardized_list:
             try:
                # Item ở đây là Dict đã được cấu trúc lại theo schema thô
                validated_label = DetectionLabel(**item)
                annotated_labels.append(validated_label)
             except Exception as e:
                logger.warning(f"Skipping invalid detection entry: {e}")
                
        # Trả về List[StandardLabel] (là Union của các Pydantic Models)
        return annotated_labels

    def _parse_coco_format(self, raw_coco_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chuyển đổi từ định dạng COCO JSON sang định dạng DetectionLabel thô.
        """
        if not isinstance(raw_coco_data, dict):
            raise TypeError("COCO Parser expects a dictionary (JSON content).")
            
        images_map = {img['id']: img['file_name'] for img in raw_coco_data.get('images', [])}
        annotations = raw_coco_data.get('annotations', [])
        
        output_data: Dict[str, Dict[str, Any]] = {}

        for ann in annotations:
            img_id = ann.get('image_id')
            category_id = ann.get('category_id')
            bbox_coco = ann.get('bbox') # [x, y, w, h]
            
            if img_id not in images_map: continue
                
            image_path = images_map[img_id]
            class_name = self.category_id_to_name.get(category_id, "unknown")
            
            # Chuyển bbox từ [x, y, w, h] (COCO) sang [x1, y1, x2, y2]
            x_min, y_min, w, h = bbox_coco
            x_max = x_min + w
            y_max = y_min + h
            
            obj = {
                "bbox": (x_min, y_min, x_max, y_max),
                "class_name": class_name
            }
            
            if image_path not in output_data:
                output_data[image_path] = {
                    "image_path": image_path,
                    "objects": []
                }
            output_data[image_path]["objects"].append(obj)
            
        return list(output_data.values())

    def _parse_voc_format(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Chuyển đổi từ nội dung VOC XML (một file cho một ảnh) sang định dạng chuẩn.
        Giả định raw_input là nội dung XML (string).
        """
        root = ET.fromstring(xml_content)
        
        # 1. Lấy đường dẫn ảnh
        filename = root.find('filename').text
        
        # 2. Lấy kích thước ảnh (cần thiết cho một số tác vụ)
        size_node = root.find('size')
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)

        objects = []
        for obj_node in root.findall('object'):
            class_name = obj_node.find('name').text
            bbox_node = obj_node.find('bndbox')
            
            # VOC XML sử dụng tọa độ pixel (thường là 1-based index)
            # Chúng ta chuyển về 0-based index [x1, y1, x2, y2]
            x_min = int(bbox_node.find('xmin').text) 
            y_min = int(bbox_node.find('ymin').text) 
            x_max = int(bbox_node.find('xmax').text) 
            y_max = int(bbox_node.find('ymax').text) 
            
            objects.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "class_name": class_name
            })
            
        # Trả về dưới dạng List chứa một Dict duy nhất (vì 1 XML = 1 ảnh)
        return [{
            "image_path": filename,
            "objects": objects,
            "image_width": width, # Bổ sung metadata
            "image_height": height
        }]