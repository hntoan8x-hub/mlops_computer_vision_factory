# shared_libs/data_labeling/manual_annotation/detection_parser.py

import logging
import json
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Union, Tuple
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import DetectionLabel, DetectionObject
from ....data_labeling.configs.labeler_config_schema import DetectionLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class DetectionParser(BaseManualAnnotator):
    """
    Specialized Parser for Object Detection: Handles raw label formats like COCO JSON and VOC XML.
    
    This parser is responsible for converting raw, manual annotations into the 
    standardized and validated Pydantic DetectionLabel schema (Trusted Labels).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the parser and validates its configuration against DetectionLabelerConfig.

        Args:
            config: Configuration dictionary matching DetectionLabelerConfig.
        """
        super().__init__(config)
        try:
            # Hardening: Use Pydantic to strictly validate and parse the config parameters
            self.parser_config = DetectionLabelerConfig(**config) 
        except ValidationError as e:
            logger.critical(f"DetectionParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")

        # Placeholder for Category ID map (for COCO). Load this from a file in production.
        self.category_id_to_name: Dict[int, str] = {1: "person", 2: "car", 3: "dog"} 

    def parse(self, raw_input: Union[Dict[str, Any], str, os.PathLike]) -> List[StandardLabel]:
        """
        Parses raw label data (JSON content or XML content/path) and standardizes it.

        Args:
            raw_input: Raw label data (Dict for JSON, str/PathLike for XML file content/path).
        
        Returns:
            List[StandardLabel]: List of validated DetectionLabel objects.

        Raises:
            ValueError: If the input format is unsupported.
            TypeError: If the raw_input type is unexpected for the format.
        """
        input_format = self.parser_config.input_format
        
        if input_format == "coco_json":
            standardized_list = self._parse_coco_format(raw_input)
        elif input_format == "voc_xml":
            standardized_list = self._parse_voc_format(raw_input) 
        else:
            raise ValueError(f"Unsupported detection format: {input_format}")
            
        annotated_labels: List[DetectionLabel] = []
        for item in standardized_list:
             # 1. Apply BBox normalization if required
             final_item = self._apply_normalization(item)
             
             # 2. Validate and create DetectionLabel (Trusted Label check)
             try:
                # We need to validate each object before validating the label list
                validated_objects = []
                for obj_data in final_item.get("objects", []):
                    # Pydantic will check BBox range, x_min < x_max, etc.
                    validated_objects.append(DetectionObject(**obj_data)) 
                
                final_item["objects"] = validated_objects
                
                validated_label = DetectionLabel(**final_item)
                annotated_labels.append(validated_label)
             except ValidationError as e:
                logger.warning(f"Skipping invalid detection entry for {final_item.get('image_path', 'N/A')}: {e}")
                
        return annotated_labels

    def _apply_normalization(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies BBox normalization (pixel to [0, 1]) based on config and image size.
        """
        if not self.parser_config.normalize_bbox:
            return item # Skip normalization
            
        width = item.pop("image_width", None)
        height = item.pop("image_height", None)
        
        if not width or not height or width <= 0 or height <= 0:
            logger.warning(f"Normalization skipped for {item.get('image_path')}: Missing or zero image dimensions.")
            return item
            
        normalized_objects = []
        for obj in item.get("objects", []):
            x_min, y_min, x_max, y_max = obj['bbox']
            
            # Hardening: Apply normalization (pixel / dimension)
            obj['bbox'] = (
                x_min / width,
                y_min / height,
                x_max / width,
                y_max / height
            )
            normalized_objects.append(obj)
            
        item['objects'] = normalized_objects
        return item
        
    def _parse_coco_format(self, raw_coco_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Converts COCO JSON format to an intermediate, raw structured dictionary 
        that includes image dimensions for potential normalization.
        """
        if not isinstance(raw_coco_data, dict):
            raise TypeError("COCO Parser expects a dictionary (JSON content).")
            
        # Map image ID to metadata (file_name, width, height)
        images_map = {
            img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']} 
            for img in raw_coco_data.get('images', [])
        }
        annotations = raw_coco_data.get('annotations', [])
        output_data: Dict[str, Dict[str, Any]] = {}

        for ann in annotations:
            img_id = ann.get('image_id')
            category_id = ann.get('category_id')
            bbox_coco = ann.get('bbox') # [x, y, w, h]
            
            image_info = images_map.get(img_id)
            if not image_info or not bbox_coco or not category_id: continue
            
            image_path = image_info['file_name']
            class_name = self.category_id_to_name.get(category_id, "unknown")
            
            # Convert bbox from [x, y, w, h] (COCO) to [x1, y1, x2, y2]
            x_min, y_min, w, h = bbox_coco
            x_max = x_min + w
            y_max = y_min + h
            
            # Temporary non-normalized object storage (pixel coordinates)
            obj = {"bbox": (x_min, y_min, x_max, y_max), "class_name": class_name}
            
            if image_path not in output_data:
                output_data[image_path] = {
                    "image_path": image_path,
                    "objects": [],
                    # Store dimensions for normalization step
                    "image_width": image_info['width'], 
                    "image_height": image_info['height'],
                }
            output_data[image_path]["objects"].append(obj)
            
        return list(output_data.values())

    def _parse_voc_format(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Converts VOC XML content (one file per image) to the intermediate structured format.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            raise ValueError("Invalid XML content provided for VOC parsing.")
            
        filename = root.find('filename').text
        
        size_node = root.find('size')
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)

        objects = []
        for obj_node in root.findall('object'):
            class_name = obj_node.find('name').text
            bbox_node = obj_node.find('bndbox')
            
            # VOC XML often uses 1-based index; parsing converts to 0-based pixel coords
            x_min = int(bbox_node.find('xmin').text) 
            y_min = int(bbox_node.find('ymin').text) 
            x_max = int(bbox_node.find('xmax').text) 
            y_max = int(bbox_node.find('ymax').text) 
            
            objects.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "class_name": class_name
            })
            
        # Return under the unified structure, including dimensions
        return [{
            "image_path": filename,
            "objects": objects,
            "image_width": width, 
            "image_height": height
        }]