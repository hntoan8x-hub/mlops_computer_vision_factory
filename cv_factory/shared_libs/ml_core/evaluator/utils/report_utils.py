import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_json_report(metrics: Dict[str, Any], path: str) -> None:
    """
    Generates a JSON report from a dictionary of metrics.

    Args:
        metrics (Dict[str, Any]): A dictionary containing evaluation metrics.
        path (str): The file path to save the JSON report.
    """
    try:
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"JSON report generated successfully at {path}")
    except Exception as e:
        logger.error(f"Failed to generate JSON report: {e}")
        
def generate_html_report(metrics: Dict[str, Any], path: str) -> None:
    """
    Generates a simple HTML report from a dictionary of metrics.

    Args:
        metrics (Dict[str, Any]): A dictionary containing evaluation metrics.
        path (str): The file path to save the HTML report.
    """
    try:
        html_content = "<h1>Evaluation Report</h1>"
        html_content += "<table border='1' style='width:100%'><tr><th>Metric</th><th>Value</th></tr>"
        for key, value in metrics.items():
            html_content += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
        html_content += "</table>"
        
        with open(path, 'w') as f:
            f.write(html_content)
        logger.info(f"HTML report generated successfully at {path}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")