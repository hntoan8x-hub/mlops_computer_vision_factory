# scripts/monitor_service_health.py

import argparse
import logging
import time
import requests
from typing import Dict, Any, Tuple

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_HEALTH_MONITOR")

# Giả định: Hàm tiện ích để kiểm tra API (thay thế cho requests.get)
def check_http_status(url: str, timeout: int = 5) -> Tuple[int, float]:
    """Kiểm tra trạng thái HTTP và đo độ trễ."""
    try:
        start_time = time.time()
        # Trong môi trường thực, cần sử dụng requests.get(url, timeout=timeout)
        # response = requests.get(url, timeout=timeout) 
        
        # Mô phỏng phản hồi thành công và độ trễ
        time.sleep(0.1) 
        latency = (time.time() - start_time) * 1000 # milliseconds
        return 200, latency 
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed for {url}: {e}")
        return 500, 0.0

def main():
    """
    Hàm chính thực thi luồng Monitor Service Health.
    """
    parser = argparse.ArgumentParser(description="Kiểm tra Liveness, Readiness, và Latency của Endpoint.")
    parser.add_argument("--endpoint-url", type=str, required=True, 
                        help="URL cơ sở của Endpoint cần kiểm tra (ví dụ: http://api.defect-detector.svc).")
    parser.add_argument("--liveness-path", type=str, default="/health/liveness", 
                        help="Đường dẫn kiểm tra Liveness.")
    parser.add_argument("--readiness-path", type=str, default="/health/readiness", 
                        help="Đường dẫn kiểm tra Readiness.")
    parser.add_argument("--max-latency-ms", type=float, default=500.0, 
                        help="Độ trễ tối đa chấp nhận được (milliseconds).")
    args = parser.parse_args()

    # Xây dựng các URL kiểm tra
    liveness_url = args.endpoint_url.rstrip('/') + args.liveness_path
    readiness_url = args.endpoint_url.rstrip('/') + args.readiness_path
    
    logger.info(f"Starting Health Monitor for Endpoint: {args.endpoint_url}")

    try:
        # 1. Kiểm tra Liveness (Dịch vụ có đang chạy không)
        status_live, latency_live = check_http_status(liveness_url)
        is_live = status_live == 200
        
        # 2. Kiểm tra Readiness (Dịch vụ có sẵn sàng nhận traffic không - mô hình đã load)
        status_ready, latency_ready = check_http_status(readiness_url)
        is_ready = status_ready == 200
        
        # 3. Kiểm tra Latency
        overall_latency = max(latency_live, latency_ready)
        is_fast_enough = overall_latency <= args.max_latency_ms

        health_status = "HEALTHY" if is_live and is_ready and is_fast_enough else "UNHEALTHY"
        
        # 4. Báo cáo
        logger.info("=====================================================")
        logger.info(f"SERVICE HEALTH STATUS: {health_status}")
        logger.info(f"  Liveness Check: {'PASS' if is_live else 'FAIL'} (Status: {status_live})")
        logger.info(f"  Readiness Check: {'PASS' if is_ready else 'FAIL'} (Status: {status_ready})")
        logger.info(f"  Max Latency: {overall_latency:.2f}ms (Threshold: {args.max_latency_ms}ms)")
        logger.info("=====================================================")

        if health_status == "UNHEALTHY":
            # Gửi cảnh báo (sẽ kích hoạt các quy trình tự động nếu cần)
            # emitter.emit_event("service_unhealthy", payload={...})
            exit(1)
        
    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Monitoring script failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()