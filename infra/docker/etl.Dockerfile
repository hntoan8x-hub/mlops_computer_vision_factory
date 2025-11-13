# ----------------------------
# 🔹 ETL ROLE IMAGE (Cleanup, Governance, Monitoring Batch)
# ----------------------------

# 1. Kế thừa từ Base Image đã Hardened
FROM hardened_base AS etl_stage 

# 2. Copy Python Packages cần thiết (pandas, requests, prometheus_client)
COPY --from=dependency_builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 3. ENTRYPOINT: Chạy job TMR Check (Hardened Glue)
ENTRYPOINT ["python3", "scripts/run_retrain_check.py"]
# NOTE: Lệnh này thường được override bằng K8s CronJob để chạy cleanup_artifacts.py hoặc monitor_service_health.py