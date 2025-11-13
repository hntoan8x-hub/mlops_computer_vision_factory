# ----------------------------
# 🔹 INFERENCE ROLE IMAGE
# ----------------------------

# 1. Kế thừa từ Base Image đã Hardened
FROM hardened_base AS inference_stage

# 2. Copy Python Packages từ Stage dependency_builder
# Packages: fastapi, uvicorn, gunicorn, numpy (cần cho API)
COPY --from=dependency_builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 3. EXPOSE PORT và Cấu hình Service
EXPOSE 8000

# 4. KHỞI ĐỘNG (CMD Production Performance) - Sửa tên file Python
CMD ["gunicorn", "api_service.main:app", "--workers", "4", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker"]