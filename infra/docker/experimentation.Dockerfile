# experimentation.Dockerfile (UPDATED FOR FLEXIBILITY)
# ----------------------------
# 🔹 EXPERIMENTATION ROLE IMAGE (Testing, Validation, Deployment)
# ----------------------------

# 1. Kế thừa từ Base Image đã Hardened
FROM hardened_base AS experimentation_stage 

# 2. Copy Python Packages cần thiết
# Packages: Thư viện kiểm thử (pytest), client deployment, mlops client
COPY --from=dependency_builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 3. Chuyên biệt hóa (Nếu cần thêm data/tool đặc biệt cho testing)
# Không cần NLTK, giữ nguyên user non-root

# 4. ENTRYPOINT: Đặt thành /bin/bash hoặc sh để dễ dàng override
# Đây là cách tiếp cận Hardened cho các job không xác định trước
ENTRYPOINT ["/bin/bash", "-c"] 
# CMD mặc định sẽ là lệnh bạn muốn chạy
CMD ["python3", "scripts/deploy_service.py"] 
# NOTE: Để chạy find_best, bạn chỉ cần override CMD: 
# docker run --rm your_image python3 scripts/find_best_experiment.py ...