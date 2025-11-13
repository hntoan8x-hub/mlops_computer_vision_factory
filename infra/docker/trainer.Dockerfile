# ----------------------------
# 🔹 TRAINER ROLE IMAGE (Indexing & Fine-Tuning)
# ----------------------------

# 1. Kế thừa từ Base Image đã Hardened
FROM hardened_base AS trainer_stage 

# 2. Copy Python Packages cần thiết
# Packages: PyTorch/TensorFlow, scikit-learn, faiss (cần cho Reranker/Indexing)
COPY --from=dependency_builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 3. Chuyên biệt hóa: Tải NLTK data (nếu cần cho các tác vụ NLP/Training)
USER root
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
USER appuser # Quay lại non-root user

# 4. ENTRYPOINT: Chạy Training Job chính (Bao gồm Feature Store Indexing)
ENTRYPOINT ["python3", "scripts/run_training_job.py"]
# NOTE: Lệnh này có thể được override tại K8s để chạy các job đặc biệt.