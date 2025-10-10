### Common Failure Scenarios & Debug Patterns for Postprocessor State Management

This section lists the most frequent failure modes encountered when managing Postprocessor state across Training and Inference phases, and how to debug them effectively.

---

#### **1. State Not Loaded (Missing Artifact Path)**
**Symptom:**
- Inference output differs significantly from training evaluation.
- Log shows: `FileNotFoundError: postprocessor_state.pkl not found`.

**Root Cause:**
- The postprocessor was initialized but its `load_state()` was never called.
- The MLflow artifact URI or S3 path was not injected into the inference configuration.

**Debug Pattern:**
1. Verify the inference YAML includes the correct artifact path:
   ```yaml
   postprocessor:
     state_uri: s3://mlflow-artifacts/cv/postprocessor_state.pkl
   ```
2. In `CVPredictor.__init__()`, check for `if hasattr(postprocessor, "load_state")`.
3. Add logging before and after `load_state()` to confirm path resolution.

---

#### **2. State Version Mismatch (Train–Serve Drift)**
**Symptom:**
- Model inference results inconsistent with evaluation metrics.
- Warning: `VersionMismatchError: Expected v3.1, got v2.9.`

**Root Cause:**
- The postprocessor state saved under one MLflow run was not updated after retraining.
- Serving still points to an older MLflow version tag.

**Debug Pattern:**
1. Check MLflow UI → Model Registry → Versions → Latest Stage (`Production`).
2. Compare the run_id and commit hash (tagged in training pipeline).
3. Ensure `registry.tag_model_version()` executed successfully after training.
4. Automate version sync via CI/CD by adding:
   ```yaml
   - name: Sync Postprocessor State Version
     run: python scripts/sync_registry_tags.py
   ```

---

#### **3. Corrupted or Incompatible Serialized State**
**Symptom:**
- Error: `pickle.UnpicklingError` or `ValueError: Unknown feature dimensions`.

**Root Cause:**
- Postprocessor updated (e.g., new PCA params) but old serialized file still referenced.
- Schema or feature count changed between versions.

**Debug Pattern:**
1. Always include `schema_version` and `feature_count` metadata inside saved state.
2. Example fix:
   ```python
   def save_state(self, path):
       metadata = {"schema_version": "1.2", "feature_count": len(self.features_)}
       joblib.dump({"metadata": metadata, "state": self.state_}, path)
   ```
3. Validate metadata before applying `load_state()`:
   ```python
   if saved_metadata["schema_version"] != current_schema:
       raise VersionMismatchError("Incompatible schema version.")
   ```

---

#### **4. Postprocessor Not Registered with MLflow**
**Symptom:**
- The model artifact in MLflow does not contain the `postprocessor_state.pkl` file.

**Root Cause:**
- `mlflow.log_artifact()` not called for the postprocessor file.
- The postprocessor was trained but not logged.

**Debug Pattern:**
1. Confirm training orchestrator logs both model and postprocessor artifacts:
   ```python
   mlflow.log_artifact(postprocessor.state_path, artifact_path="postprocessor")
   ```
2. Use MLflow UI → Artifacts → Check presence of postprocessor directory.

---

#### **5. Mixed Environment Serialization (CPU ↔ GPU)**
**Symptom:**
- `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.`

**Root Cause:**
- The postprocessor state saved on GPU context, inference running on CPU environment.

**Debug Pattern:**
1. Save models using CPU context when exporting:
   ```python
   torch.save(model.to("cpu").state_dict(), path)
   ```
2. When loading, always use:
   ```python
   map_location=torch.device("cpu")
   ```
3. Document device context in MLflow tags for auditability:
   ```python
   mlflow.set_tag("device_context", "cuda:0")
   ```

---

#### **6. Missing Dependency or Import Errors in Postprocessor**
**Symptom:**
- Inference fails with `ModuleNotFoundError` or missing package.

**Root Cause:**
- The Docker image used for serving doesn’t include the same environment as training.

**Debug Pattern:**
1. Sync `requirements.txt` between training and inference builds.
2. Use MLflow `conda.yaml` export to enforce environment consistency:
   ```bash
   mlflow run . --env-manager=local
   ```
3. Define `env_hash` tag in MLflow and validate before serving.

---

### ✅ Best Practice Summary
| Area | Problem Prevented | Strategy |
|------|------------------|-----------|
| **State Drift** | Old PCA/Scaler reused | Version tagging + CI sync |
| **Artifact Loss** | Missing state.pkl | mlflow.log_artifact() enforcement |
| **Environment Mismatch** | CPU↔GPU errors | Use `map_location` + tags |
| **Schema Change** | Feature mismatch | Metadata validation |
| **Version Inconsistency** | Wrong state | Registry tag sync |
| **Missing Dependency** | Module import fail | Environment lock via MLflow |

This checklist ensures Postprocessor consistency, enabling reproducible, reliable inference in all environments.

