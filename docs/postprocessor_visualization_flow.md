**Postprocessor Visualization Layer: From Training to Inference**

---

### 1. Training Phase: Learn and Persist State
```
CVTrainingOrchestrator.run()
        |
        v
   [Feature Engineering]
        |
        +--> CVPostprocessor.fit(train_predictions, train_labels)
        |         |
        |         --> Learns thresholds, calibration curves, or mappings
        |
        +--> CVPostprocessor.save_state()
                  |
                  --> Saves JSON artifact (thresholds, mapping tables)
                      to MLflow / S3 / MinIO under run_artifacts/postprocessor/
```

**Key Outputs:**
- `postprocessor_state.json`
- MLflow logged URI (used in inference)

---

### 2. Model Registration
```
CVTrainingOrchestrator.finalize()
        |
        +--> registry.register_model(run_id, model_uri, postprocessor_uri)
                  |
                  --> Creates model version (v1, v2...) in MLflow Registry
                  --> Tags model with matching postprocessor artifact
```

**Purpose:** Ensure train–serve consistency by pairing the model with its exact postprocessor state.

---

### 3. Inference Initialization
```
CVInferenceOrchestrator.run()
        |
        v
   CVPredictor.__init__(model_uri, postprocessor=MedicalPostprocessor())
        |
        +--> postprocessor.load_state(postprocessor_uri)
                  |
                  --> Restores learned parameters (cutoffs, mappings)
```

**Result:** The predictor is now initialized with *identical* logic and parameters as the model trained version.

---

### 4. Inference Execution
```
CVPredictor.predict_pipeline(new_input)
        |
        +--> model_output = model.predict(new_input)
        |
        +--> postprocessed_output = postprocessor.transform(model_output)
        |
        +--> return postprocessed_output
```

**Example:**
If threshold=0.7 (learned during training):
- model_output = 0.72 → returns "Positive"
- model_output = 0.65 → returns "Negative"

---

### 5. Logging and Monitoring
```
Monitoring Layer (Prometheus + Loggers)
        |
        +--> Collect latency of postprocessor.transform()
        +--> Compare output distribution vs. training baseline
        +--> Detect drift in prediction calibration or class balance
```

---

### ✅ Benefits Recap
| Layer | Purpose | Mechanism | Outcome |
|--------|----------|------------|----------|
| Training | Learn thresholds/mappings | fit() + save_state() | Stable decision rules |
| Registry | Pair model with state | register_model() | Auditability |
| Inference | Reload state | load_state() | Consistency |
| Monitoring | Validate performance | Prometheus, MLflow metrics | Early drift detection |

---

**Summary Insight:**
The Postprocessor acts as the final guardian of model correctness. It encapsulates domain-specific logic and guarantees that decisions made in production are traceable, consistent, and explainable relative to training-time behavior.

