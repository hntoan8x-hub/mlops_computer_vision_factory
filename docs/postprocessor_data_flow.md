### Data Flow: From Model Output → Postprocessor → Domain Response (Train ↔ Serve Consistency Included)

#### 1. **High-Level Overview**
```
Input Image
   │
   ▼
CVPredictor.predict_pipeline()
   │
   ├── Preprocess: Normalize, Resize (Adapter Layer)
   │
   ├── Model Inference: CNN or Vision Transformer
   │       └── Output: Raw tensor or probability scores
   │
   └── Postprocessor.postprocess(output)
            │
            ├── Domain-specific transformation
            │   (Convert logits → label, thresholding, add metadata)
            │
            └── Return structured business result

Final Response → Delivered to API / Dashboard / Decision Engine
```

---

#### 2. **Concrete Example – Medical Imaging**
```
Step 1: Raw Model Output
   {"classes": ["normal", "pneumonia", "covid"],
    "probabilities": [0.15, 0.80, 0.05]}

Step 2: Postprocessor (MedicalPostprocessor)
   postprocess():
       - Select label = max(probabilities) → "pneumonia"
       - Compute confidence = 0.80
       - Map to severity tier = "high"
       - Format result for reporting

Step 3: Domain Response
   {"diagnosis": "pneumonia",
    "confidence": "80%",
    "severity": "high"}
```

---

#### 3. **Train ↔ Serve Consistency (Critical Role of Postprocessor)**
```
During Training:
   ├── Postprocessor defines how labels and thresholds are computed.
   ├── These thresholds/parameters are serialized (save_state). 
   └── Ensures reproducibility during validation and serving.

During Inference:
   ├── Postprocessor.load_state() restores saved thresholds.
   ├── Guarantees identical mapping logic (e.g., same cutoffs, label set).
   └── Eliminates Train–Serve Skew (identical results given same data).
```
➡ *Without Postprocessor*: Each deployment environment may interpret model logits differently → inconsistent KPIs and regulatory risk in banking/medical contexts.

---

#### 4. **Why CVPredictor Needs the Postprocessor Parameter**
```
CVPredictor
   ├── Is domain-agnostic → cannot hardcode medical/retail logic.
   ├── Only guarantees it will call .postprocess(output).
   └── Requires the dependency to be injected externally.

Orchestrator
   ├── Decides which domain logic applies.
   ├── Creates postprocessor = MedicalPostprocessor().
   └── Injects it into CVPredictor(postprocessor).

DomainModel
   └── Defines its own postprocessor with domain-specific meaning.
```

**Dependency Direction:**
```
[Domain Postprocessor] ← (injected by) ← [Orchestrator] → (used by) → [CVPredictor]
```

---

#### 5. **Cross-Domain Scalability**
| Domain | Postprocessor | Example Output |
|---------|----------------|----------------|
| Medical | MedicalPostprocessor | {diagnosis: "pneumonia", severity: "high"} |
| Retail  | RetailPostprocessor  | {object_count: 12, categories: ["shoe", "bag"]} |
| OCR/Text | OCRPostprocessor | {invoice_id: "#1234", amount: 12.50} |

✅ **Result:** CVPredictor never changes — only the injected Postprocessor changes.

---

#### 6. **Key Takeaways**
| Concept | Meaning |
|----------|----------|
| `postprocessor` | A domain adapter that interprets model output. |
| Injection | Allows domain-specific logic without modifying shared_libs. |
| Train ↔ Serve Consistency | Postprocessor preserves identical interpretation of model outputs across training and inference. |
| Benefit | Decoupling, reusability, auditability, and business consistency. |

**In one line:**  
> CVPredictor doesn’t know *what* the prediction means — Postprocessor tells it *how to make it meaningful*, *and ensures it always means the same thing everywhere.*

