### Dependency Injection (DI) & Inversion of Control (IoC) Lifecycle Diagram

#### 1. **Top-Level Control – Orchestrator / Factory**
```
CVInferenceOrchestrator
   ├── Decides domain context (e.g., 'medical', 'retail')
   ├── Creates correct Postprocessor (MedicalPostprocessor)
   ├── Injects dependency into CVPredictor(postprocessor)
   └── Starts prediction workflow
```
➡ *Responsibility:* Controls creation and wiring of all objects. Nothing inside the predictor is hardcoded.

---

#### 2. **Middle Layer – Core Logic (CVPredictor)**
```
CVPredictor
   ├── Receives postprocessor via constructor (__init__)
   ├── Does not import or create domain-specific logic
   ├── Runs predict_pipeline()
   │     ├── preprocess(image)
   │     ├── model.predict()
   │     └── postprocessor.postprocess(output)
   └── Returns final result
```
➡ *Responsibility:* Execute core logic only. It remains cloud-agnostic and domain-agnostic.

---

#### 3. **Bottom Layer – Domain Logic (Injected Component)**
```
MedicalPostprocessor
   ├── Implements postprocess()
   ├── Applies medical rules, thresholding, or decision logic
   └── Returns domain-specific formatted output
```
➡ *Responsibility:* Encapsulates specialized business rules. It never interacts with ML code directly.

---

#### 4. **Flow Summary (Inversion of Control)
```
Without IoC (Normal Flow):
CVPredictor → Creates MedicalPostprocessor → Tight Coupling

With IoC (Reversed Flow):
Orchestrator → Creates MedicalPostprocessor → Injects into CVPredictor → Loose Coupling
```
➡ *Control is inverted:* CVPredictor no longer decides *what* dependencies it uses — they are provided externally.

---

#### 5. **Strategic Benefits**
| Aspect | Without DI | With DI / IoC |
|---------|-------------|---------------|
| Coupling | Tight (hard-coded imports) | Loose (external injection) |
| Testability | Hard to mock dependencies | Easy to mock/inject test doubles |
| Reusability | Locked to domain | Reusable across domains |
| Maintenance | Code changes ripple everywhere | Code isolated per layer |

---

#### 6. **Analogy: Restaurant Kitchen**
| Role | Non-IoC | IoC |
|------|----------|------|
| Chef (CVPredictor) | Goes shopping for ingredients | Receives ingredients from manager |
| Manager (Orchestrator) | - | Chooses suppliers and provides correct ingredients |
| Ingredients (Postprocessor) | Hardcoded in recipe | Supplied dynamically based on menu |

---

#### 7. **In Practice (Implementation Flow)**
```
# In orchestrator
postprocessor = MedicalPostprocessor()
predictor = CVPredictor(postprocessor=postprocessor)
result = predictor.predict_pipeline(image)
```
✅ *This flow enables modular, scalable, testable design — and forms the foundation of all modern frameworks (FastAPI, Spring, PyTorch Lightning, etc.).*

