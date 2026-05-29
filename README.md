# Tarea 2 — BME513: Inteligencia Artificial en Salud

**Predicción de concentración plasmática de biomarcador mediante redes neuronales feedforward**

---

## Resultado principal

| Modelo | MAE | Error Relativo | Predicciones negativas |
|---|---|---|---|
| MLP + MSE (base) | 1.533 | 33.3% | 3 |
| MLP + NLL log-normal | 1.370 | 24.7% | 0 |
| **MLP + NLL log-normal + L2 λ=5e⁻³** | **1.173** | **20.5%** | **0** |

---

## Descripción del problema

Se predice la **concentración plasmática de un biomarcador** (variable estrictamente positiva, distribución log-normal) a partir de 5 variables clínicas:

| Variable | Descripción |
|---|---|
| edad | Edad del paciente [18–90] años |
| dosis | Dosis del medicamento [0–1] |
| horas | Horas post-administración [0–24] |
| ir | Índice de riesgo ~ N(0,1) |
| ib | Inflamación base ~ Gamma(2,1) |

**Proceso generador:**

´´´
# Tarea 2 — BME513: Inteligencia Artificial en Salud

**Predicción de concentración plasmática de biomarcador mediante redes neuronales feedforward**

---

## Resultado principal

| Modelo | MAE | Error Relativo | Predicciones negativas |
|---|---|---|---|
| MLP + MSE (base) | 1.533 | 33.3% | 3 |
| MLP + NLL log-normal | 1.370 | 24.7% | 0 |
| **MLP + NLL log-normal + L2 λ=5e⁻³** | **1.173** | **20.5%** | **0** |

---

## Descripción del problema

Se predice la **concentración plasmática de un biomarcador** (variable estrictamente positiva, distribución log-normal) a partir de 5 variables clínicas:

| Variable | Descripción |
|---|---|
| edad | Edad del paciente [18–90] años |
| dosis | Dosis del medicamento [0–1] |
| horas | Horas post-administración [0–24] |
| ir | Índice de riesgo ~ N(0,1) |
| ib | Inflamación base ~ Gamma(2,1) |

**Proceso generador:**

´´´
y = exp( g(x) + ε ),   ε ~ N(0, 0.25²)
g(x) = -1.2 + 0.018·edad + 1.4·sin(π·dosis) + 0.03·horas
+ 0.55·ir + 0.35·√ib - 0.015·dosis·horas

´´´

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `tarea_2.ipynb` | Notebook completo con las 7 partes implementadas |
| `informe_tarea2.docx` | Informe Word con derivaciones matemáticas y análisis |
| `informe_resumen_tarea2.pdf` | Guía teórica con conceptos, analogías y resultados |
| `fig4_regularizacion.png` | Figura: 6 estrategias de regularización |
| `fig5_ruido_sesgo_varianza.png` | Figura: análisis sesgo-varianza con 3 niveles de ruido |

---

## Partes implementadas

### Parte 1 — MLP desde cero (NumPy)
Implementación completa sin frameworks: inicialización He, forward pass, backpropagation manual y SGD con minibatches. Tres arquitecturas evaluadas con diagnóstico de subajuste/sobreajuste.

### Parte 2 — PyTorch y Adam
Reimplementación equivalente con autodiferenciación. Comparación SGD vs Adam, sensibilidad al learning rate y efecto de regularización L2.

### Parte 3 — Verosimilitud log-normal
Derivación matemática completa de la pérdida NLL log-normal:

´´´

Entrada (5) → [64, ReLU] → [32, ReLU]   ← tronco compartido
↓              ↓              ↓
MSE (ΔΦ∈ℝ)    NLL log-n    Cross-entropy
(y>0)      (3 clases)

´´´

| Métrica | Red Mixta | Separados | Δ |
|---|---|---|---|
| Cambio — RMSE | **0.583** | 0.612 | +4.8% mejor |
| Biomarcador — MRE | **0.260** | 0.291 | +10.8% mejor |
| Clasificación — Acc. | 0.744 | **0.772** | −3.6% peor |
| Parámetros | **~4.133** | ~7.500 | −45% |

---

## Instalación

```bash
pip install numpy torch matplotlib scikit-learn seaborn jupyter
jupyter notebook tarea_2.ipynb
```

---

## Curso

**BME513 — Inteligencia Artificial en Salud**  
Profesor: Alejandro Veloz