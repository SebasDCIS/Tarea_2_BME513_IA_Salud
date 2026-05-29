# Tarea 2 — BME513: Inteligencia Artificial en Salud

> **Predicción de concentración plasmática de biomarcador mediante redes neuronales feedforward**  
> Verosimilitud probabilística · Regularización · Análisis sesgo-varianza · Salidas mixtas

## Ver el notebook completo

[![nbviewer](https://img.shields.io/badge/Ver%20en-nbviewer-orange)](https://nbviewer.org/github/SebasDCIS/Tarea_2_BME513_IA_Salud/blob/main/tarea_2.ipynb)

---

## Resultado principal

| Modelo | MAE | Error Relativo | Preds. negativas |
|---|---|---|---|
| MLP + MSE (base NumPy) | 1.533 | 33.3% | — |
| MLP + MSE (PyTorch) | 1.536 | 31.9% | 3 |
| MLP + NLL log-normal | 1.370 | 24.7% | 0 |
| **MLP + NLL log-normal + L2 λ=5×10⁻³** | **1.173** | **20.5%** | **0** |

---

## Descripción del problema

Se predice la **concentración plasmática de un biomarcador** (variable
estrictamente positiva con distribución log-normal) a partir de 5 variables
clínicas de 1.200 pacientes sintéticos.

**Variables de entrada:**

| Variable | Descripción | Distribución |
|---|---|---|
| `edad` | Edad del paciente | U[18, 90] años |
| `dosis` | Dosis del medicamento | U[0, 1] |
| `horas` | Horas post-administración | U[0, 24] |
| `ir` | Índice de riesgo | N(0, 1) |
| `ib` | Inflamación base | Gamma(2, 1) |

**Proceso generador:**

```
y = exp( g(x) + ε ),   ε ~ N(0, 0.25²)
g(x) = −1.2 + 0.018·edad + 1.4·sin(π·dosis) + 0.03·horas
+ 0.55·ir + 0.35·√ib − 0.015·dosis·horas

```

---

## Estructura del repositorio

```
Tarea_2_BME513_IA_Salud/
│
├── tarea_2.ipynb                      ← Notebook principal (7 partes)
├── README.md
├── .gitignore
│
├── figures/
│   ├── output.png
│   ├── parte1.png
│   ├── parte2.png
│   ├── parte3.png
│   ├── parte4.png
│   ├── parte5.png
│
└── reports/
├── informe_tarea2.docx            ← Informe técnico completo
└── informe_resumen_tarea2.pdf     ← Guía teórica con analogías

```

---

## Partes implementadas

### Parte 1 — MLP desde cero (NumPy)

Implementación completa sin frameworks: inicialización He, forward pass,
backpropagation manual y SGD con minibatches. Se evalúan 3 arquitecturas.

| Arquitectura | Parámetros | MAE | Diagnóstico |
|---|---|---|---|
| A: 5→16→1 | 113 | 2.415 | Subajuste |
| **B: 5→64→32→1** | **2.497** | **1.533** | **Balance óptimo** |
| C: 5→128→64→32→1 | 11.137 | 1.489 | Sobreajuste incipiente |

### Parte 2 — Implementación en PyTorch

Reimplementación de la Arquitectura B con autodiferenciación y Adam.
Comparación SGD vs Adam, sensibilidad al learning rate y efecto de L2.

### Parte 3 — Verosimilitud log-normal

Derivación matemática completa desde la densidad de probabilidad:

```
Error total = σ²_ruido (irreducible) + Sesgo² + Varianza

```
| σ_ruido | Piso NLL | MAE Arq. B | Brecha train/val | Componente dominante |
|---|---|---|---|---|
| 0.10 | 0.010 | ~0.67 | 0.012 | Varianza del modelo |
| 0.25 | 0.063 | ~1.40 | 0.058 | Balance sesgo-varianza |
| 0.50 | 0.250 | ~3.00 | 0.220 | Error irreducible |

Con n=2400 la std entre semillas colapsa a 0.04–0.08 (−80% vs n=300).

### Parte 6 — Preguntas de discusión

10 preguntas con fundamento matemático, incluyendo:
- ReLU como función lineal por partes
- Caché del forward pass para backpropagation
- Regularización L2 como prior gaussiano MAP
- Equivalencia early stopping ≈ L2 (y cuándo no se cumple)
- Data leakage y rol del conjunto de validación
- Costo computacional del ensemble

### Parte 7 — Red con salidas mixtas

Tronco compartido (5→64→32) con 3 cabezales especializados simultáneos:

```
Entrada (5) → [64, ReLU] → [32, ReLU]      ← tronco compartido
↓              ↓                ↓
Cabezal 1          Cabezal 2         Cabezal 3
[16→1, lineal]    [16→1, exp(μ)]    [16→3, softmax]
Cambio ΔΦ ∈ ℝ     Biomarcador > 0    Estado clínico
MSE            NLL log-normal    Cross-entropy
λ₁ = 1.0            λ₂ = 5.0          λ₃ = 1.0

```

| Métrica | Red Mixta | Modelos separados | Δ |
|---|---|---|---|
| Cambio — MAE | **0.476** | 0.492 | +3.3% mejor |
| Biomarcador — E. Rel. | **0.260** | 0.291 | **+10.8% mejor** |
| Clasificación — Accuracy | 0.744 | **0.772** | −3.6% peor |
| Parámetros totales | **~4.133** | ~7.500 | −45% |

**Hallazgo:** el tronco compartido beneficia las tareas de regresión
(+10.8%) pero genera conflicto de gradientes con la clasificación (−3.6%).
La coherencia interna de predicciones es una ventaja clínica no capturada
por las métricas individuales.

---

## Hallazgos principales

1. **La función de pérdida importa más que la arquitectura**: cambiar
   MSE → NLL log-normal redujo el error en concentraciones bajas (y<2)
   de 54.4% a 28.3% y eliminó las predicciones negativas.

2. **La regularización explícita supera a la implícita**: L2 (−14.3%)
   superó a batch pequeño (+12.1% ❌) y data augmentation (+2.7% ❌).
   Las técnicas de visión no se transfieren automáticamente a datos tabulares.

3. **El error irreducible marca el piso**: con σ=0.50, ninguna arquitectura
   puede bajar el MAE de ~3.0. La regularización ayuda, pero no elimina
   el ruido intrínseco.

4. **Más datos reduce la varianza**: cuadruplicar n (300→2400) redujo la
   std entre semillas un 80%, más que cualquier estrategia de regularización.

---

## Instalación

```bash
pip install numpy torch matplotlib scikit-learn seaborn jupyter
jupyter notebook tarea_2.ipynb
```

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-yellow?logo=jupyter)

---

## Curso

**BME513 — Inteligencia Artificial en Salud**  
Profesor: Alejandro Veloz  
Referencia: *Understanding Deep Learning* — Simon Prince (2024)


