"""
=============================================================================
PRÁCTICA 3: ANÁLISIS DE EFICIENCIA EN LOGÍSTICA GLOBAL
Asignatura: Ciencia de Datos | Ingeniería en Sistemas Computacionales
Empresa: Global-Logistics ISC
=============================================================================
Librerías : NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
Objetivo  : Análisis estadístico de 600 envíos para identificar cuellos
            de botella en las modalidades de transporte Terrestre y Aéreo
            bajo distintas condiciones climáticas.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTACIÓN DE LIBRERÍAS
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")          # backend interactivo — ventana nativa del SO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ── Tema visual (dark industrial) ────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.55,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

AZUL    = "#58a6ff"   # Terrestre
NARANJA = "#f78166"   # Aéreo
VERDE   = "#3fb950"   # referencia / regresión
AMARILLO= "#f0e68c"   # anotaciones

PALETTE = {"Terrestre": AZUL, "Aéreo": NARANJA}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GENERACIÓN SINTÉTICA DEL DATASET  (600 envíos)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  PRÁCTICA 3 — ANÁLISIS DE EFICIENCIA EN LOGÍSTICA GLOBAL")
print("  Global-Logistics ISC | Ciencia de Datos")
print("=" * 70)

np.random.seed(7)
N = 600

# Variables categóricas
tipo_transporte = np.where(np.arange(N) % 2 == 0, "Terrestre", "Aéreo")
condicion_clima = np.random.choice(["Normal", "Lluvia", "Tormenta"], size=N,
                                   p=[0.60, 0.28, 0.12])

# Distancia_km: terrestre mayor rango que aéreo (rutas más largas por tierra)
distancia_base = np.where(
    tipo_transporte == "Terrestre",
    np.random.normal(620, 180, N),
    np.random.normal(980, 250, N),
).clip(50, 2500)

# Introducir ~5 % de NA en distancia_km (falla GPS)
idx_na = np.random.choice(N, size=int(0.05 * N), replace=False)
distancia_km = distancia_base.copy().astype(float)
distancia_km[idx_na] = np.nan

# Costo_usd: aéreo más caro; penalización climática
costo_base = np.where(tipo_transporte == "Terrestre",
                      np.random.normal(320, 60, N),
                      np.random.normal(780, 120, N))
penalizacion_costo = np.where(condicion_clima == "Tormenta", 80,
                      np.where(condicion_clima == "Lluvia",   30, 0))
costo_usd = (costo_base + penalizacion_costo).clip(100)

# Tiempo_entrega_hrs: función de distancia + modalidad + clima + ruido
velocidad = np.where(tipo_transporte == "Terrestre", 65, 480)   # km/h efectiva
penalizacion_tiempo = np.where(condicion_clima == "Tormenta", 8,
                       np.where(condicion_clima == "Lluvia",   3, 0))
tiempo_entrega_hrs = (
    distancia_base / velocidad
    + penalizacion_tiempo
    + np.random.normal(0, 1.5, N)
).clip(0.5)

df = pd.DataFrame({
    "envio_id":          np.arange(1, N + 1),
    "tipo_transporte":   tipo_transporte,
    "condicion_clima":   condicion_clima,
    "distancia_km":      distancia_km,
    "costo_usd":         costo_usd,
    "tiempo_entrega_hrs": tiempo_entrega_hrs,
})

print(f"\n[INFO] Dataset generado: {df.shape[0]} filas × {df.shape[1]} columnas")
print("\n── Primeras 5 filas ──")
print(df.head().to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FASE 1 — CALIDAD DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE 1: CALIDAD DE DATOS")
print("=" * 70)

# ── 1.1  Limpieza: imputación de NA por mediana ──────────────────────────────
print("\n── 1.1  Diagnóstico y Tratamiento de Valores Faltantes ──")
nulos_antes = df.isnull().sum()
print(f"  Valores NA por columna (antes):\n{nulos_antes.to_string()}")

mediana_dist = df["distancia_km"].median()
df["distancia_km"] = df["distancia_km"].fillna(mediana_dist)

nulos_despues = df.isnull().sum()
print(f"\n  Valores NA por columna (después):\n{nulos_despues.to_string()}")
print(f"  [OK] {len(idx_na)} registros imputados con mediana = {mediana_dist:.2f} km")

# ── 1.2  Descriptivos: tiempo_entrega_hrs ───────────────────────────────────
print("\n── 1.2  Estadísticas Descriptivas: Tiempo de Entrega ──")
media_t   = df["tiempo_entrega_hrs"].mean()
std_t     = df["tiempo_entrega_hrs"].std(ddof=1)
mediana_t = df["tiempo_entrega_hrs"].median()
cv_t      = (std_t / media_t) * 100

print(f"  Media     (μ) : {media_t:.4f} hrs")
print(f"  Mediana  (M)  : {mediana_t:.4f} hrs")
print(f"  Desv. Est (s) : {std_t:.4f} hrs")
print(f"  Coef. Var (CV): {cv_t:.2f} %")
print(f"  Mín / Máx     : {df['tiempo_entrega_hrs'].min():.2f} / "
      f"{df['tiempo_entrega_hrs'].max():.2f} hrs")

print("\n── Estadísticas completas del dataset ──")
print(df.describe().round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FASE 2 — ANÁLISIS DE RELACIONES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE 2: ANÁLISIS DE RELACIONES")
print("=" * 70)

r_pearson, p_pearson = stats.pearsonr(df["distancia_km"], df["tiempo_entrega_hrs"])
print(f"\n── Correlación de Pearson: distancia_km ↔ tiempo_entrega_hrs ──")
print(f"  r  = {r_pearson:.4f}")
print(f"  p  = {p_pearson:.2e}")

sig = "SIGNIFICATIVA" if p_pearson < 0.05 else "NO significativa"
fuerza = ("muy fuerte" if abs(r_pearson) >= 0.8 else
          "fuerte"     if abs(r_pearson) >= 0.6 else
          "moderada"   if abs(r_pearson) >= 0.4 else "débil")

print(f"  Interpretación: correlación positiva {fuerza} y {sig} (α = 0.05)")
print(f"  → La distancia explica parcialmente el tiempo; la modalidad y el")
print(f"    clima introducen varianza adicional que reduce la correlación global.")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FASE 3 — COMPARATIVA DE MODALIDADES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE 3: COMPARATIVA DE MODALIDADES")
print("=" * 70)

# ── 3.1  Agrupación ──────────────────────────────────────────────────────────
print("\n── 3.1  Promedio de Tiempo de Entrega y Costo por Tipo de Transporte ──")
resumen = df.groupby("tipo_transporte").agg(
    n               = ("tiempo_entrega_hrs", "count"),
    tiempo_medio    = ("tiempo_entrega_hrs", "mean"),
    tiempo_std      = ("tiempo_entrega_hrs", "std"),
    tiempo_mediana  = ("tiempo_entrega_hrs", "median"),
    costo_medio_usd = ("costo_usd", "mean"),
    costo_std_usd   = ("costo_usd", "std"),
).round(4)
print(resumen.to_string())

# ── 3.2  Prueba de hipótesis (Welch t-test) ──────────────────────────────────
print("\n── 3.2  Prueba de Hipótesis: Tiempo de Entrega Terrestre vs Aéreo ──")
terrestre = df.loc[df["tipo_transporte"] == "Terrestre", "tiempo_entrega_hrs"]
aereo     = df.loc[df["tipo_transporte"] == "Aéreo",     "tiempo_entrega_hrs"]

t_stat, p_val = stats.ttest_ind(terrestre, aereo, equal_var=False)  # Welch

print(f"  H₀: μ_Terrestre = μ_Aéreo  (no existe diferencia significativa)")
print(f"  H₁: μ_Terrestre ≠ μ_Aéreo  (existe diferencia significativa)")
print(f"\n  Welch t-test:")
print(f"  t  = {t_stat:.4f}")
print(f"  p  = {p_val:.2e}")

if p_val < 0.05:
    decision = "RECHAZAR H₀"
    conclusion = (f"Existe una diferencia estadísticamente significativa "
                  f"(α=0.05) entre los tiempos de entrega Terrestre "
                  f"({terrestre.mean():.2f} hrs) y Aéreo ({aereo.mean():.2f} hrs).")
else:
    decision = "NO RECHAZAR H₀"
    conclusion = "No existe evidencia suficiente para afirmar diferencia significativa."

print(f"  Decisión    : {decision}")
print(f"  Conclusión  : {conclusion}")

# Tamaño del efecto (Cohen's d)
pooled_std = np.sqrt((terrestre.std(ddof=1)**2 + aereo.std(ddof=1)**2) / 2)
cohen_d    = (terrestre.mean() - aereo.mean()) / pooled_std
magnitud   = ("grande" if abs(cohen_d) >= 0.8 else
              "mediano" if abs(cohen_d) >= 0.5 else "pequeño")
print(f"  Cohen's d   : {cohen_d:.4f} → efecto {magnitud}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FASE 4 — VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE 4: VISUALIZACIÓN")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Dashboard EDA (2×2)
# ════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor("#0d1117")
fig1.suptitle("Global-Logistics ISC · Dashboard EDA — Eficiencia Logística",
              fontsize=15, color="#e6edf3", fontweight="bold")

# ── 1a Histograma + KDE tiempo_entrega_hrs ───────────────────────────────────
ax = axes[0, 0]
sns.histplot(df["tiempo_entrega_hrs"], bins=35, kde=True,
             color=AZUL, alpha=0.7, ax=ax, line_kws={"linewidth": 2})
ax.axvline(media_t,   color=NARANJA, linestyle="--", lw=1.8,
           label=f"Media = {media_t:.2f} hrs")
ax.axvline(mediana_t, color=VERDE,   linestyle=":",  lw=1.8,
           label=f"Mediana = {mediana_t:.2f} hrs")
ax.set_title("Distribución: Tiempo de Entrega")
ax.set_xlabel("Tiempo de Entrega (hrs)")
ax.set_ylabel("Frecuencia")
ax.legend(fontsize=9)

# ── 1b Histograma distancia_km por modalidad ─────────────────────────────────
ax = axes[0, 1]
for mod, color in PALETTE.items():
    sub = df.loc[df["tipo_transporte"] == mod, "distancia_km"]
    sns.histplot(sub, bins=28, kde=True, color=color, alpha=0.55,
                 label=mod, ax=ax, line_kws={"linewidth": 1.8})
ax.set_title("Distribución: Distancia por Modalidad")
ax.set_xlabel("Distancia (km)")
ax.set_ylabel("Frecuencia")
ax.legend(fontsize=9)

# ── 1c Mapa de calor — correlaciones ─────────────────────────────────────────
ax = axes[1, 0]
corr = df[["distancia_km", "costo_usd", "tiempo_entrega_hrs"]].corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues",
            linewidths=0.5, linecolor="#30363d",
            cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 11, "color": "#e6edf3"})
ax.set_title("Matriz de Correlación de Pearson")
ax.tick_params(labelsize=9)

# ── 1d Violin plot costo_usd por modalidad ───────────────────────────────────
ax = axes[1, 1]
sns.violinplot(data=df, x="tipo_transporte", y="costo_usd",
               palette=PALETTE, inner="box", ax=ax, linewidth=1.5)
ax.set_title("Distribución de Costo por Modalidad")
ax.set_xlabel("Modalidad de Transporte")
ax.set_ylabel("Costo (USD)")

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Scatter Plot: Distancia vs Tiempo + Regresión (Fase 4.1)
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(11, 7))
fig2.patch.set_facecolor("#0d1117")

# Puntos por modalidad
for mod, color in PALETTE.items():
    sub = df[df["tipo_transporte"] == mod]
    ax2.scatter(sub["distancia_km"], sub["tiempo_entrega_hrs"],
                color=color, alpha=0.40, s=20, label=mod, edgecolors="none")

# Regresión lineal global (scikit-learn)
X = df["distancia_km"].values.reshape(-1, 1)
y = df["tiempo_entrega_hrs"].values
modelo = LinearRegression().fit(X, y)
r2 = r2_score(y, modelo.predict(X))

x_range = np.linspace(df["distancia_km"].min(),
                      df["distancia_km"].max(), 300).reshape(-1, 1)
y_range = modelo.predict(x_range)

ax2.plot(x_range, y_range, color=AMARILLO, lw=2.5,
         label=(f"Regresión Global\n"
                f"ŷ = {modelo.coef_[0]:.4f}·d + {modelo.intercept_:.4f}\n"
                f"R² = {r2:.4f}  |  r = {r_pearson:.4f}"))

# IC 95 % visual
residuos_std = np.std(y - modelo.predict(X), ddof=2)
ax2.fill_between(x_range.ravel(),
                 y_range - 1.96 * residuos_std,
                 y_range + 1.96 * residuos_std,
                 color=AMARILLO, alpha=0.10, label="IC 95%")

# Líneas de regresión por modalidad
for mod, color in PALETTE.items():
    sub = df[df["tipo_transporte"] == mod]
    Xm  = sub["distancia_km"].values.reshape(-1, 1)
    ym  = sub["tiempo_entrega_hrs"].values
    m   = LinearRegression().fit(Xm, ym)
    xr  = np.linspace(Xm.min(), Xm.max(), 200).reshape(-1, 1)
    ax2.plot(xr, m.predict(xr), color=color, lw=1.8,
             linestyle="--", alpha=0.85, label=f"Regresión {mod}")

ax2.set_title("Distancia de Envío vs Tiempo de Entrega\n"
              "(diferenciado por Modalidad de Transporte)",
              fontsize=14, color="#e6edf3", pad=12)
ax2.set_xlabel("Distancia (km)")
ax2.set_ylabel("Tiempo de Entrega (hrs)")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Boxplot: Tiempo de Entrega por Modalidad (Fase 4.2)
# ════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(9, 7))
fig3.patch.set_facecolor("#0d1117")

sns.boxplot(
    data=df, x="tipo_transporte", y="tiempo_entrega_hrs",
    palette=PALETTE, width=0.45,
    flierprops =dict(marker="D", color=AMARILLO, markersize=5, alpha=0.75),
    boxprops   =dict(linewidth=1.8),
    medianprops=dict(color="#e6edf3", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops   =dict(linewidth=1.8),
    ax=ax3,
)
sns.stripplot(data=df, x="tipo_transporte", y="tiempo_entrega_hrs",
              palette=PALETTE, size=3, alpha=0.22, jitter=True, ax=ax3)

# Anotar outliers + resultado del t-test
for i, mod in enumerate(["Terrestre", "Aéreo"]):
    sub  = df.loc[df["tipo_transporte"] == mod, "tiempo_entrega_hrs"]
    q1, q3 = sub.quantile([0.25, 0.75])
    iqr  = q3 - q1
    n_out = ((sub < q1 - 1.5 * iqr) | (sub > q3 + 1.5 * iqr)).sum()
    ax3.text(i, ax3.get_ylim()[1] * 0.98,
             f"Outliers: {n_out}\nMediana: {sub.median():.2f} hrs",
             ha="center", va="top", fontsize=9, color="#8b949e",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                       edgecolor="#30363d", alpha=0.85))

# Barra de significancia
y_bar = df["tiempo_entrega_hrs"].max() * 1.04
ax3.plot([0, 1], [y_bar, y_bar], color="#e6edf3", lw=1.4)
ax3.plot([0, 0], [y_bar * 0.98, y_bar], color="#e6edf3", lw=1.4)
ax3.plot([1, 1], [y_bar * 0.98, y_bar], color="#e6edf3", lw=1.4)
sig_label = f"p = {p_val:.2e}  {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'}"
ax3.text(0.5, y_bar * 1.005, sig_label,
         ha="center", va="bottom", fontsize=10, color=AMARILLO)

ax3.set_title("Comparación de Tiempos de Entrega por Modalidad\n"
              "(Boxplot con detección de valores atípicos)",
              fontsize=14, color="#e6edf3", pad=12)
ax3.set_xlabel("Modalidad de Transporte")
ax3.set_ylabel("Tiempo de Entrega (hrs)")
ax3.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Impacto del clima por modalidad (KDE + barras agrupadas)
# ════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.patch.set_facecolor("#0d1117")
fig4.suptitle("Impacto de la Condición Climática sobre el Tiempo de Entrega",
              fontsize=14, color="#e6edf3")

# ── 4a KDE por modalidad ─────────────────────────────────────────────────────
ax = axes4[0]
for mod, color in PALETTE.items():
    sub = df.loc[df["tipo_transporte"] == mod, "tiempo_entrega_hrs"]
    sns.kdeplot(sub, ax=ax, color=color, linewidth=2.2,
                fill=True, alpha=0.25, label=mod)
    ax.axvline(sub.mean(), color=color, linestyle="--", lw=1.5,
               label=f"μ {mod} = {sub.mean():.2f} hrs")
ax.set_title("Densidad de Probabilidad\nTiempo de Entrega por Modalidad")
ax.set_xlabel("Tiempo de Entrega (hrs)")
ax.set_ylabel("Densidad")
ax.legend(fontsize=9)

# ── 4b Barras agrupadas: clima × modalidad ───────────────────────────────────
ax = axes4[1]
resumen_clima = (df.groupby(["condicion_clima", "tipo_transporte"])
                   ["tiempo_entrega_hrs"].mean().unstack())

x       = np.arange(len(resumen_clima))
ancho   = 0.35
climas  = resumen_clima.index.tolist()
colores = [AZUL, NARANJA]

for j, (mod, color) in enumerate(zip(resumen_clima.columns, colores)):
    valores = resumen_clima[mod].values
    barras  = ax.bar(x + j * ancho, valores, ancho,
                     label=mod, color=color, alpha=0.85,
                     edgecolor="#30363d")
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color="#e6edf3")

ax.set_xticks(x + ancho / 2)
ax.set_xticklabels(climas)
ax.set_title("Tiempo Promedio de Entrega\npor Clima y Modalidad")
ax.set_xlabel("Condición Climática")
ax.set_ylabel("Tiempo Promedio (hrs)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  REPORTE FINAL DE CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  REPORTE DE CONCLUSIONES — Global-Logistics ISC")
print("=" * 70)

print(f"""
┌─ FASE 1 · CALIDAD DE DATOS ──────────────────────────────────────────┐
│ Valores faltantes detectados: {len(idx_na)} NA en 'distancia_km' (falla GPS)  │
│ Técnica de imputación: mediana = {mediana_dist:.2f} km (robusta ante outliers)  │
│                                                                       │
│ Tiempo de Entrega:                                                    │
│   μ = {media_t:6.2f} hrs  |  M = {mediana_t:6.2f} hrs  |  s = {std_t:5.2f} hrs       │
│   CV = {cv_t:.1f}% → variabilidad {'moderada' if cv_t < 30 else 'alta'} en los tiempos de entrega    │
└───────────────────────────────────────────────────────────────────────┘

┌─ FASE 2 · ANÁLISIS DE RELACIONES ────────────────────────────────────┐
│ Pearson (distancia ↔ tiempo): r = {r_pearson:.4f}  p = {p_pearson:.2e}           │
│ La correlación es {fuerza} — la modalidad y el clima                 │
│ introducen varianza que impide una correlación perfecta.              │
└───────────────────────────────────────────────────────────────────────┘

┌─ FASE 3 · COMPARATIVA DE MODALIDADES ────────────────────────────────┐
│ Terrestre : {terrestre.mean():.2f} ± {terrestre.std(ddof=1):.2f} hrs  │
│ Aéreo     : {aereo.mean():.2f} ± {aereo.std(ddof=1):.2f} hrs  │
│ Welch t   : t = {t_stat:.4f}  p = {p_val:.4e}                             │
│ Decisión  : {decision}                                       │
│ Cohen's d : {cohen_d:.4f} (efecto {magnitud})                               │
│                                                                       │
│ Costo promedio:                                                       │
│   Terrestre: ${df.loc[df['tipo_transporte']=='Terrestre','costo_usd'].mean():.2f} USD  │
│   Aéreo    : ${df.loc[df['tipo_transporte']=='Aéreo','costo_usd'].mean():.2f} USD  │
└───────────────────────────────────────────────────────────────────────┘

┌─ FASE 4 · MODELO DE REGRESIÓN ───────────────────────────────────────┐
│ ŷ = {modelo.coef_[0]:.5f}·distancia + {modelo.intercept_:.4f}                         │
│ R² = {r2:.4f} → el modelo explica el {r2*100:.1f}% de la varianza           │
│                                                                       │
│ Recomendaciones operativas:                                           │
│ → Priorizar transporte Aéreo para envíos urgentes (> umbral crítico) │
│ → Implementar alertas preventivas ante condición "Tormenta"          │
│ → Revisar rutas terrestres con distancias > 800 km (alto tiempo)     │
└───────────────────────────────────────────────────────────────────────┘
""")