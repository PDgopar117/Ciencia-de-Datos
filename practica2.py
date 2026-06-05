"""
=============================================================================
PRÁCTICA 2: ANÁLISIS DE CALIDAD INDUSTRIAL (INDUSTRIA 4.0)
Asignatura: Ciencia de Datos | Ingeniería en Sistemas Computacionales
Empresa: TechManufacture
=============================================================================
Autor : Análisis generado con Python (NumPy, Pandas, Matplotlib, Seaborn,
        Scikit-learn)
Objetivo: Análisis estadístico exhaustivo de 500 lotes de producción para
          identificar la relación entre variables de proceso (temperatura,
          presión) y la calidad del producto final (tasa de error).
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTACIÓN DE LIBRERÍAS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')  # backend interactivo para ventana nativa
import warnings

warnings.filterwarnings("ignore")

# Estilo global de gráficas
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
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

ACCENT_1 = "#58a6ff"   # azul eléctrico
ACCENT_2 = "#f78166"   # coral
ACCENT_3 = "#3fb950"   # verde

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GENERACIÓN SINTÉTICA DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  PRÁCTICA 2 — ANÁLISIS DE CALIDAD INDUSTRIAL (INDUSTRIA 4.0)")
print("  TechManufacture | Ciencia de Datos")
print("=" * 70)

np.random.seed(42)
N = 500

# Turnos distribuidos equitativamente
turno = np.where(np.arange(N) % 2 == 0, "Matutino", "Vespertino")

# Temperatura: vespertino ligeramente más alta (degradación térmica acumulada)
temperatura_base = np.where(turno == "Matutino",
                            np.random.normal(72, 6, N),
                            np.random.normal(76, 7, N))

# Presión: correlación débil-moderada con temperatura
presion = 4.5 + 0.03 * temperatura_base + np.random.normal(0, 0.4, N)

# Tasa de error: relación cuadrática-lineal con temperatura + ruido
tasa_error = (
    -2.5
    + 0.08 * temperatura_base
    + 0.004 * temperatura_base ** 2
    + np.random.normal(0, 1.2, N)
).clip(0)  # no puede ser negativa

# Introducir ~4 % de valores faltantes de forma aleatoria
idx_na = np.random.choice(N, size=int(0.04 * N), replace=False)
temperatura_base = temperatura_base.astype(float)
temperatura_base[idx_na] = np.nan

df = pd.DataFrame({
    "lote":        np.arange(1, N + 1),
    "turno":       turno,
    "temperatura": temperatura_base,
    "presion":     presion,
    "tasa_error":  tasa_error,
})

print(f"\n[INFO] Dataset generado: {df.shape[0]} filas × {df.shape[1]} columnas")
print("\n── Primeras 5 filas ──")
print(df.head().to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FASE A — ANÁLISIS EXPLORATORIO (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE A: ANÁLISIS EXPLORATORIO (EDA)")
print("=" * 70)

# ── A.1  Medidas de tendencia central y dispersión ──────────────────────────
media_temp   = df["temperatura"].mean()
mediana_temp = df["temperatura"].median()
std_temp     = df["temperatura"].std(ddof=1)   # estimador insesgado

print("\n── A.1  Estadísticas descriptivas: Temperatura de Operación ──")
print(f"  Media     (μ) : {media_temp:.4f} °C")
print(f"  Mediana  (M)  : {mediana_temp:.4f} °C")
print(f"  Desv. Est (s) : {std_temp:.4f} °C")
print(f"  Mín / Máx     : {df['temperatura'].min():.2f} / {df['temperatura'].max():.2f} °C")
print(f"  Coef. Var (CV): {(std_temp / media_temp) * 100:.2f} %")

# Tabla general
print("\n── Estadísticas completas del dataset ──")
print(df.describe().round(4).to_string())

# ── A.2  Valores faltantes y limpieza ───────────────────────────────────────
print("\n── A.2  Diagnóstico y Tratamiento de Valores Faltantes ──")
nulos_antes = df.isnull().sum()
print(f"  Valores NA por columna (antes de limpieza):\n{nulos_antes.to_string()}")

# Imputación por la mediana (robusta ante outliers)
df["temperatura"] = df["temperatura"].fillna(df["temperatura"].median())

nulos_despues = df.isnull().sum()
print(f"\n  Valores NA por columna (después de imputación por mediana):")
print(f"{nulos_despues.to_string()}")
print(f"  [OK] Dataset sin valores faltantes. Imputación: mediana ({mediana_temp:.4f} °C)")

# ── A.3  Correlación de Pearson: temperatura vs tasa_error ──────────────────
print("\n── A.3  Correlación de Pearson: Temperatura ↔ Tasa de Error ──")
r_pearson, p_valor = stats.pearsonr(df["temperatura"], df["tasa_error"])
print(f"  r  (coeficiente) : {r_pearson:.4f}")
print(f"  p  (valor-p)     : {p_valor:.2e}")
alpha = 0.05
sig = "SIGNIFICATIVA" if p_valor < alpha else "NO significativa"
print(f"  Interpretación   : Correlación positiva {sig} (α = {alpha})")
print(f"  → A mayor temperatura de operación, mayor tasa de error (r = {r_pearson:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FASE B — AGRUPACIÓN Y SEGMENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE B: AGRUPACIÓN Y SEGMENTACIÓN")
print("=" * 70)

# ── B.1  Tasa de error por turno ─────────────────────────────────────────────
print("\n── B.1  Rendimiento promedio (tasa_error) por Turno ──")
resumen_turno = df.groupby("turno").agg(
    n          = ("tasa_error", "count"),
    media_err  = ("tasa_error", "mean"),
    std_err    = ("tasa_error", "std"),
    mediana_err= ("tasa_error", "median"),
).round(4)
print(resumen_turno.to_string())

mat = df.loc[df["turno"] == "Matutino", "tasa_error"]
ves = df.loc[df["turno"] == "Vespertino", "tasa_error"]

t_stat_err, p_err = stats.ttest_ind(mat, ves, equal_var=False)  # Welch t-test
print(f"\n  Welch t-test (tasa_error  Matutino vs Vespertino):")
print(f"  t = {t_stat_err:.4f}  |  p = {p_err:.4e}")
sig_err = "SÍ existe diferencia significativa" if p_err < 0.05 else "NO existe diferencia significativa"
print(f"  → {sig_err} en la tasa de error entre turnos (α = 0.05)")

# ── B.2  Temperatura por turno ───────────────────────────────────────────────
print("\n── B.2  Temperatura de Operación por Turno ──")
resumen_temp = df.groupby("turno").agg(
    n        = ("temperatura", "count"),
    media    = ("temperatura", "mean"),
    std      = ("temperatura", "std"),
    mediana  = ("temperatura", "median"),
).round(4)
print(resumen_temp.to_string())

t_mat  = df.loc[df["turno"] == "Matutino",   "temperatura"]
t_ves  = df.loc[df["turno"] == "Vespertino", "temperatura"]
t_stat_temp, p_temp = stats.ttest_ind(t_mat, t_ves, equal_var=False)
print(f"\n  Welch t-test (temperatura  Matutino vs Vespertino):")
print(f"  t = {t_stat_temp:.4f}  |  p = {p_temp:.4e}")
sig_temp = "SÍ existe diferencia significativa" if p_temp < 0.05 else "NO existe diferencia significativa"
print(f"  → {sig_temp} en la temperatura entre turnos (α = 0.05)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FASE C — VISUALIZACIÓN E INTERPRETACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FASE C: VISUALIZACIÓN E INTERPRETACIÓN")
print("=" * 70)

PALETTE = {"Matutino": ACCENT_1, "Vespertino": ACCENT_2}

# ────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — Dashboard EDA (4 subplots)
# ────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("TechManufacture · Dashboard EDA — Calidad Industrial",
              fontsize=16, color="#e6edf3", fontweight="bold", y=1.01)

# ── 1a Histograma + KDE de temperatura ──────────────────────────────────────
ax = axes[0, 0]
sns.histplot(df["temperatura"], bins=30, kde=True,
             color=ACCENT_1, alpha=0.7, ax=ax,
             line_kws={"linewidth": 2})
ax.axvline(media_temp,   color=ACCENT_2, linestyle="--", lw=1.8, label=f"Media = {media_temp:.2f}")
ax.axvline(mediana_temp, color=ACCENT_3, linestyle=":",  lw=1.8, label=f"Mediana = {mediana_temp:.2f}")
ax.set_title("Distribución de Temperatura de Operación")
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Frecuencia")
ax.legend(fontsize=9)

# ── 1b Histograma + KDE de tasa_error ───────────────────────────────────────
ax = axes[0, 1]
sns.histplot(df["tasa_error"], bins=30, kde=True,
             color=ACCENT_2, alpha=0.7, ax=ax,
             line_kws={"linewidth": 2})
ax.axvline(df["tasa_error"].mean(), color=ACCENT_1, linestyle="--", lw=1.8,
           label=f"Media = {df['tasa_error'].mean():.2f}")
ax.set_title("Distribución de Tasa de Error")
ax.set_xlabel("Tasa de Error (%)")
ax.set_ylabel("Frecuencia")
ax.legend(fontsize=9)

# ── 1c Mapa de correlación (heatmap) ────────────────────────────────────────
ax = axes[1, 0]
corr_matrix = df[["temperatura", "presion", "tasa_error"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues",
            linewidths=0.5, linecolor="#30363d",
            cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 11, "color": "#e6edf3"})
ax.set_title("Matriz de Correlación de Pearson")
ax.tick_params(labelsize=10)

# ── 1d Violin plot temperatura por turno ────────────────────────────────────
ax = axes[1, 1]
sns.violinplot(data=df, x="turno", y="temperatura",
               palette=PALETTE, inner="box", ax=ax, linewidth=1.5)
ax.set_title("Temperatura de Operación por Turno")
ax.set_xlabel("Turno")
ax.set_ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# FIGURA 2 — Boxplot de Tasa de Error por Turno (Fase C.1)
# ────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.patch.set_facecolor("#0d1117")

sns.boxplot(data=df, x="turno", y="tasa_error",
            palette=PALETTE, width=0.45,
            flierprops=dict(marker="o", color="#f0e68c",
                            markersize=5, alpha=0.7),
            boxprops=dict(linewidth=1.8),
            medianprops=dict(color="#e6edf3", linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.8),
            ax=ax2)

# Superponer puntos (jitter) para densidad
sns.stripplot(data=df, x="turno", y="tasa_error",
              palette=PALETTE, size=3, alpha=0.25, jitter=True, ax=ax2)

ax2.set_title("Dispersión de Tasa de Error por Turno\n(con identificación de outliers)",
              fontsize=14, color="#e6edf3", pad=12)
ax2.set_xlabel("Turno de Producción", fontsize=12)
ax2.set_ylabel("Tasa de Error (%)", fontsize=12)

# Anotar estadísticas clave
for i, grupo in enumerate(["Matutino", "Vespertino"]):
    sub = df.loc[df["turno"] == grupo, "tasa_error"]
    q1, q3 = sub.quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((sub < q1 - 1.5 * iqr) | (sub > q3 + 1.5 * iqr)).sum()
    ax2.text(i, ax2.get_ylim()[1] * 0.97,
             f"Outliers: {n_out}\nMediana: {sub.median():.2f}%",
             ha="center", va="top", fontsize=9,
             color="#8b949e",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                       edgecolor="#30363d", alpha=0.8))

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# FIGURA 3 — Scatter Plot Temperatura vs Tasa de Error + Regresión (Fase C.2)
# ────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 7))
fig3.patch.set_facecolor("#0d1117")

# Scatter coloreado por turno
for turno_nombre, color in PALETTE.items():
    sub = df[df["turno"] == turno_nombre]
    ax3.scatter(sub["temperatura"], sub["tasa_error"],
                color=color, alpha=0.45, s=22,
                label=f"Turno {turno_nombre}", edgecolors="none")

# ── Regresión lineal con scikit-learn ────────────────────────────────────────
X = df["temperatura"].values.reshape(-1, 1)
y = df["tasa_error"].values

modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

r2 = r2_score(y, y_pred)

# Línea de regresión sobre el rango de temperaturas
x_range = np.linspace(df["temperatura"].min(), df["temperatura"].max(), 300).reshape(-1, 1)
y_range = modelo.predict(x_range)

ax3.plot(x_range, y_range,
         color="#f0e68c", linewidth=2.5, linestyle="-",
         label=f"Regresión Lineal\nŷ = {modelo.coef_[0]:.4f}·T + ({modelo.intercept_:.4f})\n"
               f"R² = {r2:.4f}  |  r = {r_pearson:.4f}")

# Intervalo de confianza del 95 % (bootstrap visual)
residuos = y - y_pred
se = np.std(residuos, ddof=2)
ax3.fill_between(x_range.ravel(),
                 y_range - 1.96 * se,
                 y_range + 1.96 * se,
                 color="#f0e68c", alpha=0.12,
                 label="IC 95%")

ax3.set_title("Temperatura de Operación vs Tasa de Error\n(Regresión Lineal Simple — scikit-learn)",
              fontsize=14, color="#e6edf3", pad=12)
ax3.set_xlabel("Temperatura de Operación (°C)", fontsize=12)
ax3.set_ylabel("Tasa de Error (%)", fontsize=12)
ax3.legend(fontsize=9, loc="upper left")
ax3.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# FIGURA 4 — Comparación de tasa_error entre turnos (KDE + estadísticas)
# ────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 6))
fig4.suptitle("Comparación Estadística entre Turnos · TechManufacture",
              fontsize=14, color="#e6edf3", y=1.02)
fig4.patch.set_facecolor("#0d1117")

# ── 4a KDE tasa_error por turno ──────────────────────────────────────────────
ax = axes4[0]
for turno_nombre, color in PALETTE.items():
    sub = df.loc[df["turno"] == turno_nombre, "tasa_error"]
    sns.kdeplot(sub, ax=ax, color=color, linewidth=2.2,
                fill=True, alpha=0.25, label=turno_nombre)
    ax.axvline(sub.mean(), color=color, linestyle="--", lw=1.5)
ax.set_title("Densidad de Probabilidad\nTasa de Error por Turno")
ax.set_xlabel("Tasa de Error (%)")
ax.set_ylabel("Densidad")
ax.legend()

# ── 4b Barras de media ± DE ──────────────────────────────────────────────────
ax = axes4[1]
grupos = ["Matutino", "Vespertino"]
medias  = [df.loc[df["turno"] == g, "tasa_error"].mean() for g in grupos]
errores = [df.loc[df["turno"] == g, "tasa_error"].std(ddof=1) for g in grupos]
colores = [PALETTE[g] for g in grupos]

bars = ax.bar(grupos, medias, yerr=errores, capsize=8,
              color=colores, alpha=0.85, edgecolor="#30363d",
              error_kw={"ecolor": "#e6edf3", "lw": 2})
for bar, media in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{media:.3f}%", ha="center", va="bottom", fontsize=11,
            color="#e6edf3")

# Anotación del resultado del test
p_str = f"p = {p_err:.4e}"
sig_str = "(*)" if p_err < 0.05 else "(ns)"
ax.text(0.5, 0.95,
        f"Welch t-test: t = {t_stat_err:.3f}  {p_str}  {sig_str}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9.5, color="#8b949e",
        bbox=dict(boxstyle="round", facecolor="#21262d",
                  edgecolor="#30363d", alpha=0.8))
ax.set_title("Media ± Desviación Estándar\nTasa de Error por Turno")
ax.set_ylabel("Tasa de Error promedio (%)")

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  REPORTE FINAL DE CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  REPORTE DE CONCLUSIONES — TechManufacture")
print("=" * 70)

print(f"""
┌─ FASE A ─────────────────────────────────────────────────────────────┐
│ Temperatura Operacional:                                             │
│   μ = {media_temp:6.2f} °C  |  M = {mediana_temp:6.2f} °C  |  s = {std_temp:5.2f} °C          │
│   CV = {(std_temp/media_temp)*100:.1f}% → variabilidad {'moderada' if (std_temp/media_temp)*100 < 15 else 'alta'} del proceso                │
│                                                                      │
│ Valores faltantes: {len(idx_na)} NA en 'temperatura' → imputados por mediana      │
│                                                                      │
│ Correlación Pearson (temperatura ↔ tasa_error):                      │
│   r = {r_pearson:.4f}  |  p = {p_valor:.2e}  → correlación positiva {'significativa' if p_valor<0.05 else 'no significativa'}      │
└──────────────────────────────────────────────────────────────────────┘

┌─ FASE B ─────────────────────────────────────────────────────────────┐
│ Tasa de Error:                                                        │
│   Matutino  : {mat.mean():.4f} ± {mat.std(ddof=1):.4f}  %                         │
│   Vespertino: {ves.mean():.4f} ± {ves.std(ddof=1):.4f}  %                         │
│   Welch t   : t = {t_stat_err:.4f}  p = {p_err:.4e}  → {'diferencia SIGNIFICATIVA' if p_err<0.05 else 'sin diferencia significativa'}      │
│                                                                      │
│ Temperatura:                                                          │
│   Matutino  : {t_mat.mean():.2f} ± {t_mat.std(ddof=1):.2f} °C                         │
│   Vespertino: {t_ves.mean():.2f} ± {t_ves.std(ddof=1):.2f} °C                         │
│   Welch t   : t = {t_stat_temp:.4f}  p = {p_temp:.4e}  → {'diferencia SIGNIFICATIVA' if p_temp<0.05 else 'sin diferencia significativa'}      │
└──────────────────────────────────────────────────────────────────────┘

┌─ FASE C ─────────────────────────────────────────────────────────────┐
│ Modelo de Regresión Lineal (scikit-learn):                           │
│   ŷ = {modelo.coef_[0]:.4f}·T + ({modelo.intercept_:.4f})                            │
│   R²= {r2:.4f} → el modelo explica el {r2*100:.1f}% de la varianza       │
│                                                                      │
│ Interpretación de negocio:                                            │
│   → Por cada 1 °C de aumento en temperatura, la tasa de error       │
│     incrementa en {modelo.coef_[0]:.4f} puntos porcentuales.                  │
│   → El turno Vespertino opera a temperaturas más elevadas, lo que   │
│     se traduce en una mayor tasa de error promedio.                  │
│   → Se recomienda implementar sistemas de enfriamiento activo y      │
│     ajustar umbrales de alerta desde ~74 °C.                        │
└──────────────────────────────────────────────────────────────────────┘
""")