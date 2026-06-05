"""
Práctica 6: Optimización de Sensores en Smart Cities
Unidad 2: Tratamiento de Datos – TecNM
Red LoRaWAN – 10 sensores por intersección urbana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Paso 1: Generación de Datos Urbanos
# ─────────────────────────────────────────────
np.random.seed(456)
n = 400

datos_urbanos = pd.DataFrame({
    'temp_ambiente':     np.random.normal(28, 4, n),
    'humedad_rel':       np.random.normal(60, 10, n),
    'co2_ppm':           np.nan,    # Relacionado con tráfico
    'no2_ppb':           np.nan,    # Relacionado con tráfico
    'particulas_pm25':   np.nan,    # Relacionado con CO2
    'nivel_ruido_db':    np.nan,    # Relacionado con vehículos
    'densidad_vehiculos': np.random.normal(120, 30, n),
    'velocidad_viento':  np.random.normal(15, 5, n),
    'radiacion_solar':   np.random.normal(800, 100, n),
    'conteo_peatones':   np.random.normal(50, 15, n),
})

# Redundancia y correlación (lógica de ciudad)
datos_urbanos['co2_ppm']        = (datos_urbanos['densidad_vehiculos'] * 3.5) + np.random.normal(300, 50, n)
datos_urbanos['no2_ppb']        = (datos_urbanos['co2_ppm'] * 0.1)            + np.random.normal(5, 2, n)
datos_urbanos['particulas_pm25']= (datos_urbanos['co2_ppm'] * 0.05)           + np.random.normal(10, 3, n)
datos_urbanos['nivel_ruido_db'] = (datos_urbanos['densidad_vehiculos'] * 0.2) + 50 + np.random.normal(0, 5, n)

print("=== Primeras filas del dataset urbano ===")
print(datos_urbanos.head().to_string())

# Heatmap de correlación para ver redundancias
plt.figure(figsize=(10, 8))
sns.heatmap(datos_urbanos.corr().round(2), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5, square=True)
plt.title("Correlación entre Sensores Urbanos – Smart City")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 2: Estandarización (scale. = TRUE)
# ─────────────────────────────────────────────
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_urbanos)

pca_ciudad = PCA()
pca_ciudad.fit(datos_escalados)

# ─────────────────────────────────────────────
# Paso 3: Análisis de Varianza Acumulada
# ─────────────────────────────────────────────
var_prop = pca_ciudad.explained_variance_ratio_
var_acum = np.cumsum(var_prop)
std_dev  = np.sqrt(pca_ciudad.explained_variance_)

# Equivalente a summary(pca_ciudad)
print("\n=== Resumen PCA – Smart City ===")
resumen = pd.DataFrame({
    'Desv. Estándar':        np.round(std_dev, 4),
    'Proporción Varianza':   np.round(var_prop, 4),
    'Proporción Acumulada':  np.round(var_acum, 4)
}, index=[f'PC{i+1}' for i in range(len(var_prop))])
print(resumen.to_string())

n_85 = int(np.argmax(var_acum >= 0.85)) + 1
ahorro_pct = (1 - n_85 / datos_urbanos.shape[1]) * 100
print(f"\n>>> Componentes para 85% de varianza: {n_85} (de {datos_urbanos.shape[1]} sensores)")
print(f">>> Ahorro en transmisión de datos: {ahorro_pct:.0f}%")
print(f">>> PC1 explica el {var_prop[0]*100:.1f}% de la varianza total")

# Scree Plot con criterio Kaiser y umbral 85%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(var_prop) + 1), var_prop, 'o-', color='teal', linewidth=2)
axes[0].axhline(y=1 / len(var_prop), color='red', linestyle='--', label='Criterio Kaiser')
axes[0].set_title('Scree Plot: Sensores Smart City')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Varianza Explicada')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(var_acum) + 1), var_acum * 100, 's-', color='darkorange', linewidth=2)
axes[1].axhline(y=85, color='green', linestyle='--', label='Umbral 85%')
axes[1].axvline(x=n_85, color='purple', linestyle=':', label=f'PC{n_85} – {var_acum[n_85-1]*100:.1f}%')
axes[1].fill_between(range(1, n_85 + 1), var_acum[:n_85] * 100, alpha=0.15, color='green')
axes[1].set_title('Varianza Acumulada – Smart City')
axes[1].set_xlabel('Componente Principal')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA – Red de Sensores Urbanos (LoRaWAN)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 4: Loadings – traducción a ingeniería urbana
# ─────────────────────────────────────────────
loadings = pd.DataFrame(
    pca_ciudad.components_[:2].T,
    index=datos_urbanos.columns,
    columns=['PC1', 'PC2']
)
print("\n=== Cargas (Loadings) PC1 y PC2 ===")
print(loadings.round(3).to_string())

print("\n  PC1 – Actividad Vehicular e Impacto (cargas altas):")
top1 = loadings['PC1'][loadings['PC1'].abs() > 0.25].sort_values(key=abs, ascending=False)
print(top1.to_string())

print("\n  PC2 – Factor Climático (cargas altas):")
top2 = loadings['PC2'][loadings['PC2'].abs() > 0.25].sort_values(key=abs, ascending=False)
print(top2.to_string())

# Heatmap de loadings
plt.figure(figsize=(7, 6))
sns.heatmap(loadings, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Mapa de Cargas – PC1 y PC2')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 5: Biplot de Diagnóstico Urbano
# ─────────────────────────────────────────────
scores = pca_ciudad.transform(datos_escalados)
comps  = pca_ciudad.components_

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.3, s=18, c='teal', label='Nodos urbanos')

scale = np.max(np.abs(scores[:, :2])) / np.max(np.abs(comps[:2]))
colors_var = {
    'co2_ppm': 'crimson', 'no2_ppb': 'crimson', 'particulas_pm25': 'crimson',
    'nivel_ruido_db': 'crimson', 'densidad_vehiculos': 'crimson',
    'temp_ambiente': 'navy', 'humedad_rel': 'navy',
    'velocidad_viento': 'navy', 'radiacion_solar': 'navy',
    'conteo_peatones': 'forestgreen'
}
for i, var in enumerate(datos_urbanos.columns):
    x_end = comps[0, i] * scale
    y_end = comps[1, i] * scale
    color = colors_var.get(var, 'gray')
    ax.annotate("", xy=(x_end, y_end), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    ax.text(x_end * 1.1, y_end * 1.1, var, fontsize=8.5, color=color, ha='center')

# Leyenda manual de grupos
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='>', color='crimson', lw=1.5, label='Contaminación vehicular'),
    Line2D([0], [0], marker='>', color='navy',    lw=1.5, label='Factor climático'),
    Line2D([0], [0], marker='>', color='forestgreen', lw=1.5, label='Actividad peatonal'),
]
ax.legend(handles=legend_elements, loc='upper left')

ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
ax.set_xlabel(f'PC1 – Actividad Vehicular ({var_prop[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 – Factor Climático ({var_prop[1]*100:.1f}%)', fontsize=11)
ax.set_title('Mapa Biplot de Estado Urbano – Smart City', fontsize=13)
plt.tight_layout()
plt.show()