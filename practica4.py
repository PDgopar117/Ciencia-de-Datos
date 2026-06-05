"""
Práctica 4: Reducción de Dimensionalidad en Monitoreo Industrial
Unidad 2: Tratamiento de Datos - Ciencia de Datos
Planta "Metal-Tech" – 12 sensores de caldera
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# FASE 1: Preparación - Generación del dataset
# ─────────────────────────────────────────────
np.random.seed(42)
n = 200

temp1      = np.random.normal(100, 5, n)
temp2      = temp1 + np.random.normal(0, 1, n)
presion1   = np.random.normal(50, 10, n)
vibracion  = (temp1 * 0.5) + np.random.normal(0, 2, n)

datos_sensores = pd.DataFrame({
    'temp_nucleo':      temp1,
    'temp_escape':      temp2,
    'presion_interna':  presion1,
    'vibracion_motor':  vibracion,
    'flujo_gas':        np.random.normal(20, 2, n),
    'oxigeno':          np.random.normal(15, 1, n),
    'co2':              temp1 * 0.2 + np.random.normal(0, 0.5, n),
    'humedad':          np.random.uniform(30, 40, n),
    'voltaje':          np.random.normal(220, 2, n),
    'corriente':        np.random.normal(15, 0.5, n),
    'ruido_db':         vibracion * 1.2 + np.random.normal(0, 1, n),
    'eficiencia':       100 - (temp1 * 0.1)
})

print("=== Primeras filas del dataset ===")
print(datos_sensores.head().to_string())

# Correlación inicial entre sensores
print("\n=== Matriz de Correlación ===")
corr = datos_sensores.corr().round(3)
print(corr.to_string())

# Heatmap de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, square=True)
plt.title("Correlación entre Sensores – Metal-Tech")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# FASE 2: Ejecución del PCA (scale. = TRUE)
# ─────────────────────────────────────────────
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_sensores)

pca = PCA()
pca.fit(datos_escalados)

# Equivalente a summary(pca_modelo)
var_prop  = pca.explained_variance_ratio_
var_acum  = np.cumsum(var_prop)
std_dev   = np.sqrt(pca.explained_variance_)

print("\n=== Resumen PCA (equivalente a summary()) ===")
resumen = pd.DataFrame({
    'Desv. Estándar':        np.round(std_dev, 4),
    'Proporción Varianza':   np.round(var_prop, 4),
    'Proporción Acumulada':  np.round(var_acum, 4)
}, index=[f'PC{i+1}' for i in range(len(var_prop))])
print(resumen.to_string())

# ─────────────────────────────────────────────
# FASE 3: Scree Plot + Criterio Kaiser + 85%
# ─────────────────────────────────────────────
n_componentes = len(var_prop)
n_85 = int(np.argmax(var_acum >= 0.85)) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree Plot (varianza por componente)
axes[0].plot(range(1, n_componentes + 1), var_prop, 'o-', color='steelblue', linewidth=2)
axes[0].axhline(y=1 / n_componentes, color='red', linestyle='--', label='Criterio Kaiser (1/p)')
axes[0].set_title('Gráfico de Sedimentación (Scree Plot)')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Proporción de Varianza Explicada')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Varianza acumulada
axes[1].plot(range(1, n_componentes + 1), var_acum * 100, 's-', color='darkorange', linewidth=2)
axes[1].axhline(y=85, color='green', linestyle='--', label='Umbral 85%')
axes[1].axvline(x=n_85, color='purple', linestyle=':', label=f'PC{n_85} ({var_acum[n_85-1]*100:.1f}%)')
axes[1].set_title('Varianza Acumulada')
axes[1].set_xlabel('Componente Principal')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA – Sensores Metal-Tech', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n>>> Se necesitan {n_85} componentes para alcanzar el 85% de varianza "
      f"({var_acum[n_85-1]*100:.2f}% acumulado)")

# ─────────────────────────────────────────────
# Cargas (Loadings) – PC1 y PC2
# ─────────────────────────────────────────────
# pca.components_ tiene forma (n_components, n_features)
loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=datos_sensores.columns,
    columns=['PC1', 'PC2']
)
print("\n=== Cargas (Loadings) PC1 y PC2 ===")
print(loadings.round(3).to_string())
print("\n>>> Sensores con mayor peso en PC1 (|carga| > 0.3):")
print(loadings['PC1'][loadings['PC1'].abs() > 0.3].sort_values(key=abs, ascending=False))

# ─────────────────────────────────────────────
# FASE 4: Biplot
# ─────────────────────────────────────────────
scores  = pca.transform(datos_escalados)   # coordenadas de observaciones
comps   = pca.components_                  # loadings (eigenvectores)

fig, ax = plt.subplots(figsize=(11, 8))

# Observaciones (lotes de producción)
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.4, s=18, c='steelblue', label='Lotes')

# Flechas de variables (escaladas para visibilidad)
scale = np.max(np.abs(scores[:, :2])) / np.max(np.abs(comps[:2]))
for i, var in enumerate(datos_sensores.columns):
    x_end = comps[0, i] * scale
    y_end = comps[1, i] * scale
    ax.annotate("", xy=(x_end, y_end), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(x_end * 1.08, y_end * 1.08, var, fontsize=8.5, color='darkred', ha='center')

ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
ax.set_xlabel(f'PC1 ({var_prop[0]*100:.1f}% varianza)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_prop[1]*100:.1f}% varianza)', fontsize=11)
ax.set_title('Biplot de Sensores Industriales – Metal-Tech', fontsize=13)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
