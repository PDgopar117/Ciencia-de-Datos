"""
Práctica 5: Reducción de Dimensiones en Tráfico de Red (Cyber-Systems)
Unidad 2: Tratamiento de Datos – TecNM
Centro de Datos – 10 métricas por conexión
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Paso 1: Generación y Exploración de Datos
# ─────────────────────────────────────────────
np.random.seed(123)
n = 300

datos_red = pd.DataFrame({
    'duracion_ms':       np.random.normal(50, 10, n),
    'paquetes_enviados': np.random.normal(100, 20, n),
    'bytes_enviados':    np.nan,          # Se calculará con correlación
    'errores_checksum':  np.random.poisson(2, n).astype(float),
    'reintentos_tcp':    np.nan,
    'latencia_avg':      np.random.normal(15, 5, n),
    'jitter':            np.random.normal(2, 0.5, n),
    'carga_cpu_router':  np.nan,
    'uso_memoria_sw':    np.random.normal(40, 10, n),
    'peticiones_http':   np.random.normal(200, 50, n),
})

# Dependencias (redundancia intencional)
datos_red['bytes_enviados']   = datos_red['paquetes_enviados'] * 1500 + np.random.normal(0, 500, n)
datos_red['reintentos_tcp']   = datos_red['errores_checksum'] * 1.5  + np.random.normal(0, 0.5, n)
datos_red['carga_cpu_router'] = (datos_red['paquetes_enviados'] * 0.4) + (datos_red['latencia_avg'] * 0.2)

print("=== Primeras filas del dataset de red ===")
print(datos_red.head().to_string())
print("\nForma del dataset:", datos_red.shape)

# Correlación (confirmar la redundancia)
print("\n=== Correlación entre métricas de red ===")
print(datos_red.corr().round(3).to_string())

# ─────────────────────────────────────────────
# Paso 2: Estandarización y PCA (scale. = TRUE)
# ─────────────────────────────────────────────
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_red)

pca_red = PCA()
pca_red.fit(datos_escalados)

# ─────────────────────────────────────────────
# Paso 3: Análisis de Varianza Explicada
# ─────────────────────────────────────────────
var_prop = pca_red.explained_variance_ratio_
var_acum = np.cumsum(var_prop)
std_dev  = np.sqrt(pca_red.explained_variance_)

# Equivalente a summary(pca_red)
print("\n=== Resumen PCA – Tráfico de Red ===")
resumen = pd.DataFrame({
    'Desv. Estándar':        np.round(std_dev, 4),
    'Proporción Varianza':   np.round(var_prop, 4),
    'Proporción Acumulada':  np.round(var_acum, 4)
}, index=[f'PC{i+1}' for i in range(len(var_prop))])
print(resumen.to_string())

n_70 = int(np.argmax(var_acum >= 0.70)) + 1
n_85 = int(np.argmax(var_acum >= 0.85)) + 1
print(f"\n>>> PC1 + PC2 explican: {var_acum[1]*100:.1f}% de la varianza")
print(f">>> Componentes para 70%: {n_70} | Para 85%: {n_85}")

# Scree Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(var_prop) + 1), var_prop, 'o-', color='steelblue', linewidth=2)
axes[0].set_title('Scree Plot: Tráfico de Red')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Proporción de Varianza')
axes[0].grid(True, alpha=0.3)
# Añadir anotación del "codo"
axes[0].axhline(y=1 / len(var_prop), color='red', linestyle='--', label='Criterio Kaiser')
axes[0].legend()

axes[1].plot(range(1, len(var_acum) + 1), var_acum * 100, 's-', color='darkorange', linewidth=2)
axes[1].axhline(y=70, color='blue',  linestyle='--', label='Umbral 70%')
axes[1].axhline(y=85, color='green', linestyle='--', label='Umbral 85%')
axes[1].set_title('Varianza Acumulada – Red')
axes[1].set_xlabel('Componente Principal')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA – Métricas de Tráfico de Red', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 4: Loadings – interpretación de componentes
# ─────────────────────────────────────────────
loadings = pd.DataFrame(
    pca_red.components_[:2].T,
    index=datos_red.columns,
    columns=['PC1', 'PC2']
)
print("\n=== Cargas (Loadings) PC1 y PC2 ===")
print(loadings.round(3).to_string())
print("\nInterpretación:")
print("  PC1 – variables dominantes (|carga| > 0.30):")
top_pc1 = loadings['PC1'][loadings['PC1'].abs() > 0.30].sort_values(key=abs, ascending=False)
print(top_pc1.to_string())
print("  PC2 – variables dominantes (|carga| > 0.30):")
top_pc2 = loadings['PC2'][loadings['PC2'].abs() > 0.30].sort_values(key=abs, ascending=False)
print(top_pc2.to_string())

# ─────────────────────────────────────────────
# Paso 5: Biplot – Mapa de Estado de Red
# ─────────────────────────────────────────────
scores = pca_red.transform(datos_escalados)
comps  = pca_red.components_

fig, ax = plt.subplots(figsize=(11, 8))
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.35, s=18, c='steelblue', label='Conexiones')

scale = np.max(np.abs(scores[:, :2])) / np.max(np.abs(comps[:2]))
for i, var in enumerate(datos_red.columns):
    x_end = comps[0, i] * scale
    y_end = comps[1, i] * scale
    ax.annotate("", xy=(x_end, y_end), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8))
    ax.text(x_end * 1.1, y_end * 1.1, var, fontsize=8.5, color='darkred', ha='center')

ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
ax.set_xlabel(f'PC1 ({var_prop[0]*100:.1f}% varianza)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_prop[1]*100:.1f}% varianza)', fontsize=11)
ax.set_title('Mapa de Estado de Red – Biplot', fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()
