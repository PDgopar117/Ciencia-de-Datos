"""
Práctica 8: Clasificación de Intrusiones con KNN y Logística
Unidad 3: Modelado – TecNM
Centro de Datos – Clasificación Seguro (0) vs Ataque (1)
Librerías: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# Paso 1: Simulación de Tráfico de Red
# ─────────────────────────────────────────────
np.random.seed(444)
n = 400

df_red = pd.DataFrame({
    'latencia':          np.random.normal(20, 5, n),
    'intentos_fallidos': np.random.poisson(1, n).astype(float),
    'tamano_paquete':    np.random.normal(500, 100, n),
})

# Lógica de clasificación: ataque si intentos > 2 OR latencia > 35
df_red['es_ataque'] = ((df_red['intentos_fallidos'] > 2) | (df_red['latencia'] > 35)).astype(int)

print("=== Primeras filas del dataset de red ===")
print(df_red.head().to_string())
print(f"\nDistribución de clases:\n{df_red['es_ataque'].value_counts().rename({0:'Seguro', 1:'Ataque'}).to_string()}")

# ─────────────────────────────────────────────
# Paso 2: Regresión Logística
# Equivalente a glm(..., family="binomial") de R
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 2: REGRESIÓN LOGÍSTICA")
print("="*60)

features = ['latencia', 'intentos_fallidos', 'tamano_paquete']
X_log = df_red[features].values
y_log = df_red['es_ataque'].values

scaler_log = StandardScaler()
X_log_sc   = scaler_log.fit_transform(X_log)

modelo_log = LogisticRegression(max_iter=1000, random_state=444)
modelo_log.fit(X_log_sc, y_log)

# Resumen de coeficientes (equivalente a summary() de R)
# Cálculo manual de errores estándar vía matriz de información de Fisher
y_prob  = modelo_log.predict_proba(X_log_sc)[:, 1]
W       = np.diag(y_prob * (1 - y_prob))
X_c     = np.column_stack([np.ones(len(X_log_sc)), X_log_sc])
cov_mat = np.linalg.inv(X_c.T @ W @ X_c)
se      = np.sqrt(np.diag(cov_mat))
coefs   = np.concatenate([[modelo_log.intercept_[0]], modelo_log.coef_[0]])
z_stats = coefs / se

print(f"\n  {'Variable':<22} {'Coeficiente':>13} {'Error Estándar':>16} {'z-estadístico':>15}")
print(f"  {'-'*66}")
nombres = ['Intercepto'] + features
for nombre, coef, se_v, z in zip(nombres, coefs, se, z_stats):
    sig = '***' if abs(z) > 3.29 else ('**' if abs(z) > 2.58 else ('*' if abs(z) > 1.96 else ''))
    print(f"  {nombre:<22} {coef:>13.4f} {se_v:>16.4f} {z:>15.3f}  {sig}")
print("  Significancia: *** p<0.001  ** p<0.01  * p<0.05")

y_pred_log = modelo_log.predict(X_log_sc)
print(f"\n  Accuracy regresión logística: {accuracy_score(y_log, y_pred_log):.4f}")
print("\n  Reporte de clasificación:")
print(classification_report(y_log, y_pred_log, target_names=['Seguro', 'Ataque']))

print("\n  Interpretación:")
print("  Si |z| de 'intentos_fallidos' > 1.96 → predictor crítico para detectar ataques")
print("  Si |z| de 'latencia' > 1.96           → predictor crítico")

# Gráfico de probabilidades predichas
plt.figure(figsize=(8, 4))
df_plot = pd.DataFrame({'latencia': df_red['latencia'], 'prob_ataque': y_prob,
                        'clase_real': df_red['es_ataque'].map({0:'Seguro', 1:'Ataque'})})
sns.scatterplot(data=df_plot, x='latencia', y='prob_ataque',
                hue='clase_real', palette={'Seguro':'steelblue','Ataque':'crimson'},
                alpha=0.5, s=20)
plt.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Umbral 0.5')
plt.xlabel('Latencia (ms)')
plt.ylabel('P(Ataque)')
plt.title('Regresión Logística – Probabilidad de Ataque vs Latencia')
plt.legend()
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 3: Clasificación KNN con validación cruzada
# Equivalente a train(..., method="knn") + plot() de caret
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 3: CLASIFICACIÓN KNN")
print("="*60)

X = df_red[features].values
y = df_red['es_ataque'].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=444)
param_grid = {'n_neighbors': range(1, 21, 2)}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=cv, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_sc, y)

cv_results = pd.DataFrame({
    'k':        grid_knn.cv_results_['param_n_neighbors'].data,
    'Accuracy': grid_knn.cv_results_['mean_test_score'],
    'SD':       grid_knn.cv_results_['std_test_score']
})
print("\nAccuracy por k (validación cruzada 5-fold):")
print(cv_results.to_string(index=False))
print(f"\n  k óptimo:            {grid_knn.best_params_['n_neighbors']}")
print(f"  Accuracy (k óptimo): {grid_knn.best_score_:.4f}")

# plot(modelo_knn) – Accuracy vs k
plt.figure(figsize=(9, 5))
plt.errorbar(cv_results['k'], cv_results['Accuracy'],
             yerr=cv_results['SD'], fmt='o-', color='steelblue',
             ecolor='lightgray', capsize=4, linewidth=2)
plt.axvline(x=grid_knn.best_params_['n_neighbors'], color='red', linestyle='--',
            label=f"k óptimo = {grid_knn.best_params_['n_neighbors']}")
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Accuracy (CV 5-fold)')
plt.title('KNN – Clasificación de Intrusiones: Precisión vs k\n'
          '(k bajo → sobreajuste / k alto → suavizado excesivo)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Matriz de confusión
y_pred_knn = grid_knn.predict(X_sc)
print("\n  Reporte KNN (k óptimo):")
print(classification_report(y, y_pred_knn, target_names=['Seguro', 'Ataque']))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred_knn, display_labels=['Seguro', 'Ataque'], cmap='Blues', ax=ax)
ax.set_title(f'Matriz de Confusión – KNN (k={grid_knn.best_params_["n_neighbors"]})')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 4: Clustering K-Means (sin etiqueta)
# Equivalente a kmeans(..., centers=2) + table() de R
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 4: CLUSTERING K-MEANS (sin supervisión)")
print("="*60)

datos_clus = scaler.transform(df_red[features])

np.random.seed(111)
kmeans = KMeans(n_clusters=2, random_state=111, n_init=10)
kmeans.fit(datos_clus)
df_red['cluster'] = kmeans.labels_

# Equivalente a table(df_red$es_ataque, clusters$cluster) de R
tabla = pd.crosstab(
    df_red['es_ataque'].map({0: 'Seguro', 1: 'Ataque'}),
    df_red['cluster'].map(lambda x: f'Cluster {x}'),
    rownames=['Realidad'], colnames=['K-Means']
)
print("\n  Tabla de contingencia (Realidad vs Cluster K-Means):")
print(tabla.to_string())
print("\n  Si un cluster agrupa mayoritariamente 'Ataques' → firma digital detectada sin supervisión")

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_real = {0: 'steelblue', 1: 'crimson'}
for clase, grupo in df_red.groupby('es_ataque'):
    axes[0].scatter(grupo['latencia'], grupo['intentos_fallidos'],
                    c=colors_real[clase], alpha=0.5, s=18,
                    label='Seguro' if clase == 0 else 'Ataque')
axes[0].set_title('Etiquetas Reales')
axes[0].set_xlabel('Latencia (ms)')
axes[0].set_ylabel('Intentos Fallidos')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

colors_cl = {0: '#2ECC71', 1: '#E74C3C'}
for cl, grupo in df_red.groupby('cluster'):
    axes[1].scatter(grupo['latencia'], grupo['intentos_fallidos'],
                    c=colors_cl[cl], alpha=0.5, s=18, label=f'Cluster {cl}')
axes[1].set_title('Clusters K-Means (sin etiqueta)')
axes[1].set_xlabel('Latencia (ms)')
axes[1].set_ylabel('Intentos Fallidos')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Clasificación de Intrusiones – Supervisado vs No Supervisado', fontsize=13)
plt.tight_layout()
plt.show()

mejor_cluster = tabla.idxmax(axis=1)
print(f"\n  Correspondencia cluster → clase:")
for clase, cl in mejor_cluster.items():
    n_correcto = tabla.loc[clase, cl]
    n_total    = tabla.loc[clase].sum()
    print(f"    {clase}: {cl}  →  {n_correcto}/{n_total} ({n_correcto/n_total*100:.1f}% bien clasificados)")
