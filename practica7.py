"""
Práctica 7: Implementación de Modelos de Machine Learning
Unidad 3: Modelado – TecNM
DataSystems S.A. – Predicción de salario, retención y segmentación de empleados
Librerías: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# ─────────────────────────────────────────────
# Paso 1: Preparación del Dataset
# ─────────────────────────────────────────────
np.random.seed(789)
n = 500

df_empleados = pd.DataFrame({
    'experiencia':          np.random.normal(5, 2, n),
    'certificaciones':      np.random.poisson(3, n).astype(float),
    'habilidades_sociales': np.random.uniform(1, 10, n),
    'remoto':               np.random.choice([0, 1], n).astype(float),
})

# Variable objetivo continua (salario)
df_empleados['salario'] = (
    25000
    + df_empleados['experiencia']     * 5000
    + df_empleados['certificaciones'] * 2000
    + np.random.normal(0, 3000, n)
)

# Variable objetivo binaria (retención)
# plogis() de R = función sigmoide implementada con NumPy
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

prob = sigmoid(-2 + 0.0001 * df_empleados['salario']
                  + 0.2   * df_empleados['habilidades_sociales'])
df_empleados['retencion'] = (np.random.uniform(size=n) < prob).astype(int)

print("=== Primeras filas del dataset de empleados ===")
print(df_empleados.head().to_string())
print(f"\nDistribución retención:\n{df_empleados['retencion'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# Paso 2: Regresión Lineal – Predicción de Salario
# Equivalente a lm() + summary() de R
# ─────────────────────────────────────────────
train_data, test_data = train_test_split(df_empleados, test_size=0.2, random_state=123)

X_train = train_data[['experiencia', 'certificaciones']].values
y_train = train_data['salario'].values
X_test  = test_data[['experiencia', 'certificaciones']].values
y_test  = test_data['salario'].values

modelo_salario = LinearRegression()
modelo_salario.fit(X_train, y_train)

# Resumen manual equivalente a summary(lm()) de R
y_pred    = modelo_salario.predict(X_test)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test   = r2_score(y_test, y_pred)
r2_train  = modelo_salario.score(X_train, y_train)

# Cálculo manual de errores estándar y estadístico t (usando NumPy)
X_train_c = np.column_stack([np.ones(len(X_train)), X_train])   # añadir constante
y_hat_tr  = modelo_salario.predict(X_train)
residuos  = y_train - y_hat_tr
mse_tr    = np.sum(residuos**2) / (len(y_train) - X_train_c.shape[1])
cov_mat   = mse_tr * np.linalg.inv(X_train_c.T @ X_train_c)
se        = np.sqrt(np.diag(cov_mat))
coefs     = np.concatenate([[modelo_salario.intercept_], modelo_salario.coef_])
t_stats   = coefs / se

print("\n" + "="*60)
print("PASO 2: REGRESIÓN LINEAL – Predicción de Salario")
print("="*60)
print(f"\n  {'Variable':<22} {'Coeficiente':>14} {'Error Estándar':>16} {'t-estadístico':>15}")
print(f"  {'-'*67}")
nombres = ['Intercepto', 'experiencia', 'certificaciones']
for nombre, coef, se_v, t in zip(nombres, coefs, se, t_stats):
    print(f"  {nombre:<22} {coef:>14,.2f} {se_v:>16,.2f} {t:>15.3f}")
print(f"\n  R² (entrenamiento): {r2_train:.4f}")
print(f"  R² (prueba):        {r2_test:.4f}")
print(f"  RMSE (prueba):      ${rmse:,.2f}")
print(f"\n  Interpretación: por cada año extra de experiencia, el salario")
print(f"  aumenta aprox. ${modelo_salario.coef_[0]:,.0f}")

# Gráfico predicción vs real
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5, s=20, c='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Predicción perfecta')
plt.xlabel('Salario Real ($)')
plt.ylabel('Salario Predicho ($)')
plt.title('Regresión Lineal – Salario Real vs Predicho')
plt.legend()
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 3: KNN – Clasificación de Retención
# Equivalente a train(..., method="knn") con caret
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 3: CLASIFICACIÓN KNN – Predicción de Retención")
print("="*60)

features_knn   = ['experiencia', 'habilidades_sociales', 'salario']
X_knn_train    = train_data[features_knn]
y_knn_train    = train_data['retencion']
X_knn_test     = test_data[features_knn]
y_knn_test     = test_data['retencion']

scaler_knn     = StandardScaler()
X_knn_train_sc = scaler_knn.fit_transform(X_knn_train)
X_knn_test_sc  = scaler_knn.transform(X_knn_test)

param_grid = {'n_neighbors': range(1, 21, 2)}
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=cv, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_knn_train_sc, y_knn_train)

cv_results = pd.DataFrame({
    'k':        grid_knn.cv_results_['param_n_neighbors'].data,
    'Accuracy': grid_knn.cv_results_['mean_test_score'],
    'SD':       grid_knn.cv_results_['std_test_score']
})
print("\nAccuracy por número de vecinos (k) – Validación cruzada 5-fold:")
print(cv_results.to_string(index=False))
print(f"\n  k óptimo:             {grid_knn.best_params_['n_neighbors']}")
print(f"  Accuracy CV (k opt):  {grid_knn.best_score_:.4f}")
print(f"  Accuracy test:        {accuracy_score(y_knn_test, grid_knn.predict(X_knn_test_sc)):.4f}")

plt.figure(figsize=(8, 5))
plt.errorbar(cv_results['k'], cv_results['Accuracy'],
             yerr=cv_results['SD'], fmt='o-', color='steelblue',
             ecolor='lightgray', capsize=4, linewidth=2)
plt.axvline(x=grid_knn.best_params_['n_neighbors'], color='red', linestyle='--',
            label=f"k óptimo = {grid_knn.best_params_['n_neighbors']}")
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Accuracy (CV 5-fold)')
plt.title('KNN – Precisión según número de vecinos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Paso 4: Clustering K-Means – Segmentación de Empleados
# Equivalente a kmeans() + ggplot en R
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 4: CLUSTERING K-MEANS – Segmentación por Perfil")
print("="*60)

datos_clustering    = df_empleados[['experiencia', 'salario']].copy()
scaler_km           = StandardScaler()
datos_clustering_sc = scaler_km.fit_transform(datos_clustering)

np.random.seed(456)
kmeans = KMeans(n_clusters=3, random_state=456, n_init=10)
kmeans.fit(datos_clustering_sc)

df_empleados['cluster'] = kmeans.labels_.astype(str)

print("\nEstadísticas por cluster:")
print(df_empleados.groupby('cluster')[['experiencia', 'salario']].mean().round(2).to_string())

colores   = {'0': '#E74C3C', '1': '#2ECC71', '2': '#3498DB'}
etiquetas = {}
for c in ['0', '1', '2']:
    grupo     = df_empleados[df_empleados['cluster'] == c]
    exp_media = grupo['experiencia'].mean()
    sal_media = grupo['salario'].mean()
    if exp_media < 4 and sal_media < 45000:
        etiquetas[c] = 'Junior'
    elif exp_media > 6 or sal_media > 55000:
        etiquetas[c] = 'Senior'
    else:
        etiquetas[c] = 'Mid'

plt.figure(figsize=(9, 6))
for c, grupo in df_empleados.groupby('cluster'):
    plt.scatter(grupo['experiencia'], grupo['salario'],
                c=colores[c], alpha=0.6, s=22,
                label=f'Cluster {c} – {etiquetas[c]}')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario ($)')
plt.title('Segmentación de Empleados por Perfil (K-Means, k=3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nSegmentación final:")
for c, etq in etiquetas.items():
    n_c = (df_empleados['cluster'] == c).sum()
    print(f"  Cluster {c} ({etq}): {n_c} empleados")
