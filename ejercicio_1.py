# ==========================================
# PRACTICA 2 - CIENCIA DE DATOS
# ANALISIS DE CALIDAD INDUSTRIAL
# ==========================================

# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# ==========================================
# GENERAR DATOS
# ==========================================

np.random.seed(42)

n = 500

temperatura = np.random.normal(75, 5, n)
presion = np.random.normal(30, 3, n)

# Relación entre temperatura y errores
tasa_error = (temperatura * 0.3) + np.random.normal(0, 2, n)

turno = np.random.choice(['Matutino', 'Vespertino'], n)

# Crear DataFrame
df = pd.DataFrame({
    'temperatura': temperatura,
    'presion': presion,
    'tasa_error': tasa_error,
    'turno': turno
})

# Agregar valores faltantes
df.loc[10, 'temperatura'] = np.nan
df.loc[20, 'temperatura'] = np.nan

# ==========================================
# FASE A - ANALISIS EXPLORATORIO
# ==========================================

print("\n===== FASE A =====")

# Medidas estadísticas
print("Media:", df['temperatura'].mean())
print("Mediana:", df['temperatura'].median())
print("Desviación estándar:", df['temperatura'].std())

# Valores faltantes
print("\nValores faltantes:")
print(df.isnull().sum())

# Limpieza de datos
df['temperatura'] = df['temperatura'].fillna(
    df['temperatura'].mean()
)

# Correlación Pearson
correlacion = df['temperatura'].corr(df['tasa_error'])

print("\nCorrelación Pearson:", correlacion)

# ==========================================
# FASE B - AGRUPACION
# ==========================================

print("\n===== FASE B =====")

# Promedio de errores por turno
print("\nPromedio de errores:")
print(df.groupby('turno')['tasa_error'].mean())

# Temperatura promedio por turno
print("\nTemperatura promedio:")
print(df.groupby('turno')['temperatura'].mean())

# ==========================================
# MODELO DE REGRESION LINEAL
# ==========================================

X = df[['temperatura']]
y = df['tasa_error']

modelo = LinearRegression()

modelo.fit(X, y)

print("\nPendiente:", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)

# ==========================================
# FASE C - VISUALIZACIONES
# ==========================================

sns.set(style="whitegrid")

# ------------------------------------------
# BOXPLOT
# ------------------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='turno',
    y='tasa_error',
    data=df
)

plt.title('Dispersión de Errores por Turno')

plt.show()

# ------------------------------------------
# SCATTER PLOT + REGRESION
# ------------------------------------------

plt.figure(figsize=(8,5))

sns.scatterplot(
    x='temperatura',
    y='tasa_error',
    data=df
)

# Línea de regresión
plt.plot(
    df['temperatura'],
    modelo.predict(X)
)

plt.title('Temperatura vs Tasa de Error')

plt.xlabel('Temperatura')
plt.ylabel('Tasa de Error')

plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("""
CONCLUSION

Los resultados muestran que existe una relación positiva
entre la temperatura y la tasa de errores.

La correlación de Pearson y el modelo de regresión lineal
indican que al aumentar la temperatura también incrementan
los errores de producción.

Los gráficos permitieron detectar dispersión y posibles
valores atípicos.

Se recomienda que la empresa invierta en mejores sistemas
de enfriamiento y monitoreo industrial.
""")