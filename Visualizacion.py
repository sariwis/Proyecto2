
############################################

import pandas as pd
import matplotlib.pyplot as plt
# Cargar los datos desde el archivo Excel
df = pd.read_excel('Datos_P2.xlsx')

#####################################

# Porcentaje de estudiantes graduados por género

# Mapear los valores numéricos de género a "Mujer" y "Hombre"
gender_mapping = {0: 'Mujer', 1: 'Hombre'}
df['G'] = df['G'].map(gender_mapping)

# Filtrar los datos para obtener solo los graduados (donde 'target' es igual a 1=graduados)
graduados = df[df['target'] == 1]

# Calcular el porcentaje de graduados por género
gender_percentage = (graduados['G'].value_counts() / len(graduados)) * 100

# Crear un gráfico de barras para comparar los porcentajes de graduados por género
ax = gender_percentage.plot(kind='bar', color=['blue', 'pink'], ylim=(0, 100))
plt.title('Porcentaje de estudiantes graduados por género')

# Eliminar la etiqueta del eje x
plt.xlabel('')

plt.xticks(rotation=0)  # Rotar las etiquetas del eje x

# Agregar etiquetas con los valores porcentuales en la parte superior de cada barra
for index, value in enumerate(gender_percentage):
    plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

# Eliminar el eje y y el marco cuadrado
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.show()

####################################################################

#Porcentaje de Graduados y No Graduados por edades


# Filtrar los datos para obtener solo los graduados y los que no se graduaron
graduados = df[df['target'] == 1]
no_graduados = df[df['target'] == 0]

# Calcular el porcentaje de graduados y no graduados por categoría de edad
age_categories = df['AE'].value_counts().sort_index()
total_students = age_categories.sum()
graduates = graduados['AE'].value_counts().sort_index()
non_graduates = no_graduados['AE'].value_counts().sort_index()
graduates_percentage = (graduates / total_students) * 100
non_graduates_percentage = (non_graduates / total_students) * 100

# Crear un gráfico de barras apiladas que compare los porcentajes de graduados y no graduados por categoría de edad
plt.figure(figsize=(8, 6))

# Etiquetas de las categorías con rangos de edad
labels = ['Joven\n(16-30)', 'Adulto\n(31-45)', 'Mayor\n(46-63)']

# Crear el gráfico de barras apiladas
bars = plt.bar(labels, graduates_percentage, color='lightblue', label='Graduados')
plt.bar(labels, non_graduates_percentage, color='lightcoral', label='No Graduados', bottom=graduates_percentage)

plt.title('Porcentaje de Graduados y No Graduados por edades')

# Eliminar los porcentajes de las barras
for bar in bars:
    bar.set_label("")

# Eliminar la etiqueta del eje y (pero dejar los valores del eje)
plt.gca().get_yaxis().set_visible(True)

# Eliminar el eje x y su etiqueta
plt.xticks([])

# Agregar etiquetas en la parte inferior de cada barra
for index, label in enumerate(labels):
    plt.text(index, -8, label, ha='center')

# Formatear el eje y con el signo de porcentaje
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Agregar la leyenda
plt.legend()

plt.show()


################################################################

from matplotlib.ticker import FuncFormatter

#Porcentaje de Graduados y No Graduados por curso

# Filtrar los datos para obtener solo los graduados y los que no se graduaron
graduados = df[df['target'] == 1]
no_graduados = df[df['target'] == 0]

# Calcular el porcentaje de graduados y no graduados por categoría de curso
course_categories = df['C'].value_counts().sort_index()
total_students = course_categories.sum()
graduates = graduados['C'].value_counts().sort_index()
non_graduates = no_graduados['C'].value_counts().sort_index()
graduates_percentage = (graduates / total_students) * 100
non_graduates_percentage = (non_graduates / total_students) * 100

# Crear un gráfico de barras apiladas que compare los porcentajes de graduados y no graduados por categoría de curso
plt.figure(figsize=(10, 6))

# Etiquetas de las categorías de curso (puedes ajustarlas según tus necesidades)
course_labels = [
    "Biofuel Production Technologies", "Animation and Multimedia Design",
    "Social Service (evening attendance)", "Agronomy",
    "Communication Design", "Veterinary Nursing",
    "Informatics Engineering", "Equinculture",
    "Management", "Social Service",
    "Tourism", "Nursing",
    "Oral Hygiene", "Advertising and Marketing Management",
    "Journalism and Communication", "Basic Education",
    "Management (evening attendance)"
]

# Crear el gráfico de barras apiladas
bars = plt.bar(course_labels, graduates_percentage, color='lightblue', label='Graduados')
plt.bar(course_labels, non_graduates_percentage, color='lightcoral', label='No Graduados', bottom=graduates_percentage)

plt.title('Porcentaje de Graduados y No Graduados por curso')

# Eliminar las etiquetas del eje x (puedes ajustar el formato de las etiquetas)
plt.xticks(rotation=90)

# Formatear el eje y con el signo de porcentaje
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Agregar la leyenda
plt.legend()

plt.show()


