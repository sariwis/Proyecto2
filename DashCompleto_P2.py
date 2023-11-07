import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import re
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Define los estilos CSS personalizados para mejorar la apariencia del dashboard
app.layout = html.Div([

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Panorama', value='tab-1'),
        dcc.Tab(label='Usa nuestra Herramienta', value='tab-2'),
    ]),
    html.Div(id='tab-content'),

], className='container')

tab_1_content = html.Div([

html.Div([
    html.H1('Graduación y Deserción Estudiantil: Un Análisis Visual', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
    ], style={'backgroundColor': '#f2f2f2'}),

    html.Br(),

    dcc.Graph(id='vis_1'),

    html.Br(),

    dcc.Graph(id='vis_2'),

    html.Br(),

    dcc.Graph(id='vis_3'),

], style={'backgroundColor': '#f2f2f2'})

tab_2_content = html.Div([
    html.Div([
        html.H1('Herramienta de Predicción de Graduación Universitaria', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
        html.P('¡Felicitaciones por completar la educación secundaria! Esta herramienta utiliza tus datos personales y académicos para predecir tu probabilidad de graduación o retiro universitario. Responde las siguientes preguntas:', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),
    
    html.Br(),

    html.Div([
        html.Div([
            html.Label('Estado Civil'),
            dcc.Dropdown(id='MS', options=[{'label': 'Soltero/a', 'value': 1}, {'label': 'Casado/a', 'value': 2}, {'label': 'Divorciado/a', 'value': 4}, {'label': 'Unión Libre', 'value': 5}], placeholder='Selecciona estado civil'),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('Género'),
            dcc.Dropdown(id='Gen', options=[{'label': 'Femenino', 'value': 0}, {'label': 'Masculino', 'value': 1}], placeholder='Selecciona género'),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('Edad (17-62 años)'),
            dcc.Input(id='Edad', type='number', placeholder='Ingresa tu edad', min=17, max=62),
        ], className='four columns', style={'marginTop': '10px'}),
    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('¿Cuál fue su nota previa a la universidad? (50-200)'),
            dcc.Input(id='Not_prev', type='number', placeholder='Ingresa tu nota', min=50, max=200),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('¿Cuál fue su calificación de admisión? (50-200)'),
            dcc.Input(id='Not_adm', type='number', placeholder='Ingresa tu nota', min=50, max=200),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
                html.Label('¿Usted o su familia asumirá una dedua para costear la universidad?'),
                dcc.Dropdown(id='Deuda', options=[{'label':'Sí', 'value':1},{'label':'No', 'value':0}], placeholder='Deudor'),
            ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),


    html.Div([
        html.Div([
            html.Label('Curso'),
            dcc.Dropdown(id='Curso', options=[{'label': 'Tecnologías de Producción de Biocombustibles', 'value': 1}, {'label': 'Diseño de Animación y Multimedia', 'value': 2 },{'label': 'Servicio Social (turno vespertino)', 'value': 3},{'label': 'Agronomía', 'value': 4},{'label': 'Diseño de Comunicación', 'value': 5},{'label': 'Enfermería Veterinaria', 'value': 6},{'label': 'Ingeniería Informática', 'value': 7},{'label': 'Equinocultura', 'value': 8},{'label': 'Gestión', 'value': 9},{'label': 'Servicio Social', 'value':10},{'label': 'Turismo', 'value':11},{'label': 'Enfermería', 'value':12},{'label': 'Higiene Oral', 'value':13},{'label': 'Dirección de Publicidad y Marketing', 'value':14},{'label': 'Periodismo y Comunicación', 'value':15},{'label': 'Educación Básica', 'value':16},{'label': 'Gestión (turno en la tarde)', 'value':17}], placeholder='Seleccione su curso'),
        ], className='six columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿El curso seleccionado fue su primera opción? (0-6)'),
            dcc.Input(id='Apl_order', type='number', placeholder='Ingresa la orden de aplicación', min=0, max=6),
            html.P('Nota: 0 significa que quedó en su opción preferida y 6 significa la menos preferida.', style={'fontSize': '14px', 'color': 'gray'}),
        ], className='six columns', style={'marginTop': '10px'})
    ], className='row'),

    html.Br(),

    ##

    html.Br(),

    html.Button('CALCULA TU RESULTADO', id='submit', n_clicks=0, style={'backgroundColor': '#6C9B37', 'color': 'white', 'fontSize': '18px'}),
    html.Br(),

    html.Div(id='output'),

    html.Br(),

    dcc.Graph(id='probability-plot'),  # Visualización de probabilidad de graduación o retiro
], style={'backgroundColor': '#f2f2f2'})



def by_pred(MS, AO, C, PQ, AG, D, G, AE):
    datos = 'Datos_P2.xlsx'
    df = pd.read_excel(datos)

    modelo = BayesianNetwork([('MS', 'AE'), ('MS', 'target'), ('C', 'target'), ('C', 'AO'), ('C', 'G'), ('PQ', 'AG'), ('AG', 'C'), ('G', 'target'), ('AE', 'C'), ('AE', 'target'), ('AE', 'G'), ('AE', 'AG'), ('target', 'D')])

    sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
    emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

    modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(modelo)

    resp = infer.query(['target'], evidence={'MS': MS, 'AO': AO, 'C': C, 'PQ': PQ, 'AG': AG, 'D': D, 'G': G, 'AE': AE})

    return resp


import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

# Conectarse a la base de datos PostgreSQL
engine = psycopg2.connect(
    dbname="p2",
    user="postgres",
    password="proyecto2",
    host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
    port="5432"
)

# Consulta SQL para obtener los datos de graduados
query = """
SELECT G, target
FROM grado;
"""

# Ejecutar la consulta y cargar los resultados en un DataFrame
df = pd.read_sql_query(query, engine)

# Cerrar la conexión a la base de datos
engine.close()

# Mapear los valores numéricos de género a "Mujer" y "Hombre"
gender_mapping = {0: 'Mujer', 1: 'Hombre'}
df['g'] = df['g'].map(gender_mapping)

# Filtrar los datos para obtener solo los graduados (donde 'target' es igual a 1)
graduados = df[df['target'] == 1]

# Calcular el porcentaje de graduados por género
gender_percentage = (graduados['g'].value_counts() / len(graduados)) * 100

# Define tus colores personalizados en formato hexadecimal
custom_colors = ['#1B4D2C', '#66A550']

# Crear un gráfico de barras para comparar los porcentajes de graduados por género
ax = gender_percentage.plot(kind='bar', color=custom_colors, ylim=(0, 100), width=0.3)

# Personalizar el título
plt.title('Porcentaje de estudiantes graduados por género', fontsize=15, fontweight='bold')

# Eliminar la etiqueta del eje x
plt.xlabel('')

# Aumentar el tamaño de fuente y poner en negrita los valores de porcentaje
for index, value in enumerate(gender_percentage):
    plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Ajustar los márgenes
plt.margins(x=0.2)

# Eliminar el eje y y el marco cuadrado
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Ajustar la rotación de las etiquetas en el eje x
plt.xticks(rotation=0)

# Guardar la gráfica como una imagen
plt.savefig('gender_percentage.png')


@app.callback(Output('tab-content', 'children'), [Input('tabs', 'value')])
# Define el contenido de la pestaña 1
def render_tab_1_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Contenido de la Pestaña 1'),

            # Gráfica 1: Porcentaje de graduados por género
            html.Img(src='C:/Users/ceden/OneDrive/Escritorio/Analítica Computacional/Proyecto 1/Proyecto2/gender_percentage.png', width="80%"),  # Reemplaza 'gender_percentage.png' con la ruta de tu imagen
            html.P('Esta gráfica muestra el porcentaje de estudiantes graduados por género.'),
            
            # Puedes agregar más gráficas o contenido aquí si es necesario
        ])
    else:
        return html.Div([])

def render_content(tab):
    if tab == 'tab-1':
        return render_tab_1_content(tab)
    elif tab == 'tab-2':
        return tab_2_content



@app.callback(
    Output('output', 'children'),
    Output('probability-plot', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('MS', 'value'),
    State('Gen', 'value'),
    State('Edad', 'value'),
    State('Not_prev', 'value'),
    State('Not_adm', 'value'),
    State('Deuda', 'value'),
    State('Curso', 'value'),
    State('Apl_order', 'value')]
)
def update_output(n_clicks, MS, G, AE, PQ, AG, D, C, AO):
    try:
        if n_clicks > 0:
        
            lim_Edad = [16, 30, 45, 63]
            labels_Edad = [1, 2, 3]
            AE = pd.cut([AE], bins=lim_Edad, labels=labels_Edad)[0]

            lim_Notas = [-1, 49, 99, 149, 201]
            labels_Notas = [1, 2, 3, 4]
            PQ = pd.cut([PQ], bins=lim_Notas, labels=labels_Notas)[0]
            AG = pd.cut([AG], bins=lim_Notas, labels=labels_Notas)[0]

            probabilidad = by_pred(MS, AO, C, PQ, AG, D, G, AE)
            
            # Crea un gráfico de barras para visualizar la probabilidad
            df = pd.DataFrame({'Probabilidad': [probabilidad], 'Resultado': ['Probabilidad de Graduación']})

            prob_df = pd.DataFrame({'Categoría': ['Graduación', 'Retiro'], 'Probabilidad': [probabilidad.values[1], probabilidad.values[0]]})

            fig = px.bar(prob_df, x='Categoría', y='Probabilidad', text='Probabilidad', height=400,
                     labels={'Categoría': 'Resultado', 'Probabilidad': 'Probabilidad'},
                     color='Categoría', title='Probabilidad de Graduación vs. Probabilidad de Retiro')

            # Personaliza el diseño del gráfico
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(legend_title_text='Resultado')
            #fig = {'data': [{'x': ['Probabilidad de Graduación'], 'y': [probabilidad.values[1]], 'type': 'bar'}],'layout': {'xaxis': {'title': 'Resultado'}, 'yaxis': {'title': 'Probabilidad'}}}
            #fig = px.bar(df, x='Resultado', y='Probabilidad', text='Probabilidad', height=400)
            resp = probabilidad.values[1]*100
            return f'Probabilidad de Graduación: {round(resp, 2)}%', fig
    except Exception as e:
        return f'Error: {str(e)}', {}



#Se ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True,port =8070)
        
