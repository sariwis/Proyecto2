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
import pandas.io.sql as sqlio

psw='proyecto2'

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

########################
import psycopg2
import pandas as pd
import sqlite3
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import psycopg2

def porcentaje_graduados_por_curso():
    #CONECTARSE
    
    engine = psycopg2.connect(
        dbname="p2",
        user="postgres",
        password=psw,
        host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
        port="5432"
    )
    cursor = engine.cursor()

    query = """
    SELECT C AS curso,
        SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS graduados
    FROM grado
    WHERE C BETWEEN 1 AND 17  -- Filtrar por cursos del 1 al 17
    GROUP BY C
    ORDER BY C;
    """

    # Mapeo de números de curso a nombres
    curso_nombre_map = {
        1: "Biofuel Production Technologies",
        2: "Animation and Multimedia Design",
        3: "Social Service (evening attendance)",
        4: "Agronomy",
        5: "Communication Design",
        6: "Veterinary Nursing",
        7: "Informatics Engineering",
        8: "Equinculture",
        9: "Management",
        10: "Social Service",
        11: "Tourism",
        12: "Nursing",
        13: "Oral Hygiene",
        14: "Advertising and Marketing Management",
        15: "Journalism and Communication",
        16: "Basic Education",
        17: "Management (evening attendance)"
    }

    # Ejecutar la consulta SQL y cargar los resultados en un DataFrame
    df = pd.read_sql_query(query, engine)

    # Mapear los números de curso a nombres
    df['curso'] = df['curso'].map(curso_nombre_map)


    # Calcular el porcentaje de graduados y no graduados
    df['Graduados'] = (df['graduados'] / df['graduados'].sum()) * 100
    df['Desertores'] = 100 - df['Graduados']

    # Crear la figura de la gráfica de barras horizontales
    fig = px.bar(
        df, y='curso', x=['Graduados', 'Desertores'],
        labels={'value': 'Porcentaje', 'curso': 'Curso'}, title='Porcentaje de Graduados y Desertores por Curso',
        color_discrete_map={'Graduados': 'green', 'Desertores': 'lightgreen'},
        orientation='h'
    )

    fig.update_layout(
        width=950,  # Ancho de la gráfica
        height=500,  # Altura de la gráfica
        margin=dict(l=50, r=50, b=50, t=50)  # Márgenes
    )
    
    return dcc.Graph(figure=fig)

def porcentaje_genero():
    # Establecer la conexión a la base de datos
    engine = psycopg2.connect(
        dbname="p2",
        user="postgres",
        password=psw,
        host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
        port="5432"
    )

    # Definir la consulta SQL
    query = """
    SELECT g AS genero, 
           SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS graduados
    FROM grado
    WHERE g IN (0, 1) -- Filtrar por géneros 0 (Mujer) y 1 (Hombre)
    GROUP BY g;
    """

    # Ejecutar la consulta y obtener el resultado como un DataFrame
    df = sqlio.read_sql_query(query, engine)

    # Reemplazar 0 con "Mujeres" y 1 con "Hombres" en la columna "genero"
    df['genero'] = df['genero'].map({0: 'Mujeres', 1: 'Hombres'})

    # Calcular el porcentaje de graduados y desertores
    df['Graduados'] = (df['graduados'] / df['graduados'].sum()) * 100
    df['Desertores'] = 100 - df['Graduados']

    # Crear la figura de la gráfica de barras
    fig = px.bar(
        df, x='genero', y=['Graduados', 'Desertores'],
        labels={'value': 'Porcentaje', 'genero': 'Género'},
        title='Porcentaje de Graduados y Desertores por Género',
        color_discrete_map={'Graduados': 'green', 'Desertores': 'lightgreen'},
    )

    fig.update_layout(
        width=950,  # Ancho de la gráfica
        height=500,  # Altura de la gráfica
        margin=dict(l=50, r=50, b=50, t=50)  # Márgenes
    )
    
    return dcc.Graph(figure=fig)

def porc_edades():
    #CONECTARSE
    import psycopg2
    import pandas as pd
    import sqlite3
    engine = psycopg2.connect(
        dbname="p2",
        user="postgres",
        password=psw,
        host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
        port="5432"
    )
    cursor = engine.cursor()
    query = """
    SELECT AE AS age,
       SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS graduados,
       SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) AS no_graduados
    FROM grado
    GROUP BY AE;
    """
    df = pd.read_sql_query(query, engine)

    # Calcular los porcentajes
    df['Graduados'] = (df['graduados'] / (df['graduados'] + df['no_graduados'])) * 100
    df['Desertores'] = (df['no_graduados'] / (df['graduados'] + df['no_graduados'])) * 100

    # Mapear los valores de age a etiquetas más descriptivas
    df['age'] = df['age'].map({1: 'Jóvenes', 2: 'Adultos', 3: 'Mayores'})

    # Crear una nueva columna con el rango de edad
    df['Rango de Edad'] = df['age'].map({'Jóvenes': '16-30 años', 'Adultos': '31-45 años', 'Mayores': '46-63 años'})

    # Crear la figura de la gráfica de barras con colores verdes y leyenda personalizada
    fig = px.bar(
        df, x='Rango de Edad', y=['Graduados', 'Desertores'],
        labels={'value': 'Porcentaje'}, title='Porcentaje de Graduados y Desertores por Rango de Edad',
        color_discrete_map={'Graduados': 'green', 'Desertores': 'lightgreen'},
    )

    fig.update_layout(
        width=950,  # Ancho de la gráfica
        height=500,  # Altura de la gráfica
        margin=dict(l=50, r=50, b=50, t=50)  # Márgenes
    )
    
    return dcc.Graph(figure=fig)




#################

tab_1_content = html.Div([

html.Div([
    html.H1('Graduación y Deserción Estudiantil: Un Análisis Visual', style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#6C9B37', 'padding': '20px'}),
    html.P('A continuación encontrarás tres visualizaciones que te darán a conocer la situación actual de graduación y deserción en tu universidad', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ], style={'backgroundColor': '#f2f2f2'}),

    html.Br(),

    html.Div([
    porcentaje_genero()
    ]),

    html.Br(),

    html.Div([
    porcentaje_graduados_por_curso()
    ]),

    html.Br(),

    html.Div([
    porc_edades()
    ]),
    
    html.Br(),
    html.Br(),

    html.Div([
    html.P('¡En la pestaña "Usa nuestra Herramienta" te invitamos a calcular tu probabilidad de graduación!', style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    ]),


    html.Br(),
    html.Br(),

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
            dcc.Dropdown(id='MS', options=[{'label': 'Soltero/a', 'value': 1}, {'label': 'Casado/a', 'value': 2}, {'label': 'Divorciado/a', 'value': 4}, {'label': 'Unión Libre', 'value': 5}], placeholder='Selecciona estado civil', value=1),
        ], className='four columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('Género'),
            dcc.Dropdown(id='Gen', options=[{'label': 'Femenino', 'value': 0}, {'label': 'Masculino', 'value': 1}], placeholder='Selecciona género', value=1),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('Edad (17-62 años)'),
            dcc.Input(id='Edad', type='number', placeholder='Ingresa tu edad', min=17, max=62,value=17),
        ], className='four columns', style={'marginTop': '10px'}),
    ], className='row'),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('¿Cuál fue su nota previa a la universidad? (50-200)'),
            dcc.Input(id='Not_prev', type='number', placeholder='Ingresa tu nota', min=50, max=200, value=80),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
            html.Label('¿Cuál fue su calificación de admisión? (50-200)'),
            dcc.Input(id='Not_adm', type='number', placeholder='Ingresa tu nota', min=50, max=200, value=120),
        ], className='four columns', style={'marginTop': '10px'}),

        html.Div([
                html.Label('¿Usted o su familia asumirá una dedua para costear la universidad?'),
                dcc.Dropdown(id='Deuda', options=[{'label':'Sí', 'value':1},{'label':'No', 'value':0}], placeholder='Deudor', value=0),
            ], className='four columns', style={'marginTop': '10px'}),

    ], className='row'),

    html.Br(),


    html.Div([
        html.Div([
            html.Label('Curso'),
            dcc.Dropdown(id='Curso', options=[{'label': 'Tecnologías de Producción de Biocombustibles', 'value': 1}, {'label': 'Diseño de Animación y Multimedia', 'value': 2 },{'label': 'Servicio Social (turno vespertino)', 'value': 3},{'label': 'Agronomía', 'value': 4},{'label': 'Diseño de Comunicación', 'value': 5},{'label': 'Enfermería Veterinaria', 'value': 6},{'label': 'Ingeniería Informática', 'value': 7},{'label': 'Equinocultura', 'value': 8},{'label': 'Gestión', 'value': 9},{'label': 'Servicio Social', 'value':10},{'label': 'Turismo', 'value':11},{'label': 'Enfermería', 'value':12},{'label': 'Higiene Oral', 'value':13},{'label': 'Dirección de Publicidad y Marketing', 'value':14},{'label': 'Periodismo y Comunicación', 'value':15},{'label': 'Educación Básica', 'value':16},{'label': 'Gestión (turno en la tarde)', 'value':17}], placeholder='Seleccione su curso', value=2),
        ], className='six columns', style={'marginTop': '10px'}),
        
        html.Div([
            html.Label('¿El curso seleccionado fue su primera opción? (1-6)'),
            dcc.Input(id='Apl_order', type='number', placeholder='Ingresa la orden de aplicación', min=1, max=6, value=1),
            html.P('Nota: 1 significa que quedó en su opción preferida y 6 significa la menos preferida.', style={'fontSize': '14px', 'color': 'gray'}),
        ], className='six columns', style={'marginTop': '10px'})
    ], className='row'),

    html.Br(),

    ##

    html.Br(),

    html.Button('CALCULA TU RESULTADO', id='submit', n_clicks=0, style={'backgroundColor': '#6C9B37', 'color': 'white', 'fontSize': '18px'}),
    
    html.Br(),
    html.Br(),

    html.Div(id='output'),

    html.Br(),

    #html.Div[(id='probability-plot')]
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

@app.callback(Output('tab-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab_1_content
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
        if n_clicks >= 0:
        
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

            #fig = px.bar(prob_df, x='Categoría', y='Probabilidad', text='Probabilidad', height=600,
             #        labels={'Categoría': 'Resultado', 'Probabilidad': 'Probabilidad'},
              #       color='Categoría', title='Probabilidad de Graduación vs. Probabilidad de Retiro')

            colors = {'Graduación': 'green', 'Retiro': 'lightgreen'}

            fig = px.bar(prob_df, x='Categoría', y='Probabilidad', text='Probabilidad', height=600,
                        labels={'Categoría': 'Resultado', 'Probabilidad': 'Probabilidad'},
                        color_discrete_map=colors,  # Establecer los colores
                        title='Probabilidad de Graduación vs. Probabilidad de Retiro')
            
            # Personaliza el diseño del gráfico
            fig.update_traces(marker_color=['green', 'lightgreen'])
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
        

