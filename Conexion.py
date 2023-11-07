#Carga de BD
import psycopg2
engine = psycopg2.connect(
dbname="p2",
user="postgres",
password="proyecto2",
host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
port="5432"
)

cursor = engine.cursor()

query = """
SELECT * 
FROM pg_catalog.pg_tables 
WHERE schemaname='public';"""
cursor.execute(query)
result = cursor.fetchall()
result

query = """
SELECT * 
FROM grado;"""
cursor.execute(query)
result = cursor.fetchall()
result