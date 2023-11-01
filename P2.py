#Carga de BD
import psycopg2
engine = psycopg2.connect(
dbname="postgres",
user="postgres",
password="proyecto2",
host="proyecto2.c9pexl84mjtw.us-east-1.rds.amazonaws.com",
port="5432"
)

