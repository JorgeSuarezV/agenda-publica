import json

from openai import OpenAI
import os
import numpy as np
import pandas as pd
from ast import literal_eval
from myutils.embeddings_utils import get_embedding, cosine_similarity

# start timer
import time

from python_dump import read_embedding, parse_json, search_reviews, read_file, load_embeddings

start = time.time()

key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=key)

prompt = """
             Agarrá las notiticias que te van a venir a continuación y generá a partir de las siguientes categorías ideas para columnas de opinión.
             Categorías:
                Internacional
                Nacional
                Economía
                Ciencia y Tecnología
                Salud
                Medio Ambiente
                Crimen y Justicia
                Educación
                Social y Humanitario
             Generá 5 ideas por categoría.
             Las ideas no tienen que estar directamente relacionadas con la noticia, pueden ser ideas que se te ocurran a partir de la noticia o un analisis de una nueva perpectiva.
             Las ideas deben incluir una lista de temas a los que estén relacionadas. 
             Las ideas no deben incluir temas politicos a menos que sea una noticia politica.
             Valorá más ideas que no incluyan temas politicos.
             2 de las 5 ideas deben ser un poco controversiales.
             Preferí ideas que puedan ser desarrolladas en una columna de opinión en un aspecto más profesional.
             Las ideas tienen que estar en español.
             Escribí las ideas en formato de json, por ejemplo:
             [
                {
                    "categoria": "Internacional",
                    "ideas": [
                        {
                            "idea": "idea1",
                            "temas": [tema1, tema2, tema3],
                        }
                        {
                            "idea": "idea2",
                            "temas": [tema1, tema2, tema3],
                        }
                        ...
                    ]
                }
                ...
             ]
             Empezá el mensaje con un "[" y terminá el mensaje con un "]"
             """

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": read_file(os.getcwd() + "/resources/News.txt")
        },
    ]
)

df = load_embeddings()

# parse from json


response = parse_json(response.choices[0].message.content)

for r in response:
    category = r["categoria"]
    ideas = r["ideas"]
    for index, idea in enumerate(ideas):
        r["ideas"][index] = {
            "idea": idea["idea"],
            "profesional_relacionado": search_reviews(df, idea["idea"] + ". temas:" + idea["temas"].__str__(),
                                                      n=5).tolist()
        }

print(response)
print("Time elapsed: ", time.time() - start)
