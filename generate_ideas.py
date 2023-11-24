import json

from openai import OpenAI
import os
import numpy as np
import pandas as pd
from ast import literal_eval
from myutils.embeddings_utils import get_embedding, cosine_similarity

#start timer
import time

start = time.time()

key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=key)


def read_file(path):
    with open(path, 'r') as f:
        contents = f.read()
    return contents


def read_embedding(file_path):
    with open(file_path, 'r') as f:
        data_str = f.read()
        data_list = literal_eval(data_str)
        return np.array(data_list)


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
            "content": read_file(os.getcwd() + "/resources/Noticias.txt")
        },
    ]
)

directory_path = os.getcwd() + "/resources/embedded_profiles"
data = []

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        embedding = read_embedding(file_path)
        data.append((filename, embedding))

df = pd.DataFrame(data, columns=["filename", "embedding"])


# parse from json
def parse_json(string):
    return json.loads(string)


response = parse_json(response.choices[0].message.content)


def search_reviews(df, product_description, n=3, pprint=False):
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .filename
    )
    if pprint:
        for r in results:
            print(r)
            print()
    return results


for r in response:
    category = r["categoria"]
    ideas = r["ideas"]
    for index, idea in enumerate(ideas):
        r["ideas"][index] = {
            "idea": idea["idea"],
            "profesional_relacionado": search_reviews(df, idea["idea"] + ". temas:" + idea["temas"].__str__(), n=5).tolist()
        }

print(response)
print("Time elapsed: ", time.time() - start)
