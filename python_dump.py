import json
from ast import literal_eval

import numpy as np
import pandas as pd
from openai import OpenAI
import os

from myutils.embeddings_utils import get_embedding, cosine_similarity


def read_file(path):
    with open(path, 'r') as f:
        contents = f.read()
    return contents


def parse_json(string):
    return json.loads(string)


def read_file(path):
    with open(path, 'r') as f:
        contents = f.read()
    return contents


def read_embedding(file_path):
    with open(file_path, 'r') as f:
        data_str = f.read()
        data_list = literal_eval(data_str)
        return np.array(data_list)


def load_embeddings():
    directory_path = os.getcwd() + "/resources/embedded_profiles"
    data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            embedding = read_embedding(file_path)
            data.append((filename, embedding))

    return pd.DataFrame(data, columns=["filename", "embedding"])


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


def generate_article(user_input=read_file(os.getcwd() + "/resources/User_prompt_generate_article.txt")):
    key = os.environ.get('OPENAI_API_KEY')

    client = OpenAI(api_key=key)

    prompt = """
             Vas a generar un texto en bulleteado o una columna de opinión a partir de una noticia.
             Si se especifica un perfil de un profesional, el texto debe estar escrito en ese perfil.
             El texto debe estar escrito como si lo hubiera escrito alguien con el perfil especificado si se especifica.
             El tipo de texto lo determina el usuario en el prompt. De no ser especificado, se asume que es un texto en columna de opinión.
             Si se te pasa una noticia, usala como fuente de información para generar el texto.
             Para la colmna de opinion:       
                Debe incluir un titulo breve.  
                Escribí el texto en prosa.
                A partir de la siguiente idea sobre una noticia y el contexto proporcionado del profesor y el tema generá una columna de opinión.
                La columna de opinión debe tener una introducción, un desarrollo y una conclusión.
                No escribas introduccion: desarrollo: conclusión sino que escribí el texto en prosa.
                La columna de opinión debe tener un título.
                La columna de opinión debe tener un tono formal.
             Para el bulleteado:
                A partir de la siguiente idea sobre una noticia y el contexto proporcionado del profesor y el tema generá un texto en bulleteado.
                El bulleteado es una lista de ideas que se pueden desarrollar en una columna de opinión.

            Si el usuario envia noticias, debes usarlas para contexto de la columna de opinión.
            Si no estan relacionados con la idea no los uses.
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
                "content": user_input
            },
        ]
    )

    print(response.choices[0].message.content)
