import pandas as pd
from openai import OpenAI
import os

from generate_ideas import search_reviews, read_embedding
from myutils.embeddings_utils import get_embedding, cosine_similarity

key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=key)


def read_file(path):
    with open(path, 'r') as f:
        contents = f.read()
    return contents





prompt = f"""
         Vas a generar un texto en bulleteado o una columna de opinión a partir de una noticia.
         Si se especifica un perfil de un profesional, el texto debe estar escrito en ese perfil.
         El texto debe estar escrito como si lo hubiera escrito alguien con el perfil especificado si se especifica.
         El tipo de texto lo determina el usuario en el prompt. De no ser especificado, se asume que es un texto en columna de opinión.
         Si se te pasa una noticia, usala como fuente de información para generar el texto.
         Para la colmna de opinion:         
            A partir de la siguiente idea sobre una noticia y el contexto proporcionado del profesor y el tema generá una columna de opinión.
            La columna de opinión debe tener una introducción, un desarrollo y una conclusión.
            La columna de opinión debe tener un título.
            La columna de opinión debe tener un tono formal.
         Para el bulleteado:
            A partir de la siguiente idea sobre una noticia y el contexto proporcionado del profesor y el tema generá un texto en bulleteado.
            El bulleteado es una lista de ideas que se pueden desarrollar en una columna de opinión.
            
        A continuación hay noticias y contextos que pueden o no servirte para generar el texto.
        Si no estan relacionados con la idea no los uses.
         """


user_input = read_file(os.getcwd() + "/resources/User_prompt.txt")


response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": read_file(os.getcwd() + "/resources/User_prompt.txt")
        },
    ]
)

print(response.choices[0].message.content)
