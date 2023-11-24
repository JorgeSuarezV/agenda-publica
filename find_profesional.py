import os

from openai import OpenAI

from python_dump import search_reviews, load_embeddings

from functools import reduce


def read_file(path):
    with open(path, 'r') as f:
        contents = f.read()
    return contents


key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key=key)

user_input = read_file(os.getcwd() + "/resources/User_prompt_find_profesional.txt")

professionals = search_reviews(load_embeddings(), user_input, 5).tolist()

print(user_input)

for r in professionals:
    content = read_file(os.getcwd() + "/resources/raw_profiles/" + r)
    prompt = """
        - A partir de la idea una columna de opinion y el perfil del profesional debes primero 
        asignar un puntaje del 1 al 10 para hablar del tema que te van a proporcionar.
        - tambien debes justificar la calificación que le diste al profesional para hablar del tema.
        - se breve y consiso.
        - Escribí Nombre:  y el nombre del profesional.
        - Escribí Calificación: y el puntaje que le diste al profesional.
        - Escribí Justificación: y la justificación que le diste al profesional.
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
                "content": f"""
                tema: {user_input}
                perfil: {content}
                """
            },
        ]
    )
    print(response.choices[0].message.content)

