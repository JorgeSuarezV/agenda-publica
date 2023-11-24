import os

from openai import OpenAI


def iterate_files(directory, saveDirectory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            print(f"Processing file: {filename}")
            with open(os.path.join(directory, filename), 'r') as file:
                file_content = file.read()
                with open(os.path.join(saveDirectory, filename), 'w') as saveFile:
                    saveFile.write(create_embeddings(file_content).data[0].embedding.__str__())



client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def create_embeddings(text):
    return client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )


iterate_files(os.getcwd() + "/resources/raw_profiles", os.getcwd() + "/resources/embedded_profiles")
