from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import os

load_dotenv()

# الحصول على API Key من المتغيرات البيئية
API_KEY = os.getenv('API_KEY')

embiddings = OpenAIEmbeddings(api_key=API_KEY)

vector_db = Chroma(embedding_function=embiddings,
    collection_name="tutorial",
    persist_directory="./chroma_db"
)


def add_document_to_chroma (file_path):
    loader = TextLoader(file_path)
    doc = loader.load()

    text_splitter = CharacterTextSplitter (chunk_size=1000, chunk_overlap=20)
    text = text_splitter.split_documents(doc)

    vector_db.add_documents(text)

    print("Added text chunks from the file to the Chroma Vector Database.")

def main():
    while True:
        file_path = input("Enter the path to the text file you want to add to the Chroma Vector Database: ")
        add_document_to_chroma(file_path)

        
        if  file_path.lower() != 'q':
            break
        if file_path.exists():
            add_document_to_chroma(file_path)
        else:
            print("The file does not exist. Please try again.")
            

if __name__ == '__main__':
    main()   