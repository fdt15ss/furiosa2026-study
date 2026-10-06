import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.
api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    # model='text-embedding-3-large', # 3072
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = './_db/Chroma11/'

db = Chroma(
    embedding_function=embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma11"
)
print("=============================")
print(db.get())

print("=============================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2)
print(aaa)