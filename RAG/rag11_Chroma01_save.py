import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.
api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

#01 데이터 불러온다.
path = './_data/rag_data/'
loader1 = TextLoader(path + "samsung_outlook.txt", encoding='utf-8')
loader2 = TextLoader(path + "nvidia_outlook.txt", encoding='utf-8')

# 문서를 자른다 / 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators = ["\n\n", "\n", " ", ""],   # 통상 디폴트
)

split_doc1 = loader1.load_and_split(text_splitter)   # 청크 300, 오버랩 100
split_doc2 = loader2.load_and_split(text_splitter)   # 청크 300, 오버랩 100

# 문서 개수 확인
# print(split_doc1)
print(len(split_doc1), len(split_doc2))     # 9 9

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    # model='text-embedding-3-large', # 3072
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)

DB_PATH = './_db/Chroma11/'
# 저장
db = Chroma.from_documents(
    documents=split_doc1+split_doc2,
    embedding=embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma11"
)

print("Chroma 문서저장 끝")
