# 11-1 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

# pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

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

########################### 요기부터 faiss #########################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))) # 어떤 문장을 써도 1536
# faiss_index = faiss.IndexFlatL2(1536)
print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)

# faiss_db = FAISS(
#     embedding_function=embeddings,
#     index= faiss_index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )
# # 저장된 문서의 갯수 확인.
# print(faiss_db.index.ntotal)    # 0
########################## 준비 완료 ##########################
##############################################################

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding = embeddings,

)

DB_PATH = "./_db/Faiss17"
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17'
)