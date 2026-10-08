# 챗봇까지 맹그러봐요
# transformer 논문을 벡터 디비로 불러와서
# 요약, 인용 등등 할 수 있는 챗봇으로!!!

# 11-1 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import PyPDFLoader

# pip install faiss-cpu
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.
api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

#01 데이터 불러온다.
path = './_data/'
pdf_loader = PyPDFLoader(path + "attention is all you needs.pdf")
pdf_docs = pdf_loader.load()

# # 문서를 자른다 / 청킹
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 300,
#     chunk_overlap = 100,
#     separators = ["\n\n", "\n", " ", ""],   # 통상 디폴트
# )

# split_doc = pdf_docs.load_and_split(text_splitter)   # 청크 300, 오버랩 100

# 문서 개수 확인
# print(split_doc1)
print(len(pdf_docs))     # 9 9
# exit()
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name='Qwen/Qwen3-Embedding-0.6B',
    model_kwargs={
        # "device" : "cuda", # Torch not compiled with CUDA enabled
        "device" : "cpu",
        # "local_files_only" : True,
    }
)

########################### 요기부터 faiss #########################
# faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))) # 어떤 문장을 써도 1536
# faiss_index = faiss.IndexFlatL2(1536)
# print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
# print(faiss_index.d)

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
    documents=pdf_docs,
    embedding = embeddings,

)

DB_PATH = "./_db/Faiss20-2"
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index20-2'
)


