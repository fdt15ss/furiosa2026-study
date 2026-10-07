import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv() # vscode 자체에서 없어도 가져옴. 하지만 안전빵으로 적어두는 게 좋다.
api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"
"""
from glob import glob

path = './_data/rag_data/'
# 폴더에서 텍스트 파일 목록 가져오기
txt_files = glob(os.path.join(path,'*.txt'))

print(txt_files)
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt',
# './_data/rag_data\\samsung_outlook.txt']
# exit()

#01 데이터 불러온다.

data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding="utf-8")
    data += loader.load()

print(data)

print("==================================")
print(len(data))        # 3

print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)   # [8158, 2049, 1898]

#02 문서를 자른다 / 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 10,
    separators = ["\n\n", "\n", " ", ""],   # 통상 디폴트
)

texts = text_splitter.split_documents(data)
print("생성된 텍스트 청크수 :", len(texts)) # 생성된 텍스트 청크수 : 52
print("각 청크의 길이 :", list(len(text.page_content) for text in texts))
# 각 청크의 길이 : [259, 282, 282, 128, 276, 214, 158, 249, 262, 291, 268, 182, 286, 295, 182, 162, 283, 286, 257, 235, 214, 258, 207, 286, 220, 198, 271, 57, 272, 122, 9, 269, 299, 284, 289, 209, 222, 230, 254, 249, 296, 90, 247, 243, 185, 219, 239, 235, 298, 282, 187, 249]

# document안에 page_content, metadata
# print("첫번째 청크의 내용 : ", texts[0].page_content)
# print("첫번째 청크의 길이 : ", len(texts[0].page_content))
# print("두번째 청크의 내용 : ", texts[1].page_content)
"""
#03. 임베딩

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    # model='text-embedding-3-large', # 3072
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,
)
# sample_text = "삼성전자의 창업자는 누구인가요?"
# vector = embeddings.embed_query(sample_text)
# # print(vector)
# print(len(vector))  # 1536

# exit()
DB_PATH = './_db/Chroma12/'
# 저장
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory = DB_PATH,
#     collection_name = "croma12"
# )
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory = DB_PATH,
    collection_name = "croma12"
)

print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")
# 벡터 저장소에 저장된 문서 수 : 52

query = " 삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query)  # k값 기본값은 4

print(f"검색 결과의 길이: {len(result)}")       # 4

################################### Retrievers ########################################
################################### 검색기 ###########################################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)
aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}")
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")