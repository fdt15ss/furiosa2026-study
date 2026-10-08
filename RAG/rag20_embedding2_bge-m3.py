# 10-1 카피

# LCEL = Lanchain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 공백이나 줄바꿈 삭제함
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = "삼성전자의 창업주는 누구인가요?"

# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     model='text-embedding-3-small',
#     api_key=api_key,
#     base_url=base_url,
# )
# pip install langchain-huggingface
# pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={
        # "device" : "cuda", # Torch not compiled with CUDA enabled
        "device" : "cpu",
        # "local_files_only" : True,
    }
)

# exit()
vector = embeddings.embed_query(prompt)
print(vector)
print("==========================================")
print(f"임베딩 벡터의 차원 : {len(vector)}") #1024 # 벡터는 한개인데 차원이 1024개다.
