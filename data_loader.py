#use LlamaIndex to load PDF docs, chunk them and embed them, 
#then stock the embeddings in Qdrant vectorDB (logic in : vectorDB.py)

from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EMBED_model = "text-embedding-3-large" #OpenAI embedding model
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size= 1000, chunk_overlap=200) #split PDF into chunks of 1000 characters with an overlap of 200 characters

#chunking process
def load_and_chunk_pdf(path:str):
    docs = PDFReader().load_data(file=path)
    #only looking at text not images
    #get all of the text ccontent for every single doc in our docs if the doc has text content
    texts = [d.text for d in docs if getattr (d, "text", None)]
    chunks = []
    for t in texts:
        #split the text into chunks using our splitter
        chunks.extend(splitter.split_text(t))
    return chunks

#embedding process
def embed_texts(texts : list[str]) -> list[list[float]] :
    response = client.embeddings.create(model=EMBED_model, input=texts) 
    return [item.embedding for item in response.data]
