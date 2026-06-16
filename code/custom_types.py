#the most widely used data validation and serialization library for python 
# pydantic.BaseModel automatically:
# * validates data types
# * converts compatible data when possible
# * makes data easy to serialize to JSON
import pydantic 

#represents a document after it has been split into chunks
class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str] #text chunks extracted from a document
    source_id: str = None #identifier of the original source document
#output example: {"chunks": ["chunk1 text", "chunk2 text"], "source_id": "document1.pdf"} => the document "document1.pdf" has been split into 2 chunks of text
#first function of chunking (in data_loader.py) second this function

#stores the result of upserting embeddings into a vectorDB Qdrant
class RAGUpsertResult(pydantic.BaseModel):
    ingested: int #number of chunks successfully stored
#output example: {"ingested": 150} => 150 chunks have been successfully stored in vectorDB

#represents retrieved information from the vector database.
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str] #retrieved text chunks
    sources: list[str] #IDs corresponding to those chunks
#RAG workflow: user question -> vector search -> RAGsearchResult

#represents the final response returned to the user after retrieval and generation
class RAGQueryResult(pydantic.BaseModel):
    answer: str #generated answer from the LLM
    sources: list[str] #documents used to generate the answer
    num_context: int #number of retrieved chunks used
#output example: {"answer": "The capital of France is Paris.", "sources": ["document1.pdf", "document2.pdf"], "num_context": 5} 
