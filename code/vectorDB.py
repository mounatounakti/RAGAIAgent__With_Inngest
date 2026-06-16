# *** VECTOR DB ***
#create vectorDB class to connect to our vectorDB, stock data and search for similar vectors

#connect to our vectorDB & search for similar vectors
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    #collection where we are going to stock our data in vectorDB, we create it if it doesn't exist
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        #client is the connection to our vectorDB, we will use it to send commands to the vectorDB
        self.client = QdrantClient(url=url, timout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        
    #payloads are the metadata we want to stock with our vectorDB (title, author name ...)
    #upsert = insert or update
    def upsert(self, ids, vectors, payloads):
        #each point is a vector with its metadata. We create a list of points to upsert in our vectorDB
        points = [PointStruct(id=id[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        #insert now the point in vectorDB
        self.client.upsert(self.collection, points=points)

    def search (self, query_vector, top_k: int = 5):
        #search for similar vectors in vectorDB, we get the top_k most similar vectors
        results = self.client.search(
            collection_name = self.collection,
            query_vector = query_vector, #search based on this query
            with_payload = True,
            limit = top_k # we are looking for 5 results (top-k similar) from vectorDB
        )

        #all the information we get from vectorDB about the similar vectors will be stored in contexts, 
        #we will use it later to create the prompt for LLM
        contexts = [] 
        #source of each information (original file name)
        sources = set ()

        #orginize result into a dictionary
        for r in results:
            payload = getattr(r, "payload, None") or {}
            text = payload.get("text","")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.append(source)

        return {"contexts": contexts, "sources": list(sources)}



