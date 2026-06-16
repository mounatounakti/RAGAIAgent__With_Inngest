# *** RAG Pipeline Orchestrator Using FastAPI and Inngest ***
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vectorDB import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

#client definition
ingest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

#workflow definition
@ingest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
#the workflow function
async def rag_ingest_pdf(ctx: inngest.Context): 
    #RAG logic
    #we have 2 steps in our function
    #a step transform inngest function into a workflow
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc: #def functionName (parameter : type) -> return type (that we define in custom_types.py)
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path) #function from data_loader.py
        #return the chunks and the source_id
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        vecs = embed_texts(chunks) #function from data_loader.py
        #generate unique IDs for each chunk
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))for i in range(len(chunks))] 
        #metadata for each chunk
        payloads = [{"source": source_id, "text": chunks[i]} for i in range (len(chunks))]
        #store in VectorDB
        QdrantStorage().upsert(ids, vecs, payloads)
        #return num of chunks ingested in vectorDB
        return RAGUpsertResult(ingested=len(chunks))

    #how wel call a step. not like calling a normal function.
    chunks_and_src = await ctx.step.run("load_and_chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc) #run this first the wait
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult) #run this after the first step is done
    #return the result of the workflow as a dictionary
    return ingested.modeldump() 

async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        #embed the question
        query_vec = embed_texts([question])[0]
        #search in vectorDB
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    #get the question and top_k from the event data
    question = ctx.event.data["questions"]
    top_k = int(ctx.event.data.get("top_k", 5))

    #this is a step to run the search function and wait for the result before moving on to the next step.
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    #format the retrieved contexts into a block of text
    context_block = "\n\n".join(f" - {c}" for c in found.contexts)
    #this is the propmt
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    
    #call the llm to answer based on the retrieved context and the user question.
    adapter = ai.openai.Adapter(
        auth_key = os.getenv("OPENAI_API_KEY"),
        model = "gpt-4o-mini"
    )


    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokes": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer quesitions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    #returned answer from llm
    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_context": len(found.contexts)}


#create web server
#this is what runs API endpoints and Inngest integration
app = FastAPI()

#connects everything together 
inngest.fast_api.serve(app, ingest_client, [rag_ingest_pdf])
