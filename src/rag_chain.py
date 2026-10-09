from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Formats retrieved Document chunks with page numbers for clean LLM context."""
    if not docs:
        return "No relevant context found in document."
    formatted = []
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get("page", None)
        page_str = f"Page {page_num + 1}" if isinstance(page_num, int) else "Page N/A"
        formatted.append(f"--- Context {i+1} ({page_str}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_rag_chain(retrieved_docs, llm):
    """
    Builds the RAG pipeline:
    query -> [retriever -> format_docs, query] -> prompt -> llm -> StrOutputParser
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent assistant for answering questions based on retrieved document context. "
                   "Provide clear, concise, and accurate answers based on the retrieved text. "
                   "If the answer cannot be found in the context, truthfully state that the document does not contain that information."),
        ("human", "Answer the following question based on the retrieved document context:\n\n"
                  "Context:\n{retrieved_docs}\n\n"
                  "Question: {question}\n\n"
                  "Answer:")
    ])
    
    rag_chain = (
        {"retrieved_docs": retrieved_docs | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain