from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool

from pathlib import Path
from typing import List, Dict


def vector_query(
    nodes:list,
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """
    vector_index = VectorStoreIndex(nodes)
    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    response = query_engine.query(query)
    return response
    



def get_vector_tool(nodes:List,algo:str):

    vector_index = VectorStoreIndex(nodes)
    def vector_query(query:str) -> str:
        """Perform a vector search over an index.
    
        query (str): the string query to be embedded.
        """
        query_engine = vector_index.as_query_engine(similarity_top_k=4)
        response = query_engine.query(query)
        return response
    vector_query_tool = FunctionTool.from_defaults(
    name=f"{algo}_vector_tool",
    fn=vector_query
    )

    return vector_query_tool

def get_summary_tool(nodes:List, algo:str,llm:any):
    '''returns vector index and summary index tools for the provided document'''

    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"{algo}_summary_tool",
        query_engine=summary_query_engine,
        description=(
            f"Useful if you want to get a summary or explanation of {algo} algorithm"
        ),
    )

    return summary_tool

def get_tools(doc_paths:Dict, loader:any,llm:any) -> Dict:


    algo_to_tools_dict = {}
    for algo, doc_path in doc_paths.items():
        document = loader.load_data(file=Path(doc_path))
        splitter = TokenTextSplitter(
            chunk_size=64,
            chunk_overlap=10,
            separator=" ",
        )
        nodes = splitter.get_nodes_from_documents(document)
        if len(nodes)<=1:
            raise ValueError('Number of generated nodes is less than or equal to 1. Check the document at {doc_path}')
        vector_tool = get_vector_tool(nodes=nodes, algo=algo)
        summary_tool = get_summary_tool(nodes=nodes, algo=algo,llm=llm)
        algo_to_tools_dict[algo] = [vector_tool, summary_tool]
    algos = list(doc_paths.keys())
    tools = [tool for algo in algos for tool in algo_to_tools_dict[algo]]

    return tools