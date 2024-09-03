from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_text_splitters import CharacterTextSplitter


class DocumentSummarizer:
    def __init__(self, 
                 llm, 
                 chunk_size=1000, 
                 chunk_overlap=0, 
                 token_max=4000):
        self.llm = llm
        
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        
        # Create map and reduce prompts
        self.map_template = """The following is a set of documents:
{docs}
Based on this list of docs, please summarize the document
Helpful Answer:"""
        self.reduce_template = """The following is a set of documents:
{docs}
Based on this list of docs, please summarize the document
Helpful Answer:"""

        # Create LLM Chains
        self.map_prompt = PromptTemplate.from_template(self.map_template)
        self.map_chain = LLMChain(llm=self.llm, prompt=self.map_prompt)

        self.reduce_prompt = ChatPromptTemplate.from_template(self.reduce_template)
        self.reduce_chain = LLMChain(llm=self.llm, prompt=self.reduce_prompt)

        # Create Document Chains
        self.combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain, document_variable_name="docs"
        )

        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=self.combine_documents_chain,
            collapse_documents_chain=self.combine_documents_chain,
            token_max=token_max,
        )

        # Create the Map-Reduce Chain
        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=self.map_chain,
            reduce_documents_chain=self.reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

    def summarize_documents(self, 
                            docs):
        """
        Summarize the provided documents using the map-reduce chain.

        Args:
            docs (list): A list of documents to be summarized.

        Returns:
            str: The summarized text.
        """
        split_docs = self.text_splitter.split_documents(docs)
    
        return self.map_reduce_chain.invoke(split_docs)["output_text"]