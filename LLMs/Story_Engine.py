import ipywidgets as widgets #documentation: https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Events.html
from IPython.display import display
import os
from IPython.display import Markdown, display
from llama_index.query_engine import CitationQueryEngine

from pathlib import Path
from llama_index import  VectorStoreIndex
import time
from LLMs.utils import multi_copy
from transformers import  __version__
import torch
from LLMs.LLM import LLM
from LLMs.Corpus import Corpus



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scratch = os.getenv('SCRATCH') 
corral_llms = "/corral-tacc/projects/TDIS/LLMs/HF"



class Story_Engine():
    """Story engine is a class that manages the LLM and Corpus Class. The goal is to simplify the interactions
    between the LLM and Corpus. 
    """
    def __init__(self):
        """Initialize the Story Engine Classs. 
            Create a LLM and Corpus Class
            Create an Accoridon widget for Interaction with the User. 
        """
        self.llm = LLM()
        self.corpus = Corpus()
        self.accordion = widgets.Accordion(children=[self.llm.llm_select_widget, self.corpus.corpus_select_widget ])
        self.accordion.set_title(0, 'Choose The LLM you want to use.')
        self.accordion.set_title(1, 'Select the Corpus to base your queries on.')
        
        display(self.accordion)
        
    def create_index(self):
        """Based on the loaded Corpus create a vector store index. 
        """
        try: 
            self.index = VectorStoreIndex.from_documents(self.corpus.documents, service_context=self.llm.service_context)
        except:
            if self.llm.model_loaded==False or self.corpus.fc_chosen==False:
                print("Make Sure you have loaded both the Model and the Corpus in the previous cell block. \
                      The buttons should be green. ") 

    def create_query_engine(self, k :int=3, chunk_size :int=512, engine_type:str = "Citation"):
        """Once the vector index has been created, the next step is an engine. This provides a wrapper for a different Engines. 
        Currently Citation has been implemented, Others will be added later. 

        Args:
            k (int, optional): Number of Corpus Nearest Neighbors to draw from. Defaults to 3.
            chunk_size (int, optional): Size of the documents added to the engine. . Defaults to 512.
            engine_type (str, optional): Query engine to declare. Defaults to "Citation".
        """
        if engine_type=="Citation":
            self.query_engine = CitationQueryEngine.from_args(
                self.index,
                similarity_top_k=k,
                # here we can control how granular citation sources are, the default is 512
                citation_chunk_size=512,
            )
        else: 
            self.query_engine = self.index.as_query_engine(streaming=True)
   
    def query(self, text:str):
        """Function that prompts the query engine, and organizes the responses. Defaults to organizings citations below. 

        Args:
            text str: Prompt to as the query engine and corpus

        Returns:
            str: Return response after visually formatting. 
        """
        self.response = self.query_engine.query(text)
        display(Markdown(f"<b>{self.response}</b>"));
        self.corpus.corpus_panel(self.response.source_nodes)
        return self.response

