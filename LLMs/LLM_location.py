import ipywidgets as widgets #documentation: https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Events.html
from IPython.display import display
import zipfile
from ipyfilechooser import FileChooser # Documentation: https://github.com/crahan/ipyfilechooser
import os
import logging
import sys
from IPython.display import Markdown, display
from llama_index.query_engine import CitationQueryEngine
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from pathlib import Path
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, set_global_service_context
import time
from utils import multi_copy
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, __version__
import torch


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scratch = os.getenv('SCRATCH') 
corral_llms = "/corral-tacc/projects/TDIS/LLMs/HF"
system_prompt = """<|SYSTEM|># You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES]. 
Pay attention to the type of the properties, if you can't find data for a property set it to null. Don't make anything up and don't add any extra data. If you can't find any data for a node or relationship don't add it.
Only add nodes and relationships that are part of the schema. If you don't get any relationships in the schema only add nodes.

Example:
Schema: Nodes: [Person {age: integer, name: string}] Relationships: [Person, roommate, Person]
Alice is 25 years old and Bob is her roommate.
Nodes: [["Alice", "Person", {"age": 25, "name": "Alice}], ["Bob", "Person", {"name": "Bob"}]]
Relationships: [["Alice", "roommate", "Bob"]]"""
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
            if self.model_loaded==False or self.corpus.fc_chosen==False:
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

class LLM:
    """Large Language Model Wrapper Class
    """
    def __init__(self, embed_model = "local:BAAI/bge-small-en"): 
        #alternative embed: "local:all-MiniLM-L6-v2"
        """Initialize GUI for model. Allows for selecting local models through a drop down."""
        self.button = widgets.Button(
            description='Load Model',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Submit',
            icon='check'
        )
        self.embed_model = embed_model 
        self.model_loaded = False
        self.dropdown = widgets.Dropdown(
            options=['LLAMA2 7B',"LLAMA2 7B CHAT", 'LLAMA2 13B', 'LLAMA2 13B CHAT',"LLAMA2 70B CHAT",
                     'Zephyr'
                     ],
            value='LLAMA2 13B CHAT',
            description='Model:',
            disabled=False,
            )
        self.location = {"LLAMA2 7B":"Llama7b",
                        "LLAMA2 7B CHAT": "Llama7bchat",
                        "LLAMA2 13B":  "Llama-2-13b-hf",
                        "LLAMA2 13B CHAT": "Llama-2-13b-chat-hf",
                         "LLAMA2 70B CHAT":"Llama-2-70b-chat-hf",
                        "Zephyr": "zephyr-7b-beta"
                        }
        self.llm_select_widget = widgets.HBox([self.dropdown, self.button])
        
        self.button.on_click(self.on_button_clicked)

    def get_llm_path(self):
        """Looks up the path of the LLM based on the self.location Dictionary

        Returns:
            str: directory location for the LLM
        """
        return  os.path.join(corral_llms, self.location[self.dropdown.value])
        
    def on_button_clicked(self, path:str):
        """Copies then loads the LLM based on the dropdown widget selections. 

        Args:
            path (str): Directory Location
        """
        self.button.button_style = "warning"
        self.path = os.path.join(scratch,  self.location[self.dropdown.value])
        
        ## Multiprocessing copy of the LLM to ensure speedy transfers. 
        multi_copy( self.path, self.get_llm_path())
        self.load_llm()
        self.model_loaded=True
        self.button.button_style = 'success'


    def load_llm(self, system_prompt:str=system_prompt, context_window:int=4096, max_new_tokens:int=1024):
        """Loads the selected LLM. 

        Args:
            system_prompt (str, optional): Prompt used prior to the query to give the LLM directions. Defaults to system_prompt.
            context_window (int, optional): Total number of characters that can be loaded into the prompt at once. . Defaults to 4096.
            max_new_tokens (int, optional): Total number of characters the LLM can respond withl. Defaults to 1024.
        """
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
        self.model = AutoModelForCausalLM.from_pretrained(self.path, 
                                             device_map='auto', 
                                             torch_dtype=torch.float16, 
                                             rope_scaling={"type": "dynamic", "factor": 2},
                                             load_in_8bit=True
                                            ) 
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.llm = HuggingFaceLLM(
                context_window=context_window,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                generate_kwargs={"temperature": 0.1, "do_sample": True},
                # tokenizer_name=self.path,
                # model_name=self.path,
                model=self.model,
                tokenizer = self.tokenizer,
                device_map="balanced",
                model_kwargs={ "load_in_8bit": False, "cache_dir":f"{scratch}"},
            )
        self.set_service_context()

    def set_service_context(self):
        """Sets the model and embed model for querying
        """
        self.service_context = ServiceContext.from_defaults(
                num_output=256,  # The amount of token-space to leave in input for generation.

                llm=self.llm, embed_model=self.embed_model
            )
        
        set_global_service_context(self.service_context)

class Corpus:
    def __init__(self):
        """Initialize the class that selects PDF's MD, or Directories and reads them into memory as a corpus.
        """
        self.fc = FileChooser('./')
        self.fc_chosen = False
        self.button = widgets.Button(
            description='Choose Corpus',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Choose Corpus',
            icon='check'
        )
        self.corpus_select_widget = widgets.VBox([self.fc, self.button])
        self.button.on_click(self.load_corpus)

    def unzip(self, filename:str):
        """ If the corpus is compressd, Unzips the corpus and stores in $SCRATCH.

        Args:
            filename (str): Corpus File Name
        """
        with zipfile.ZipFile(self.fc.selected, 'r') as zip_ref:
            print(os.path.join(scratch, "Corpus"))
            zip_ref.extractall( os.path.join(scratch, "Corpus"))
        self.corpus_path = os.path.join(scratch, "Corpus", filename.split(".")[0])
        
    def load_corpus(self, x):
        print(x)
        """Once the corpus is selected, Load the corpus into memory. 
        """
        self.button.button_style = 'warning'
        ## Makesure a file has been selected
        if self.fc.selected_filename == None: 
            filename = self.fc.default_filename
            path = self.fc.default_path
        else:
            filename = self.fc.selected_filename
            path = self.fc.selected_path
        #Check if selected is compressed or not. 
        if filename.endswith('zip'):
            self.unzip(filename)
            self.required_exts = ['.pdf', '.htm','.html']
            self.reader = SimpleDirectoryReader(
                input_dir=self.corpus_path,
                required_exts=self.required_exts,
                recursive=True,
            )
        else: 
            self.corpus_path = os.path.join(path, filename)
            self.reader = SimpleDirectoryReader(
                input_files=[self.corpus_path]
            )
        
        self.documents = self.reader.load_data()
        display(f"Loaded {len(self.documents)} docs")
        self.fc_chosen=True
        self.button.button_style = 'success'
        

    
    def corpus_panel(self, tab_contents:list):
        """Createa GUI with a subset of the Corpus. 

        Args:
            tab_contents (list): Documents from Corpus
        """
        try:
            children = [widgets.HTML(
                            value=f"<b>{node.metadata['file_name']}</b><br>{node.text}",
                            placeholder='Some HTML',
                            description=f"Page: {node.metadata['page_label']}")
                                    for node in tab_contents]
        except: 
            children = [widgets.HTML(
                    value=f"<b>{node.metadata['file_name']}</b><br>{node.text}",
                    placeholder='Some HTML',
                    description=f"Page: {i+1}")
                            for i, node in enumerate(tab_contents)]
        tab = widgets.Tab(layout=widgets.Layout( height='280px'))
        tab.children = children
        for i in range(len(children)):
            tab.set_title(i, str(i))
        display(tab)