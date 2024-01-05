import ipywidgets as widgets #documentation: https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Events.html
from IPython.display import display
import zipfile

from ipyfilechooser import FileChooser # Documentation: https://github.com/crahan/ipyfilechooser
import os
import subprocess
import logging
import sys
from IPython.display import Markdown, display
from llama_index.query_engine import CitationQueryEngine
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate, PromptType
from pathlib import Path
from llama_index import download_loader, KnowledgeGraphIndex, SimpleDirectoryReader, VectorStoreIndex, ServiceContext, set_global_service_context
import time
import glob
import shutil
from functools import partial
from multiprocessing.pool import ThreadPool
import multiprocessing

def copy( src: str,dst : str):
    dest_file = os.path.join(dst, src.split("/")[-1])
    if os.path.exists(dest_file)==False:
        shutil.copy2(src=src, dst=dst)
    else:
        print(f"File exists: {dest_file}")
    return None

def multi_copy(DST_DIR, SRC_DIR):
    if os.path.exists(DST_DIR)==False:
        os.mkdir(DST_DIR)
    # copy_to_mydir will copy any file you give it to DST_DIR
    copy_to_mydir = partial(copy, dst=f"{DST_DIR}")
    
    # list of files we want to copy
    to_copy = glob.glob(os.path.join(SRC_DIR, '*'))
    length = len(to_copy)
    progress = widgets.IntProgress(
            value=0,
            min=0,
            max=len(to_copy),
            description='Loading:',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            style={'bar_color': 'green'},
            orientation='horizontal'
        )
    display(progress)
    if length> (multiprocessing.cpu_count()//2)-1:
        length = (multiprocessing.cpu_count()//2)-1
    with ThreadPool(length) as p:
      for i, result in enumerate(p.map(copy_to_mydir, to_copy)):
           progress.value= i
    return None


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


scratch = os.getenv('SCRATCH') 
print(scratch)
corral_llms = "/corral-tacc/projects/TDIS/LLMs/HF"

class Story_Engine():
    def __init__(self):
        self.llm = LLM()
        self.corpus = Corpus()
        self.accordion = widgets.Accordion(children=[self.llm.llm_select_widget, self.corpus.corpus_select_widget ])
        self.accordion.set_title(0, 'Choose The LLM you want to use.')
        self.accordion.set_title(1, 'Select the Corpus to base your queries on.')
        
        display(self.accordion)
        
    def create_index(self):
        self.index = VectorStoreIndex.from_documents(self.corpus.documents, service_context=self.llm.service_context)
    
    def create_query_engine(self, k=3, chunk_size=512):
        # Better Explain Each of these steps. 
        self.query_engine = CitationQueryEngine.from_args(
        self.index,
        similarity_top_k=k,
        # here we can control how granular citation sources are, the default is 512
        citation_chunk_size=512,
        )
   
    def query(self, text):
        
        self.response = self.query_engine.query(text)
        display(Markdown(f"<b>{self.response}</b>"));
        self.corpus.corpus_panel(self.response.source_nodes)
        return self.response

class LLM:
    def __init__(self, embed_model = "local:BAAI/bge-small-en"):
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
            options=['LLAMA2 7B',"LLAMA2 7B CHAT", 'LLAMA2 13B', 'LLAMA2 13B CHAT',
                    #  'Mixtral'
                     ],
            value='LLAMA2 13B CHAT',
            description='Model:',
            disabled=False,
            )
        self.location = {"LLAMA2 7B":"Llama7b",
                        "LLAMA2 7B CHAT": "Llama7bchat",
                        "LLAMA2 13B":  "Llama-2-13b-hf",
                        "LLAMA2 13B CHAT": "Llama-2-13b-chat-hf",
                        "Mixtral": "Mixtral-8x7B-v0.1"
                        }
        self.llm_select_widget = widgets.HBox([self.dropdown, self.button])
        
        self.button.on_click(self.on_button_clicked)

    def get_llm_path(self):
        return  os.path.join(corral_llms, self.location[self.dropdown.value])
        
    def on_button_clicked(self, path):
        self.path = os.path.join(scratch,  self.location[self.dropdown.value])
        
        # print(subprocess.call(["rsync","-r", self.get_llm_path(), self.path]))
        start = time.time()
        multi_copy( self.path, self.get_llm_path())
        print((time.time()-start)/60)
          
        self.load_llm()
        self.model_loaded=True
        
    def load_llm(self ):
        system_prompt = """<|SYSTEM|># You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES]. 
Pay attention to the type of the properties, if you can't find data for a property set it to null. Don't make anything up and don't add any extra data. If you can't find any data for a node or relationship don't add it.
Only add nodes and relationships that are part of the schema. If you don't get any relationships in the schema only add nodes.

Example:
Schema: Nodes: [Person {age: integer, name: string}] Relationships: [Person, roommate, Person]
Alice is 25 years old and Bob is her roommate.
Nodes: [["Alice", "Person", {"age": 25, "name": "Alice}], ["Bob", "Person", {"name": "Bob"}]]
Relationships: [["Alice", "roommate", "Bob"]]"""
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
        self.llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=2048,
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                generate_kwargs={"temperature": 0.0, "do_sample": False},
                tokenizer_name=self.path,
                model_name=self.path,
                device_map="balanced",
                model_kwargs={ "load_in_8bit": False, "cache_dir":f"{scratch}"},
            )
        self.set_service_context()
    def set_service_context(self):
        self.service_context = ServiceContext.from_defaults(
                llm=self.llm, embed_model=self.embed_model
            )
        
        set_global_service_context(self.service_context)

class Corpus:
    def __init__(self):
        
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
    def unzip(self, filename):
        with zipfile.ZipFile(self.fc.selected, 'r') as zip_ref:
            print(os.path.join(scratch, "Corpus"))
            zip_ref.extractall( os.path.join(scratch, "Corpus"))
        self.corpus_path = os.path.join(scratch, "Corpus", filename.split(".")[0])
        
    def load_corpus(self, button):
        if self.fc.selected_filename ==None: 
            filename = self.fc.default_filename
            path = self.fc.default_path
        else:
            filename = self.fc.selected_filename
            path = self.fc.selected_path
        if filename.endswith('zip'):
            self.unzip(filename)
        else: 
            self.corpus_path = path
        self.required_exts = ['.pdf', '.htm','.html']
        self.reader = SimpleDirectoryReader(
            input_dir=self.corpus_path,
            required_exts=self.required_exts,
            recursive=True,
        )
        self.documents = self.reader.load_data()
        display(f"Loaded {len(self.documents)} docs")
        self.fc_chosen=True
        

    
    def corpus_panel(self, tab_contents):
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
            description=f"Page: {i}")
                    for i, node in enumerate(tab_contents)]
        tab = widgets.Tab(layout=widgets.Layout( height='280px'))
        tab.children = children
        for i in range(len(children)):
            tab.set_title(i, str(i))
        display(tab)