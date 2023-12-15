import ipywidgets as widgets #documentation: https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Events.html
from IPython.display import display
import zipfile

from ipyfilechooser import FileChooser # Documentation: https://github.com/crahan/ipyfilechooser
import os
import shutil
import os
import logging
import sys
from IPython.display import Markdown, display
from llama_index.query_engine import CitationQueryEngine
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate, PromptType
from pathlib import Path
from llama_index import download_loader, KnowledgeGraphIndex, SimpleDirectoryReader
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


scratch = os.getenv('SCRATCH') 
print(scratch)
corral_llms = "/corral-tacc/projects/TDIS/LLMs/HF"


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

        self.dropdown = widgets.Dropdown(
            options=['LLAMA2 7B',"LLAMA2 7B CHAT", 'LLAMA2 13B', 'LLAMA2 13B CHAT'],
            value='LLAMA2 13B CHAT',
            description='Model:',
            disabled=False,
            )
        self.location = {"LLAMA2 7B":"Llama7b",
                        "LLAMA2 7B CHAT": "Llama7bchat",
                        "LLAMA2 13B":  "Llama-2-13b-hf",
                        "LLAMA2 13B CHAT": "Llama-2-13b-chat-hf"}
        display(self.dropdown)
        display(self.button)
        self.button.on_click(self.on_button_clicked)

    def get_llm_path(self):
        return  os.path.join(corral_llms, self.location[self.dropdown.value])
        
    def on_button_clicked(self, path):
        self.path = os.path.join(scratch,  self.location[self.dropdown.value])
        if os.path.exists(self.path)==False:
            shutil.copytree(self.get_llm_path(), self.path )
        self.load_llm()
        
    def load_llm(self ):
        system_prompt = """<|SYSTEM|># Your system prompt here"""
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
        display(self.fc)
        self.button = widgets.Button(
    description='Choose Corpus',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Choose Corpus',
    icon='check'
)
        display(self.button)
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
        self.required_exts = ['.pdf']
        self.reader = SimpleDirectoryReader(
            input_dir=self.corpus_path,
            required_exts=self.required_exts,
            recursive=True,
        )
        self.documents = self.reader.load_data()
        display(f"Loaded {len(self.documents)} docs")
    def create_index(self, service_context):
        self.index = VectorStoreIndex.from_documents(self.documents, service_context=service_context)
