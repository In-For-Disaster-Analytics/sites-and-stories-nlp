import ipywidgets as widgets
from IPython.display import display
import zipfile
from ipyfilechooser import FileChooser
from ipyfilechooser import FileChooser
import os

corral_llms = "/corral-tacc/projects/TDIS/LLMs/HF"
class llm_location:
    def __init__(self):
        self.dropdown = widgets.Dropdown(
    options=['LLAMA2 7B',"LLAMA2 7B CHAT", 'LLAMA2 13B', 'LLAMA2 13B CHAT'],
    value='LLAMA2 13B CHAT',
    description='Number:',
    disabled=False,
    )
        self.location = {"LLAMA2 7B": os.path.join(corral_llms, "Llama7b"),
                        "LLAMA2 7B CHAT": os.path.join(corral_llms, "Llama7bchat"),
                        "LLAMA2 13B": os.path.join(corral_llms, "Llama-2-13b-hf"),
                        "LLAMA2 13B CHAT":os.path.join(corral_llms, "Llama-2-13b-chat-hf")}
    def get_llm_path(self):
        return self.location[self.dropdown.value]


class corpus:
    def __init__(self):
        self.fc = FileChooser('./')

    def unzip(self):
        with zipfile.ZipFile(self.fc.selected, 'r') as zip_ref:
            zip_ref.extractall("Corpus")
        self.corpus_path = os.path.join(self.fc.selected_path, "Corpus", self.fc.selected_filename.split(".")[0])
