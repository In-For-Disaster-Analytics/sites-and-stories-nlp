import ipywidgets as widgets #documentation: https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Events.html
from IPython.display import display
import zipfile
from ipyfilechooser import FileChooser # Documentation: https://github.com/crahan/ipyfilechooser
import os


from llama_index import SimpleDirectoryReader

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