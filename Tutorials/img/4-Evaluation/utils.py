import ipywidgets as widgets
import ollama 

class model_chooser():

    def click(self, x):
        ollama.pull(self.Model.value)
        self.button.button_style= 'success'

    def __init__(self):
        self.Model = widgets.Combobox(
            options=['mixtral', 'mistral', 'llama2'],
            # value='mixtral',
            description='Number:',
            disabled=False,
        )

        self.button = widgets.Button(
            description='Select',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )

        self.button.on_click(self.click)
        self.vbox = widgets.VBox([self.Model, self.button])

        display(self.vbox)