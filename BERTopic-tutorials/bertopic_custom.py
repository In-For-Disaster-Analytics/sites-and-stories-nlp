import pandas as pd
import os
import openai
from openai import OpenAI

import ipywidgets as widgets

## Functions
def check_openai_api_key(api_key):
    if api_key == '':
        return False
    else:
        client = OpenAI(api_key=api_key)
        try:
            client.models.list()
        except openai.AuthenticationError:
            return False
        else:
            return True

def print_openai_api_key_status(api_key):
    if api_key == '':
        print('No OpenAI key provided.  Please continue if this is intentional')
    else:
        is_valid = check_openai_api_key(api_key)
        if is_valid:
            print("Valid OpenAI API key.")
        else:
            print("Invalid OpenAI API key.")

## Widget definitions
openai_token_widget = widgets.Text(
        value='',
        placeholder='Enter your OpenAI token here to use ChatGPT',
        description='OpenAI Token:',
        disabled=False,
        style={'description_width': 'initial'},
        layout = widgets.Layout(width='70%')
    )

# BERTopic Wizard model class                       
class BERTopicWizard():
    """Class that returns a BERTopic model
    """
    def __init__(self):
        """Initialize the openai_token_widget. 
            Create an openai_token_widget widget for Interaction with the User. 
        """
        self.openai_token_widget = widgets.Text(
                    value='',
                    placeholder='Enter your OpenAI token here to use ChatGPT',
                    description='OpenAI Token:',
                    disabled=False,
                    style={'description_width': 'initial'},
                    layout = widgets.Layout(width='70%')
                )
        display(self.openai_token_widget)
        self.value = self.openai_token_widget.value 
        # 
