import os
import re
import json

import numpy as np
import pandas as pd 
import fitz # this is PyMuPDF. Weird name because developers.

# Functions to load in pdf files 

def generate_block_dict(PyMuPDF_Document):
    '''Take a PyMuPDF document object and read it in using the blocks option'''
    block_dict = {}
    page_num = 1
    for page in PyMuPDF_Document:
        # Load blocks for each page
        block_dict[page_num] = page.get_text('dict')['blocks']
        page_num += 1
    return block_dict

def extract_text_data_from_block_dict(block_dict):   
    '''Take a dictionary of block objects from a PyMuPDF object and extract the text by page number and block into a data dictionary'''
    text_data = []
    
    for page in block_dict.keys():
        for block in block_dict[page]:
            block_number = block['number']
            if block['type']==0:
                linetext=''
                for line in block['lines']:
                    for span in line['spans']:
                        linetext=linetext+span['text']
                block_text_data = {
                    'Page': page,
                    'Block': block_number,
                    'Text': linetext
                }
                text_data.append(block_text_data)  
    return text_data

def extract_image_data(block_dict):
    '''Take a dictionary of block objects from a PyMuPDF object and extract images into a doata dictionary'''
    images_data = []
    for page in block_dict.keys():
        for block in block_dict[page]:
            block_number = block['number']
            if block['type']==1:
                image_block = block
                image_block['page'] = page
                images_data.append(image_block)
    return images_data

def pdf_into_data_dict(pdf_path):
    '''If there is a valid pdf path, (1) read the pdf into a PyMuPDF document; (2) generate block_dict for pdf; 
    (3) extract text; (4) extract images; (5) return this data in a dictionary with keys  "text" and "images" '''
    if os.path.exists(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            block_dict = generate_block_dict(doc)
            
            text_data = extract_text_data_from_block_dict(block_dict)    
            images_data = extract_image_data(block_dict)
        
            data = {
                'filepath': pdf_path,
                'text': text_data,
                'images': images_data        
            }
        
            return data 
        except:
            print("Something has gone wrong")
            
    elif pdf_path == '':
        print("Please enter a filename above")
    else:
        print("Your file wasn't found.  Please check the local filename entered above")     
        
# def run_pdf_worfklow(pdf_path):
#     if os.path.exists(pdf_path):
#         doc = fitz.open(pdf_path)
#         print("The pdf file: '" + pdf_path +"' has been loaded")
#         return doc
#     elif local_filename.value == '':
#         print("Please enter a filename above")
#     else:
#         print("Your file wasn't found.  Please check the local filename entered above")        

# Flags_decomposer function and process from documentation: https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-font-characteristics
def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)
    

def load_and_chunk(pdffile):
    doc = fitz.open(os.path.join(pdffile)) # Use the 'Blocks' option in the PyMuPDF package
    
    # Load each page in doc into a dictionary
    block_dict = {}
    page_num = 1
    for page in doc:
        # Load blocks for each page
        block_dict[page_num] = page.get_text('dict')['blocks']
        page_num += 1
    
    # Read blocks into dataframe
    blocksdf = pd.DataFrame()
    for page_num, blocks in block_dict.items():
        blockdf = pd.DataFrame(blocks)
        blockdf['page_num'] = page_num

        if blocksdf.empty:
            blocksdf = blockdf
        else:
            blocksdf = pd.concat([blocksdf, blockdf])
            
    # Extract span information for each block
    textcols = ['page_num', 'number', 'lines']
    textdf = blocksdf[blocksdf['type'] == 0][textcols].reset_index(drop=True)
    textdf = textdf.explode('lines')
    textdf['spans'] = textdf['lines'].apply(lambda x: x['spans'])
    textdf = textdf.explode('spans')
    textdf.reset_index(inplace=True, drop=True)           
    
    # Turn spans dictionary into dataframe and then merge back in
    spans_df = pd.DataFrame(textdf['spans'].tolist())
    alldf = pd.concat([textdf, spans_df], axis=1)    
    
    # Subset data and rename
    col_dict = {
        'page_num' : 'Page',
        'number': 'Block',
        'size': 'font_size',
        'flags':'font_flag', 
        'font': 'font',
        'color': 'color',
        'bbox':'bbox',
        'text':'text'}
    df = alldf[col_dict.keys()].copy()
    df.rename(columns=col_dict, inplace=True)    
    
    # Drop lines of empty text
    dftext = df[(df.text != ' ') & (df.text != '  ')].copy()

    dftext['page_text'] = dftext.apply(lambda x: str(x['Page'])+ ' ' == x['text'], axis = 1)
    dftext = dftext[~dftext['page_text']].copy()

    dftext.reset_index(drop=True, inplace=True)    
    
    # Add columnst o flag if text is lower case, upper case, convert Flag to human readable form
    dftext['is_bold'] = dftext['font'].apply(lambda x: True if "bold" in x.lower() else False)
    dftext['is_upper'] = dftext['text'].apply(lambda x: True if re.sub(r"[\(\[].*?[\)\]]", "", x).isupper() else False)
    dftext['Flag'] = dftext['font_flag'].apply(lambda x: flags_decomposer(x))     
    
    return dftext