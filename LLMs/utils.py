import os
import glob
import shutil
from functools import partial
from multiprocessing.pool import ThreadPool
import multiprocessing
def copy( src: str,dst : str):
    """This is a function to copy files using SHUTIL.copy2. The intension is to use this function to multiprocess the copy

    Args:
        src (str): Source Directory to be copied
        dst (str): Destination Directory to be copied. 

    Returns:
        _type_: None
    """
    dest_file = os.path.join(dst, src.split("/")[-1])
    if os.path.exists(dest_file)==False:
        shutil.copy2(src=src, dst=dst)
    else:
        print(f"File exists: {dest_file}")
    return None

def multi_copy(DST_DIR: str, SRC_DIR : str ):
    """THis function maps the copy function across multiple cores. 


    Args:
        DST_DIR (str):  Source Directory to be copied
        SRC_DIR (str): Destination Directory to be copied. 

    Returns:
        _type_: _description_
    """
    if os.path.exists(DST_DIR)==False:
        os.mkdir(DST_DIR)
    # copy_to_mydir will copy any file you give it to DST_DIR
    copy_to_mydir = partial(copy, dst=f"{DST_DIR}")
    
    # list of files we want to copy
    to_copy = glob.glob(os.path.join(SRC_DIR, '*'))
    length = len(to_copy)
    if length > (multiprocessing.cpu_count()//2)-1:
        length = (multiprocessing.cpu_count()//2)-1
    with ThreadPool(length) as p:
      p.map(copy_to_mydir, to_copy)
           
    return None
