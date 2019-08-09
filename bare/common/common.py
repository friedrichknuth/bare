import os

def create_dir(directory):
    if directory == None:
        return None
    
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.abspath(directory)
