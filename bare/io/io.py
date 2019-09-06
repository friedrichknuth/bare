import os
import xmltodict
import json

def create_dir(directory):
    if directory == None:
        return None
    
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.abspath(directory)

def xml_to_json(xml_file_name):
    with open(xml_file_name, 'r') as f:
        xml_as_json = json.loads(json.dumps(xmltodict.parse(f.read())))
        
    return xml_as_json
    
def split_file(file_path_and_name):
    file_path = os.path.split(file_path_and_name)[0]
    file_name = os.path.splitext(os.path.split(file_path_and_name)[-1])[0]
    file_extension = os.path.splitext(os.path.split(file_path_and_name)[-1])[-1]
    
    return file_path, file_name, file_extension