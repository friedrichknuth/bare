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