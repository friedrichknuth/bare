import os

class Common:
    
    @staticmethod
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.abspath(directory)
