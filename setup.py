from setuptools import setup , find_packages 
from typing import List 

def get_requirements(path:str)->List[str]:

    requirements = []
    with open(path) as file_obj :
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements if req != '-e .']
    
    return requirements 

setup(
    name = 'MLPROJECT',
    version='0.0.1',
    author='KasaPavan',
    author_email='pavankasa86@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)