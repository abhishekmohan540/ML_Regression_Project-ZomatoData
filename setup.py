from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()    # Reading each lines from requirements.txt
        requirements=[req.replace("\n","") for req in requirements]     # Removing \n from each line

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)          # -e . is not a installable file, hence removing it.

    return requirements


setup(
    name='ZomatoProject',
    version='1.0.0',
    author='abhishekmohan',
    author_email='abhishekmohan540@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)