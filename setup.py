from setuptools import setup, find_packages

import sys
sys.path.append(os.path.join(os.path.os.path.dirname(__file__), 'src'))

setup(
    name = "trulens_viz",
    version = "0.0.1",
    packages = find_packages(),

    install_requires = [
                        'domonic>=0.9.10',
                        'matplotlib>=3.5.1',
                        'numpy>=1.22.3',
                        ],
    
    author = "Andreas Paepcke",
    author_email = "paepcke@cs.stanford.edu",
    description = "Draft of NLP related attribution strength",
    url = "https://github.com/paepcke/trulens_viz",
    )
