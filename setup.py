from setuptools import setup

setup(
    name='bertron',
    version='0.0.1',
    packages=['tests', 'bertron', 'bertron.models'],
    url='https://github.com/btlakey/bertron',
    license='',
    author='blakey',
    author_email='brianlakey@gmail.com',
    description='(toy) BERT authorship assignment against Enron Corpus',
    install_requires=[
        'numpy==1.20.1',
        'pandas==1.2.4',
        'pyarrow==4.0.0',
        'scikit-learn==0.24.1',
        'tokenizers==0.10.2',
        'toolz==0.11.1',
        'torch==1.8.1',
        'tqdm==4.60.0',
        'transformers==4.5.1'
    ]
)
