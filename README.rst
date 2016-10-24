
MLsandbox
---------

To install, do the following in commandline:

1. clone MLsandbox on your local machine:

   $ git clone git@github.com:NickiRom/MLsandbox.git 

2. navigate to MLsandbox/

   $ cd MLsandbox

3. set up virtual environment within MLsandbox (if virtualenv is not yet installed, start with pip install virtualenv)
   
   $ virtualenv venv

4. start virtual environment
   
   $ source venv/bin/activate

5. install the package and its dependencies

   $ pip install .

6. start exploring notebooks!

   $ cd MLsandbox/notebooks/

   $ jupyter notebook

7. when finished, deactivate the virtual environment
 
   $ deactivate



** If virtualenv venv command gives the following error: **

	OSError: ... setuptools pip wheel failed with error code 1

Then please use a virtual environment from conda instead: [https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/](conda env)

1. $ conda create -n yourenvname python=x.x anaconda
2. $ source activate yourenvname
3. $ pip install .

