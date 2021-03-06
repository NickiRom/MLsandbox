
MLsandbox
---------

To install, do the following in commandline:

1. clone MLsandbox on your local machine:

   $ git clone https://github.com/NickiRom/MLsandbox 

2. navigate to MLsandbox/

   $ cd MLsandbox

3. set up virtual environment within MLsandbox 

   - if virtualenv is not yet installed, start with pip install virtualenv
   - if virtualenv is giving an error, see bottom of README for alternate option

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

# Gotchas:

## ** If virtualenv venv command gives the following error: **

	OSError: ... setuptools pip wheel failed with error code 1

Then please use a virtual environment from conda instead: [https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/](conda env)

1. $ conda create -n yourenvname python=x.x anaconda
2. $ source activate yourenvname
3. $ pip install .

## If you're unable to import MLsandbox:

1. check that you've opened jupyter notebook from within a virtual environment
2. check MLsandbox/venv/lib/python2.7/site-packages/ for the MLsandbox package
3. cd to MLsandbox/ and run the following command:
        $ python -m pip install --upgrade .
        
