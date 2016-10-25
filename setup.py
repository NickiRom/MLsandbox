
from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='MLsandbox', 
	version='0.1',
	description='tools to learn ML algorithms in a jupyter notebook environment',
	url='http://github.com/nickirom/MLsandbox',
	author='Nicole H. Romano', 
	author_email='romano.nh@gmail.com',
	packages=['MLsandbox'],
	install_requires=['pandas >= 0.19.0', 'sklearn', 'numpy', 'scipy', 'matplotlib', 'notebook', 'pydotplus', 'graphviz'],
	include_package_data=True,
	zip_safe=False)

