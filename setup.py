import numpy, nutils, os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('mapping/xml')
print(extra_files)

extra = {}
try:
  from setuptools import setup, Extension
except:
  from distutils.core import setup, Extension
else:
  extra['install_requires'] = [ 'numpy>=1.8', 'matplotlib>=1.3', 'scipy>=0.13' ]

long_description = """
The mapping library for Python 3
"""

setup(
  name = 'mapping',
  version = '1beta',
  description = 'Mapping',
  author = 'Jochen Hinz',
  author_email = 'j.p.hinz@tudelft.nl',
  url = 'http://google.com',
  packages = [ 'mapping' ],
  package_data={'': extra_files}, 
  long_description = long_description,
  **extra
)
