import os
import sys
import platform

from setuptools import find_packages, setup, Extension

copy_args = sys.argv[1:]
#copy_args.append('--user')
 
ext_modules = []

setup(
      name="corgisim",
      version = "0.0.1",
      packages=find_packages(),

      install_requires = ['numpy>=1.8', 'scipy>=0.19', 'astropy>=1.3', 'PyPROPER3>=3.3'],

      package_data = {
        '': ['*.*']
      },

      script_args = copy_args,

      zip_safe = False, 

      # Metadata for upload to PyPI
      author="Jingwen Zhang, Max Millar-Blanchaer,...",
      #author_email = "",
      description="xxxx",
      license = "BSD",
      platforms=["any"],
      url="",
      ext_modules = ext_modules
)

