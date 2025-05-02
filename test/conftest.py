from corgisim import scene
from corgisim import instrument
import proper
import roman_preflight_proper
import pytest
import shutil
import os

def pytest_sessionstart(session):
    """
    setting up the prescription files before running pytestfiles
    """    

    # Copy the prescription files
    path_directory = os.path.dirname(os.path.abspath(__file__))

    if not (os.path.isfile( path_directory + '/roman_preflight.py')):
        prescription_file = roman_preflight_proper.lib_dir + '/roman_preflight.py'
        shutil.copy( prescription_file, path_directory )

    if not (os.path.isfile(path_directory + '/roman_preflight_compact.py')):
        prescription_file = roman_preflight_proper.lib_dir + '/roman_preflight_compact.py'
        shutil.copy( prescription_file, path_directory )
