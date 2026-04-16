# Corgisim
A simulation suite for the Nancy Grace Roman Space Telescope Coronagraphic Instrument

### Documentation
The automatic documentation is available at https://corgisim.readthedocs.io/en/latest/

## Installation Instruction
### Python version
**This repository requires Python version 3.12 or higher.**

### Environnment
We strongly recommend that you use a virtual environment for your installation. You can use conda or venv, but be aware that conda is a package manager as well as an virtual environment manager. This means that you _may_ have conflicts between packages installed with pip and packages installed with conda.

#### venv
First, make sure you are not already in a environment (no environment name in parenthesis at the beginning of your command line) with `deactivate`. 

Then, do 
```
python3.12 -m venv environment_name
```
This will createn a folder called environment_name which will contain your environment. To activate it, do 

```
source environment_name/bin/activate
```
The name of your environment appears at the beginning of your command line.

#### conda
To make sure that you are not already in a environment, do `conda deactivate`. 
Then, do 
```
conda create --name environment_name
```
To activate your environment, do 
```
conda activate environment_name
```
### Install Proper
Proper is an optical propagation library that is needed for CGISim to function
Go to the [Proper website](https://sourceforge.net/projects/proper-library/) and download proper_v3.3.4_python.zip
Unzip it in your working directory
Enter the directory that contains setup.py and run the following: 
```
python -m pip install .
```
### Install roman_preflight_proper, cgi-eetc and CGISIm
In your working directory, first clone [roman_preflight_proper](https://github.com/roman-corgi/cgisim_cpp)(
please note that this is not the official version, but a modified version needed to implement certain functions):
```
git clone https://github.com/roman-corgi/cgisim_cpp.git
cd 
```
Enter the directory that contains setup.py and run the following: 
```
python -m pip install .
```
Then return to your working directory. 

Clone cgi-eetc : 
```
git clone https://github.com/nasa-jpl/cgi-eetc.git
```
Enter the directory that contains setup.py and run the following: 
```
python -m pip install .
```
Go to the [CGISim website](https://sourceforge.net/projects/cgisim/) and download cgisim 
Unzip it in your working directory. Enter the directory that contains setup.py and run the following: 
```
python -m pip install .
```

### Install corgisim

Clone this directory:  

```
git clone https://github.com/roman-corgi/corgisim.git
```

Enter the directory that contains setup.py and run the following:
```
pip install -r requirements.txt 
```
``` 
pip install -e .
```
### Check you installation

Go to corgisim/test/ and run 
```
python test_installation.py
```

### Troubleshooting
If you get the following error:
```
ImportError: Unable to run roman_preflight prescription. Stopping.
```
it means that the program cannot locate roman_preflight_compact.py and/or roman_preflight.py. The error is described in page two of roman_preflight_proper_public_v2.0.pdf.
roman_preflight_compact.py and roman_preflight.py need to be copied in the same directory as the script you are trying to run (i.e. /corgisim/test/ if you are trying to run a test, corgisim/examples/if you are trying to run an example)

### Stale branches
A branch is marked as stale after 90 days. At that point, a comment is made on the last commit and the author is notified. If nothing is done, the branch is removed after 7 more days.

### Test
test_minimal.py runs on every pull request and push to main. Make sure that this test passes before you request a review. 

Put your 'nominal' case here. The goal is to ensure that developers don't break any existing functionality by mistake. If possible, insert your test in an existing test and only simulate images if necessary. 

Longer tests and unit tests should be in another file. All other tests run on push to main (which includes when a branch is merged) and once a week. Make sure that these tests pass on your branch. 

If your PR contains only documentation (i.e. notebooks, changes to README.md, etc.), there's no need to run the tests.

