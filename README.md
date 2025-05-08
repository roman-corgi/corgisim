# corgisim
A simulation suite for the Nancy Grace Roman Space Telescope Coronagraphic Instrument

## Installation Instruction

### Install Proper
Proper is an optical propagation library that is needed for CGISim to function
Go to the [Proper website](https://sourceforge.net/projects/proper-library/) and download proper_v3.3.3_python.zip
Unzip it in you working directory
Enter the directory that contains setup.py and run the following: 
```
python -m pip install .
```
### Install roman_preflight_proper and CGISIm
Go to the [CGISim website](https://sourceforge.net/projects/cgisim/) and download roman_preflight_proper_public_v2.0.1_python.zip and cgisim_v4.0.zip
Unzip them in you working directory. For each of them, enter the directory that contains setup.py and run the following: 
```
python -m pip install .
```

### Install corgidrp

Clone this directory:

```
git clone https://github.com/roman-corgi/corgidrp.git
```

Enter the directory that contains setup.py and run the following:

``` 
pip install -e .
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
