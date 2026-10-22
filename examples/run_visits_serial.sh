#!/bin/bash
#SBATCH -J corgisim_visits_serial
#SBATCH -N 1
#SBATCH -c 7                    # one PSF at a time: PROPER opens nlam = 7 processes; peak CPU usage is 7 x 100% = 700%
#SBATCH --mem=22GB
#SBATCH -t 24:00:00
#SBATCH -o corgisim_visits_serial_%j.txt
#SBATCH -e corgisim_visits_serial_err_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<EMAIL_ADDRESS>

# Sized for setup_observation() in generate_visits_example.py (same as cpgs_ref_wfov_satspots.xml):
# nlam = 7, satellite spots (4 PSF states per visit; +,0,-, science), 7 visits, at most 20 frames in one visit, 107 frames in total.
# Submit from corgisim/examples: the PROPER prescription files are copied into the working directory.

conda activate <your conda environment>

python generate_visits_example.py --option serial --seed 1 --output-dir ./output_visits_serial

