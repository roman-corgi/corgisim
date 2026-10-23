#!/bin/bash
#SBATCH -J corgisim_cpgs_mp
#SBATCH -N 1
#SBATCH -c 28                   # PSF round: 4 PSF states at once x nlam = 7 PROPER processes = 28 CPUs
#SBATCH --mem=100G              # 6 detector frames at once x ~13 GB + manager and PSF round margin
#SBATCH -t 24:00:00
#SBATCH -o corgisim_cpgs_mp_%j.txt
#SBATCH -e corgisim_cpgs_mp_err_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<EMAIL_ADDRESS>

# Sized for test/test_data/cpgs_ref_wfov_satspots.xml: nlam = 7, satellite spots
# (4 PSF states per visit; +,0,-, science), 7 visits, at most 20 frames in one visit, 107 frames in total.
#   PSF round:      min(n, 4) x 7 = 28 PROPER processes -> -c 28
#   detector round: one CPU per worker, but each frame briefly needs ~13 GB: emccd_detect partial_CIC
#   -n 6: 6 x 13 GB = 78 GB fits in --mem=100G. With -n 20, 9 frames at once ran out of memory, a
#   worker was killed and the pool hung. More workers need ~13 GB more --mem each.
# Submit from corgisim/examples: the PROPER prescription files are copied into the working directory.


conda activate <your conda environment>

python generate_visits_from_cpgs_example.py --option mp -n 6 \
    --cpgs ../test/test_data/cpgs_ref_wfov_satspots.xml --seed 1 --output-dir ./output_mp

