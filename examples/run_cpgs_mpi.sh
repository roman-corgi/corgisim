#!/bin/bash
#SBATCH -J corgisim_cpgs_mpi
#SBATCH --nodes=2               # 8 ranks / 4 per node (32 cores per node)
#SBATCH --ntasks=8              # 7 workers + 1 manager
#SBATCH --ntasks-per-node=4     # 4 x 7 = 28 of the 32 cores per node
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=7       # each rank runs one PSF at a time: nlam = 7 PROPER processes
#SBATCH --mem=60G               # per node: 4 ranks x ~13 GB per detector frame 
#SBATCH -o corgisim_cpgs_mpi_%j.txt
#SBATCH -e corgisim_cpgs_mpi_err_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<EMAIL_ADDRESS>

# Sized for test/test_data/cpgs_ref_wfov_satspots.xml: nlam = 7, satellite spots
# (4 PSF states per visit; +,0,-, science), 7 visits, at most 20 frames in one visit, 107 frames in total.
#   7 worker ranks: 4 run the PSF states of a visit at once (more PSF workers would sit idle). The largest visit (20 frames) runs its detector round in three passes (7 + 7 + 6). More nodes only shorten the detector round.
#   7 CPUs per rank: the script checks each rank against nlam
# Submit from corgisim/examples: rank 0 copies the PROPER prescription files into the working directory.

# Load the MPI library that your mpi4py was built against (module names depend on your cluster)
module purge
module load <your MPI module>

conda activate <your conda environment>

# -n is the number of workers: one less than the number of ranks
mpirun -np $SLURM_NTASKS python generate_visits_from_cpgs_example.py --option mpi -n $((SLURM_NTASKS - 1)) \
    --cpgs ../test/test_data/cpgs_ref_wfov_satspots.xml --seed 1 --output-dir ./output_mpi

