#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=a.out


module purge
module load cuda-10.0


#get tunneling info

node=$(hostname -s)

echo -e "
ssh -4 -N -f -L ${portnum}:localhost:${portnum} ct2243@${node}.cs.nyu.edu
Use a Browser on your local machine to go to:
localhost:${portnum}
"
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $portnum --notebook-dir=$(pwd)

