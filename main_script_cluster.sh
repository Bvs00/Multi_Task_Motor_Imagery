#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --ntasks=1
#SBATCH --account=lm_foggia
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --nodelist=gnode14
#SBATCH --job-name=ctnet_soft_segrec
#SBATCH --output=ctnet_soft_segrec.log


export TORCH_DEVICE=cuda
export PYTHON=/home/bvosmn000/.conda/envs/ICareMeEnv/bin/python

if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$SAVED_PATH" ] || [ -z "$NORM" ] || [ -z "$BANDPASS" ] || [ -z "$PARADIGM" ] || [ -z "$ALPHA" ]; then
    echo "Errore: Devi specificare NET, PRIME, AUG, SAVED_PATH, NORM, BANDPASS, PARADIGM!"
    echo "Utilizzo: NET=<valore> PRIME=<valore> AUG=<valore> ./script.sh"
    exit 1
fi

echo "$NET"
echo "$PRIME"
echo "$AUG"
echo "$SAVED_PATH"
echo "$NORM"
echo "$BANDPASS"
echo "$PARADIGM"
echo "$ALPHA"
echo "$TORCH_DEVICE"

if [ "$PRIME" == "1" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521)
elif [ "$PRIME" == "2" ]; then
  primes=(402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "3" ]; then
  primes=(42 71 101 113 127 131 139)
elif [ "$PRIME" == "4" ]; then
  primes=(149 157 163 173 181 322 521)
fi

echo "${primes[@]}"
network="$NET"
aug="$AUG"
saved_path="$SAVED_PATH"
normalization="$NORM"
bandpass="$BANDPASS"
paradigm="$PARADIGM"
alpha="$ALPHA"

for seed in "${primes[@]}"; do
  echo "Train seed: $seed"
  $PYTHON -u train_motor_imagery.py --seed "$seed" --name_model "$network" --saved_path "$saved_path" --lr 0.001 \
          --augmentation "$aug" --num_workers 32 --normalization "$normalization" --paradigm "$paradigm" \
          --train_set "/mnt/beegfs/sbove/2B/train_2b_$bandpass.npz" \
          --alpha "$alpha" --patience 150 --batch_size 72 --device "$TORCH_DEVICE"
  $PYTHON -u test_motor_imagery.py --name_model "$network" --saved_path "$saved_path" --paradigm "$paradigm" \
          --test_set "/mnt/beegfs/sbove/2B/test_2b_$bandpass.npz" \
          --seed "$seed" --alpha "$alpha" --device "$TORCH_DEVICE"
done

# $PYTHON create_excel_motor_imagery.py --network "$network" --path "$saved_path"
echo 'ok'