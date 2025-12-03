#!/bin/bash

if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$NUM_AUG" ] || [ -z "$SAVED_PATH" ] || [ -z "$NORM" ] || [ -z "$BANDPASS" ] \
    || [ -z "$PARADIGM" ] || [ -z "$ALPHA" ] || [ -z "$AUX" ] || [ -z "$DATASET" ] || [ -z "$MAPS" ] || [ -z "$P1" ] || [ -z "$P2" ]; then
    echo "Errore: Devi specificare NET, PRIME, AUG, SAVED_PATH, NORM, BANDPASS, PARADIGM!"
    echo "Utilizzo: NET=<valore> PRIME=<valore> AUG=<valore> ./script.sh"
    exit 1
fi
# MAPS="9 9 9 9"
# P1=8
# P2=7
echo "SONO IN INBIT SCRIPT"
echo "$NET"
echo "$PRIME"
echo "$AUG"
echo "$NUM_AUG"
echo "$SAVED_PATH"
echo "$NORM"
echo "$BANDPASS"
echo "$PARADIGM"
echo "$ALPHA"
echo "$AUX"
echo "$DATASET"
echo "$MAPS"
echo "$P1"
echo "$P2"

if [ "$PRIME" == "1" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521)
elif [ "$PRIME" == "2" ]; then
  primes=(402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "3" ]; then
  primes=(42 71 101 113 127 131 139)
elif [ "$PRIME" == "4" ]; then
  primes=(149 157 163 173 181 322 521)
elif [ "$PRIME" == "5" ]; then
  primes=(42)
fi

echo "${primes[@]}"
network="$NET"
aug="$AUG"
num_aug="$NUM_AUG"
saved_path="$SAVED_PATH"
normalization="$NORM"
bandpass="$BANDPASS"
paradigm="$PARADIGM"
alpha="$ALPHA"
aux="$AUX"
dataset="$DATASET"
maps="$MAPS"
p1="$P1"
p2="$P2"

for seed in "${primes[@]}"; do
  echo "Train seed: $seed"
  python -u train_motor_imagery.py --seed "$seed" --name_model "$network" --saved_path "$saved_path" --lr 0.001 \
          --augmentation "$aug" --num_augmentations "$num_aug" --num_workers 5 --normalization "$normalization" \
          --paradigm "$paradigm" --train_set "/home/inbit/Scrivania/Datasets/${dataset}/Signals_BCI_2classes/train_${dataset}_$bandpass.npz" \
          --alpha "$alpha" --patience 150 --batch_size 72 --auxiliary_branch "$aux" --feature_maps $maps \
          --p1 "$p1" --p2 "$p2"
  echo "Test seed: $seed"
  python -u test_motor_imagery.py --name_model "$network" --saved_path "$saved_path" --paradigm "$paradigm" \
          --test_set "/home/inbit/Scrivania/Datasets/${dataset}/Signals_BCI_2classes/test_${dataset}_$bandpass.npz" \
          --seed "$seed" --alpha "$alpha" --auxiliary_branch "$aux" --feature_maps $maps \
          --p1 "$p1" --p2 "$p2"
done

python create_excel_motor_imagery.py --network "$network" --path "$saved_path"
echo 'ok'