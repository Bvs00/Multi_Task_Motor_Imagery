#!/bin/bash

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

if [ "$PRIME" == "1" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521)
elif [ "$PRIME" == "2" ]; then
  primes=(402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "3" ]; then
  primes=(42 71 101 113 127 131 139 149 157 163 173 181 322 521 402 701 1001 1013 1207 1031 1339 1449 1527 1613 1743 1841 3222 5421)
elif [ "$PRIME" == "4" ]; then
  primes=(173 181 322 521)
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
  python -u train_motor_imagery.py --seed "$seed" --name_model "$network" --saved_path "$saved_path" --lr 0.001 \
          --augmentation "$aug" --num_workers 5 --normalization "$normalization" --paradigm "$paradigm" \
          --train_set "/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/train_2b_$bandpass.npz" \
          --alpha "$alpha" --patience 300 --batch_size 72
  python -u test_motor_imagery.py --name_model "$network" --saved_path "$saved_path" --paradigm "$paradigm" \
          --test_set "/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/test_2b_$bandpass.npz" \
          --seed "$seed" --alpha "$alpha"
done

# python create_excel_motor_imagery.py --network "$network" --path "$saved_path"
echo 'ok'