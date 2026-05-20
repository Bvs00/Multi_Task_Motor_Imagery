#!/bin/bash

if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$NORM" ] || [ -z "$BANDPASS" ] \
    || [ -z "$PARADIGM" ] || [ -z "$ALPHA" ] || [ -z "$PROBING" ] || [ -z "$AUX" ] || [ -z "$DATASET" ]; then
    echo "Errore: Devi specificare NET, PRIME, AUG, SAVED_PATH, NORM, BANDPASS, PARADIGM!"
    echo "Utilizzo: NET=<valore> PRIME=<valore> AUG=<valore> ./script.sh"
    exit 1
fi

network="$NET"
paradigm="$PARADIGM"
alpha_values=($ALPHA)
aux="$AUX"
dataset="$DATASET"


for alpha in "${alpha_values[@]}"; do
    case "$alpha" in
        0.001)
        alpha_str="0001"
        ;;
        0.01)
        alpha_str="001"
        ;;
        0.10)
        alpha_str="010"
        ;;
        0.25)
        alpha_str="025"
        ;;
        0.5)
        alpha_str="05"
        ;;
        0.75)
        alpha_str="075"
        ;;
        0.9)
        alpha_str="09"
        ;;
    esac
    echo $alpha_str

    case "$aux" in 
        "True")
        name_model="Results_${network}"
        ;;
        "False")
        name_model="Results_${network}_Wout_Aux"
        ;;
    esac
    echo $name_model

    saved_path="Results_${dataset}/Results_Alpha${alpha_str}/Results_SegRec/Results_FineTuning/${name_model}"
    echo $saved_path
    saved_path_loso="Results_${dataset}/Results_Alpha${alpha_str}/Results_SegRec/Results_Cross/${name_model}"
    echo $saved_path_loso

    export NET="$network"
    export PRIME="$PRIME"
    export AUG="$AUG"
    export SAVED_PATH_LOSO="$saved_path_loso"
    export SAVED_PATH="$saved_path"
    export NORM="$NORM"
    export BANDPASS="$BANDPASS"
    export PARADIGM="$PARADIGM"
    export PROBING="$PROBING"
    export AUX="$AUX"
    export DATASET="$DATASET"

    ./main_script_finetuning_inbit.sh
done
