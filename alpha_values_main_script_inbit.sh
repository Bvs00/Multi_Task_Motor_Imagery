#!/bin/bash

if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$NUM_AUG" ] || [ -z "$NORM" ] || [ -z "$BANDPASS" ] \
    || [ -z "$PARADIGM" ] || [ -z "$ALPHA" ] || [ -z "$AUX" ] || [ -z "$DATASET" ] \
    || [ -z "$MAPS" ] || [ -z "$P1" ] || [ -z "$P2" ]; then
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

    saved_path="Results_${dataset}/Results_Alpha${alpha_str}/Results_SegRec/Results_${paradigm}/${name_model}"
    echo $saved_path

    export NET="$network"
    export PRIME="$PRIME"
    export AUG="$AUG"
    export NUM_AUG="$NUM_AUG"
    export SAVED_PATH="$saved_path"
    export NORM="$NORM"
    export BANDPASS="$BANDPASS"
    export PARADIGM="$PARADIGM"
    export ALPHA="$alpha"
    export AUX="$AUX"
    export DATASET="$DATASET"
    export MAPS="$MAPS"
    export P1="$P1"
    export P2="$P2"

    ./main_script_inbit.sh
done

# #!/bin/bash

# # Check required variables
# if [ -z "$NET" ] || [ -z "$PRIME" ] || [ -z "$AUG" ] || [ -z "$NUM_AUG" ] \
#    || [ -z "$NORM" ] || [ -z "$BANDPASS" ] || [ -z "$PARADIGM" ] \
#    || [ -z "$ALPHA" ] || [ -z "$AUX" ] || [ -z "$DATASET" ] \
#    || [ -z "$MAPS" ] || [ -z "$P1" ] || [ -z "$P2" ]; then
#     echo "Errore: Devi specificare NET, PRIME, AUG, SAVED_PATH, NORM, BANDPASS, PARADIGM!"
#     exit 1
# fi

# network="$NET"
# paradigm="$PARADIGM"
# aux="$AUX"
# dataset="$DATASET"

# # Make ALPHA iterable
# alpha_values=($ALPHA)

# for alpha in "${alpha_values[@]}"; do

#     case "$alpha" in
#         0.001) alpha_str="0001" ;;
#         0.01)  alpha_str="001"  ;;
#         0.10)  alpha_str="010"  ;;
#         0.25)  alpha_str="025"  ;;
#         0.5)   alpha_str="05"   ;;
#         0.75)  alpha_str="075"  ;;
#         0.9)   alpha_str="09"   ;;
#         *)
#             echo "Valore ALPHA non gestito: $alpha"
#             exit 1
#             ;;
#     esac

#     echo "$alpha_str"

#     case "$aux" in
#         "True")
#             name_model="Results_${network}"
#             ;;
#         "False")
#             name_model="Results_${network}_Wout_Aux"
#             ;;
#         *)
#             echo "AUX deve essere True o False"
#             exit 1
#             ;;
#     esac

#     echo "$name_model"

#     saved_path="Results_${dataset}/Results_Alpha${alpha_str}/Results_SegRec/Results_${paradigm}/${name_model}"
#     echo "$saved_path"

#     # Override only the variables that change:
#     ALPHA="$alpha"
#     SAVED_PATH="$saved_path"

#     ./main_script_inbit.sh
# done

