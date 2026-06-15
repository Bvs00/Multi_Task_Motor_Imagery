import argparse
import json
import pandas as pd
import numpy as np


# CSV Test Set
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='PatchEmbeddingNet_Autoencoder')
    parser.add_argument('--path', type=str, default='Results_400/Results_Percentile_unique/Results_PatchEmbeddingNet_Autoencoder')
    parser.add_argument('-full_seeds', action='store_true')
    parser.add_argument('-fine_tuning', action='store_true')
    parser.add_argument('-loso', action='store_true')
    args = parser.parse_args()

    path = args.path
    if 'OpenBMI' in path:
        num_patient = 54
    elif 'PhysionetMI' in path:
        num_patient=106
    else:
        num_patient = 9
    f1_score_seeds = {}
    for i in range(num_patient):
        f1_score_seeds[f'Patient{i+1}'] = []
    f1_score_seeds['Average'] = []
    
    columns = ['Seed'] + [f'Patient{i+1}' for i in range(num_patient)] + ['Average']
    dataframe = pd.DataFrame(columns=columns)
    dataframe_balanced = pd.DataFrame(columns=columns)
    dataframe_balanced_subjects = pd.DataFrame(columns=columns)
    dataframe_kappa = pd.DataFrame(columns=columns)
    list_seeds = [42, 71, 101, 113, 127, 131, 139, 149, 157, 163, 173, 181, 322, 521, 
                  402, 701, 1001, 1013, 1207, 1031, 1339, 1449, 1527, 1613, 1743, 
                  1841, 3222, 5421] if args.full_seeds else [42, 71, 101, 113, 127, 131, 
                    139, 149, 157, 163, 173, 181, 322, 521]

    for count, seed in enumerate(list_seeds):
        
        file_path = f"{path}/Final_results_{args.network}_seed{seed}.json"

        with open(file_path, 'r') as f:
            data = json.load(f)
            balanced_accuracies = [round(data[i]['Balanced Accuracy Tasks'], 4)*100 for i in range(num_patient)]
            kappa = [round(data[i]['Kappa Tasks'], 3) for i in range(num_patient)]
            if not args.fine_tuning and not args.loso:
                balanced_accuracies_subjects = [round(data[i]['Balanced Accuracy Subjects'], 4)*100 for i in range(num_patient)]
            
            for i in range(num_patient):
                key = f'Patient{i+1}'
                f1_val_arr = [round(x, 4) for x in data[i]['F1 Score Tasks']]
                if f1_score_seeds[key]:
                    f1_score_seeds[key] = [a+b for a,b in zip(f1_score_seeds[key], f1_val_arr)]
                else:
                    f1_score_seeds[key] = f1_val_arr
            f1_val_arr = [round(x, 4) for x in data[num_patient]['Average']['F1 Score Tasks']]
            if f1_score_seeds['Average']:
                f1_score_seeds['Average'] = [a+b for a,b in zip(f1_score_seeds['Average'], f1_val_arr)]
            else:
                f1_score_seeds['Average'] = f1_val_arr
            
            # Aggiungere una nuova riga al DataFrame
            dataframe_balanced.loc[count] = [seed] + balanced_accuracies + [round(np.mean(balanced_accuracies), 2)]
            if not args.fine_tuning and not args.loso:
                dataframe_balanced_subjects.loc[count] = [seed] + balanced_accuracies_subjects + [round(np.mean(balanced_accuracies_subjects), 2)]
                dataframe_kappa.loc[count] = [seed] + kappa + [np.mean(kappa)]
        dataframe_balanced['Seed'] = dataframe_balanced['Seed'].astype('int32')
        dataframe_kappa['Seed'] = dataframe_kappa['Seed'].astype('int32')
        if not args.fine_tuning and not args.loso:
            dataframe_balanced_subjects['Seed'] = dataframe_balanced_subjects['Seed'].astype('int32')
    
    tmp_acc_list, tmp_bal_acc_list, tmp_bal_acc_sub_list, tmp_kappa_list = [], [], [], []
    for i in range(num_patient):
        key = f'Patient{i+1}'
        f1_score_seeds[key] = [round(x / len(list_seeds), 4)*100 for x in f1_score_seeds[key]]
        tmp_bal_acc_list.append(round(dataframe_balanced[key].mean(), 2))
        tmp_kappa_list.append(round(dataframe_kappa[key].mean(), 2))
        if not args.fine_tuning and not args.loso:
            tmp_bal_acc_sub_list.append(round(dataframe_balanced_subjects[key].mean(), 2))
    
    f1_score_seeds['Average'] = [round(x / len(list_seeds), 4)*100 for x in f1_score_seeds['Average']]
    dataframe_balanced.loc[len(list_seeds)] = ['Average'] + tmp_bal_acc_list + [round(np.mean(tmp_bal_acc_list), 2)]
    dataframe_kappa.loc[len(list_seeds)] = ['Average'] + tmp_kappa_list + [np.mean(tmp_kappa_list)]
    if not args.fine_tuning and not args.loso:
        dataframe_balanced_subjects.loc[len(list_seeds)] = ['Average'] + tmp_bal_acc_sub_list + [round(np.mean(tmp_bal_acc_sub_list), 4)]
    
    # Salvare il DataFrame in un file CSV
    dataframe_balanced.to_excel(f'{path}/A_seed_results_{args.network}_balanced.xlsx', index=False)
    dataframe_kappa.to_excel(f'{path}/A_seed_results_{args.network}_kappa.xlsx', index=False)
    if not args.fine_tuning and not args.loso:
        dataframe_balanced_subjects.to_excel(f'{path}/A_seed_results_{args.network}_balanced_subjects.xlsx', index=False)
    with open(f"{path}/A_seed_results_{args.network}_f1_score.json", 'w') as f:
        json.dump(f1_score_seeds, f, indent=4)
