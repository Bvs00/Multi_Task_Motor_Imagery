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
    args = parser.parse_args()

    path = args.path
    f1_score_seeds = {'Patient1':[], 'Patient2':[], 'Patient3':[], 'Patient4':[], 'Patient5':[], 'Patient6':[],
                                    'Patient7':[], 'Patient8':[], 'Patient9':[], 'Average':[]}
    dataframe_balanced = pd.DataFrame(columns=['Seed', 'Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient5', 'Patient6',
                                    'Patient7', 'Patient8', 'Patient9', 'Average'])
    dataframe_balanced_subjects = pd.DataFrame(columns=['Seed', 'Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient5', 'Patient6',
                                    'Patient7', 'Patient8', 'Patient9', 'Average'])
    list_seeds = [42, 71, 101, 113, 127, 131, 139, 149, 157, 163, 173, 181, 322, 521, 
                  402, 701, 1001, 1013, 1207, 1031, 1339, 1449, 1527, 1613, 1743, 
                  1841, 3222, 5421] if args.full_seeds else [42, 71, 101, 113, 127, 131, 
                    139, 149, 157, 163, 173, 181, 322, 521]

    for count, seed in enumerate(list_seeds):
        
        file_path = f"{path}/Final_results_{args.network}_seed{seed}.json"

        with open(file_path, 'r') as f:
            data = json.load(f)
            balanced_accuracies = [round(data[i]['Balanced Accuracy Tasks'], 2) for i in range(9)]
            if not args.fine_tuning:
                balanced_accuracies_subjects = [round(data[i]['Balanced Accuracy Subjects'], 2) for i in range(9)]
            
            for i in range(9):
                key = f'Patient{i+1}'
                f1_val_arr = [round(x, 3) for x in data[i]['F1 Score Tasks']]
                if f1_score_seeds[key]:
                    f1_score_seeds[key] = [a+b for a,b in zip(f1_score_seeds[key], f1_val_arr)]
                else:
                    f1_score_seeds[key] = f1_val_arr
            f1_val_arr = [round(x, 3) for x in data[9]['Average']['F1 Score Tasks']]
            if f1_score_seeds['Average']:
                f1_score_seeds['Average'] = [a+b for a,b in zip(f1_score_seeds['Average'], f1_val_arr)]
            else:
                f1_score_seeds['Average'] = f1_val_arr
            
            # Aggiungere una nuova riga al DataFrame
            dataframe_balanced.loc[count] = [seed] + balanced_accuracies + [round(np.mean(balanced_accuracies),2)]
            if not args.fine_tuning:
                dataframe_balanced_subjects.loc[count] = [seed] + balanced_accuracies_subjects + [round(np.mean(balanced_accuracies_subjects),2)]
        dataframe_balanced['Seed'] = dataframe_balanced['Seed'].astype('int32')
        if not args.fine_tuning:
            dataframe_balanced_subjects['Seed'] = dataframe_balanced_subjects['Seed'].astype('int32')
    
    tmp_acc_list, tmp_bal_acc_list, tmp_bal_acc_sub_list = [], [], []
    for i in range(9):
        key = f'Patient{i+1}'
        f1_score_seeds[key] = [round(x / len(list_seeds), 3) for x in f1_score_seeds[key]]
        tmp_bal_acc_list.append(round(dataframe_balanced[key].mean(), 2))
        if not args.fine_tuning:
            tmp_bal_acc_sub_list.append(round(dataframe_balanced_subjects[key].mean(), 2))
    
    f1_score_seeds['Average'] = [round(x / len(list_seeds), 3) for x in f1_score_seeds['Average']]
    dataframe_balanced.loc[len(list_seeds)] = ['Average'] + tmp_bal_acc_list + [round(np.mean(tmp_bal_acc_list),2)]
    if not args.fine_tuning:
        dataframe_balanced_subjects.loc[len(list_seeds)] = ['Average'] + tmp_bal_acc_sub_list + [round(np.mean(tmp_bal_acc_sub_list),2)]
    
    # Salvare il DataFrame in un file CSV
    dataframe_balanced.to_excel(f'{path}/seed_results_{args.network}_balanced.xlsx', index=False)
    if not args.fine_tuning:
        dataframe_balanced_subjects.to_excel(f'{path}/seed_results_{args.network}_balanced_subjects.xlsx', index=False)
    with open(f"{path}/seed_results_{args.network}_f1_score.json", 'w') as f:
        json.dump(f1_score_seeds, f, indent=1)
