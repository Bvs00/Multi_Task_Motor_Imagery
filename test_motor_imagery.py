import sys
import os
from utils import create_tensors, create_tensors_subjects, find_minum_loss, validate, validate_loso, \
    load_normalizations, available_paradigm, available_network, network_factory_methods, JointCrossEntoryLoss
import argparse
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import torch
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/test_2b_full.npz')
    parser.add_argument("--name_model", type=str, default='PatchEmbeddingNet', help="Name of model that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_Prova')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paradigm', type=str, choices=available_paradigm, default='Cross')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    
    data_test_tensors, labels_test_tensors, subjects_test_tensors = create_tensors_subjects(args.test_set)
    loss_list, final_results = [], []
    loss_list_tasks, f1_list_tasks, accuracy_list_tasks, balanced_accuracy_list_tasks = [], [], [], []
    loss_list_subjects, f1_list_subjects, accuracy_list_subjects, balanced_accuracy_list_subjects = [], [], [], []
    num_subjects=9
    
    if args.paradigm=='LOSO':
        dir_path, filename = os.path.split(args.test_set)
        new_filename = filename.replace("test", "train")
        train_set_path = os.path.join(dir_path, new_filename)
        data_train_tensors, labels_train_tensors, subjects_train_tensors = create_tensors_subjects(train_set_path)
        num_subjects=8
    
    for patient in range(len(data_test_tensors)):
        data, labels, subjects = data_test_tensors[patient], labels_test_tensors[patient], subjects_test_tensors[patient]
        
        if args.paradigm=='LOSO':
            data_train, labels_train, subjects_train = data_train_tensors[patient], labels_train_tensors[patient], subjects_train_tensors[patient]
            data, labels, subjects = torch.cat([data, data_train]), torch.cat([labels, labels_train]), torch.cat([subjects, subjects_train])
        
        saved_path = args.saved_path if args.paradigm=='Cross' else f'{args.saved_path}/Patient_{patient + 1}'
        
        mean, std, min_, max_ = load_normalizations(f'{saved_path}/{args.name_model}')
        
        
        if mean != None:
            data = (data - mean)/std
        else:
            data = (data - min_)/(max_ - min_)

        dataset = TensorDataset(data, labels, subjects)
        test_loader = DataLoader(dataset, batch_size=256, num_workers=5)

        best_fold = find_minum_loss(f'{saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')

        model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=len(np.unique(labels)), subjects=num_subjects,
                samples=data.shape[3], channels=data.shape[2])
        )
        model.to(args.device)
        model.load_state_dict(torch.load(f'{saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth'))

        if args.name_model == "MSVTNet" or args.name_model == "MSVTSENet":
            criterion_tasks = JointCrossEntoryLoss()
            criterion_subjects = JointCrossEntoryLoss()
        else:
            criterion_tasks = nn.CrossEntropyLoss()
            criterion_subjects = nn.CrossEntropyLoss()
        
        if args.paradigm == 'LOSO':
            (val_f1_tasks, 
            val_accuracy_tasks, 
            val_balanced_accuracy_tasks, 
            subjects_prediction) = validate_loso(model, test_loader, criterion_tasks, criterion_subjects, alpha=args.alpha, device=args.device)
            median_subject_prediciton = np.median(subjects_prediction)
            print(f"The median value of patient {patient+1} is {median_subject_prediciton}")
            f1_list_tasks.append(val_f1_tasks), accuracy_list_tasks.append(val_accuracy_tasks), balanced_accuracy_list_tasks.append(val_balanced_accuracy_tasks)
            final_results.append({'Patient': patient+1, 
                                'F1 Score Tasks': val_f1_tasks, 'Accuracy Tasks': val_accuracy_tasks, 'Balanced Accuracy Tasks': val_balanced_accuracy_tasks, 
                                'Median Subject Prediction': median_subject_prediciton})
        else:
            (val_loss, val_loss_tasks, 
            val_loss_subjects, val_f1_tasks, 
            val_f1_subjects, val_accuracy_tasks, 
            val_accuracy_subjects, val_balanced_accuracy_tasks, 
            val_balanced_accuracy_subjects) = validate(model, test_loader, criterion_tasks, criterion_subjects, alpha=args.alpha, device=args.device)

            loss_list.append(val_loss), loss_list_tasks.append(val_loss_tasks), loss_list_subjects.append(val_loss_subjects)
            f1_list_tasks.append(val_f1_tasks), accuracy_list_tasks.append(val_accuracy_tasks), balanced_accuracy_list_tasks.append(val_balanced_accuracy_tasks)
            f1_list_subjects.append(val_f1_subjects), accuracy_list_subjects.append(val_accuracy_subjects), balanced_accuracy_list_subjects.append(val_balanced_accuracy_subjects)

            final_results.append({'Patient': patient+1, 'Loss': val_loss, 
                                'F1 Score Tasks': val_f1_tasks, 'Accuracy Tasks': val_accuracy_tasks, 'Balanced Accuracy Tasks': val_balanced_accuracy_tasks,
                                'Accuracy Subjects': val_accuracy_subjects, 'Balanced Accuracy Subjects': val_balanced_accuracy_subjects})
    
    if args.paradigm == 'LOSO':
        final_results.append(
        {f"Average": {"F1 Score Tasks": np.mean(f1_list_tasks, axis=0).tolist(), "Accuracy Tasks": np.mean(accuracy_list_tasks), 
                      "Balanced Accuracy Tasks": np.mean(balanced_accuracy_list_tasks)}})
    else:
        final_results.append(
            {f"Average": {"Loss": np.mean(loss_list), 
                        "F1 Score Tasks": np.mean(f1_list_tasks, axis=0).tolist(), "Accuracy Tasks": np.mean(accuracy_list_tasks), "Balanced Accuracy Tasks": np.mean(balanced_accuracy_list_tasks),
                        "Accuracy Subjects": np.mean(accuracy_list_subjects), "Balanced Accuracy Subjects": np.mean(balanced_accuracy_list_subjects)}}
        )

    with open(f'{args.saved_path}/Final_results_{args.name_model}_seed{args.seed}.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    print('All patient test results have been saved.')
    
    sys.exit()
