from utils import train_model, plot_training_complete, normalize_subset, create_tensors, create_tensors_subjects, fix_seeds,\
    create_data_loader, saved_normalizations,\
    available_network, network_factory_methods, available_augmentation, available_normalization, \
    normalization_factory_methods, available_paradigm
import sys
import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, Subset
import os
from sklearn.utils import compute_class_weight


def _train(data, labels, labels_subjects):
    """    
    This function compute the normalization of the training data, save the normalization in saved_path 
    and create a 5 fold cross validation and train the model. 
    """
    # Normalize Full Dataset
    mean, std, min_, max_ = normalization_factory_methods[args.normalization](data)
    
    saved_normalizations(saved_path=f'{args.saved_path}/{args.name_model}', mean=mean, std=std, min_=min_, max_=max_)
    
    fold_performance = []
    dataset = TensorDataset(data, labels, labels_subjects)
    kfold = KFold(n_splits=args.fold, shuffle=True, random_state=42)

    # Iterare su ciascun fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        fix_seeds(args.seed)
        model = (
            network_factory_methods[args.name_model](
                model_name_prefix=f'{args.saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=len(np.unique(labels)),
                samples=data.shape[3], channels=data.shape[2])
        )
        model.to(args.device)
        print(f"Fold {fold + 1}/{args.fold}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_tensor, val_tensor = normalize_subset(train_subset, val_subset, normalization_factory_methods[args.normalization])
        train_loader, val_loader = create_data_loader(train_tensor, val_tensor, args.batch_size, args.num_workers)

        y_train = torch.stack([train_subset[i][1] for i in range(len(train_subset))]).numpy()
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train), dtype=torch.float32).to(args.device)
        y_train_subjects = torch.stack([train_subset[i][2] for i in range(len(train_subset))]).numpy()
        subjects_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(y_train_subjects), y=y_train_subjects), dtype=torch.float32).to(args.device)
        print(f"Class weights for this fold: {class_weights}")
        print(f"Subjects weights for this fold: {subjects_weights}")
        train_model(model=model, fold_performance=fold_performance, train_loader=train_loader, val_loader=val_loader, 
                    fold=fold, lr=args.lr, alpha=args.alpha, class_weight=class_weights, subjects_weights=subjects_weights, 
                    epochs=args.epochs, device=args.device, augmentation=args.augmentation, patience=args.patience, checkpoint_flag=args.checkpoint_flag)

    with open(f'{args.saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt', 'w') as f:
        pass

    plot_training_complete(fold_performance,
                           f'{args.saved_path}/{args.name_model}_seed{args.seed}', args.fold)

    with open(f'{args.saved_path}/{args.name_model}_seed{args.seed}_model_params.json', 'w') as fw:
        out_params = vars(args)
        out_params['num_classes'] = len(np.unique(labels))
        out_params['samples'] = data.shape[3]
        out_params['channels'] = data.shape[2]
        json.dump(out_params, fw, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str,
                        default="/mnt/datasets/eeg/Dataset_BCI_2b/Signals_BCI_2classes/train_2b_full.npz", help="Path to train set file")
    parser.add_argument("--epochs", type=int, default=1000, help="Numbers of Epochs")
    parser.add_argument("--fold", type=int, default=5, help="Numbers of Folds")
    parser.add_argument("--patience", type=int, default=100, help="Numbers of epochs to stop the train")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batch")
    parser.add_argument("--name_model", type=str, default='PatchEmbeddingNet', help="Name of tensors that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_Prova')
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate value")
    parser.add_argument("--seed", type=int, default=42, help="Seed of initialization")
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--augmentation', type=str, choices=available_augmentation, default=None)
    parser.add_argument('--normalization', type=str, choices=available_normalization, default='Z_Score_unique')
    parser.add_argument('--paradigm', type=str, choices=available_paradigm, default='Cross')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('-checkpoint_flag', action='store_true', default=True)
    args = parser.parse_args()
    
    print(f"DEVICE: {args.device}")
    print(f"alpha: {args.alpha}")
    if args.augmentation == "None":
        args.augmentation= None
    
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    
    # create tensors
    data_train_tensors, labels_train_tensors, labels_train_subjects = create_tensors_subjects(args.train_set)
    
    if args.paradigm=='Cross':
        data = torch.cat(data_train_tensors)
        labels = torch.cat(labels_train_tensors)
        labels_subjects = torch.cat(labels_train_subjects)
        fold_performance = []
        
        _train(data, labels, labels_subjects)
        
    print("FINISHED Training")
    
    sys.exit()



