import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, balanced_accuracy_score
############ Import Network ############
from EEGNet import EEGNet
from PatchEmbeddingNet import PatchEmbeddingNet, PatchEmbeddingNet_Autoencoder, PatchEmbeddingNet_Soft 
from MSVTNet import MSVTNet, MSVTSENet, MSVT_SE_Net, MSVT_SE_SE_Net
from CTNet import CTNet, CTNet_Soft, CSETNet, CCBAMTNet
################################################
import seaborn as sns
from data_augmentation import chr_augmentation, reverse_channels, segmentation_reconstruction, reverse_channels_segmentation_reconstruction


available_network = [
    'LMDANet',
    'EEGNet',
    'EEGConformer',
    'EEGConformerPositional',
    'EEGEncoder',
    'EEGNetDilated',
    'CKRLNet',
    'SSCL_CSD',
    'EEGNetConformer', 
    'EEGConformer_Wout_Attention', 
    'CTNet',
    'PatchEmbeddingNet',
    'PatchEmbeddingNet_Autoencoder',
    'PatchEmbeddingNet_Soft',
    'CTNet_Soft',
    'MSVTNet',
    'CSETNet',
    'CCBAMTNet',
    'MSVTSENet',
    'MSVT_SE_Net', 
    'MSVT_SE_SE_Net'
]

network_factory_methods = {
    'PatchEmbeddingNet': PatchEmbeddingNet, 
    'PatchEmbeddingNet_Autoencoder': PatchEmbeddingNet_Autoencoder,
    'CTNet': CTNet,
    'EEGNet': EEGNet,
    'PatchEmbeddingNet_Soft': PatchEmbeddingNet_Soft,
    'CTNet_Soft': CTNet_Soft,
    'MSVTNet': MSVTNet,
    'CSETNet': CSETNet,
    'CCBAMTNet': CCBAMTNet,
    'MSVTSENet': MSVTSENet,
    'MSVT_SE_Net': MSVT_SE_Net,
    'MSVT_SE_SE_Net': MSVT_SE_SE_Net
}

available_augmentation = [
    'None',
    'chr_augmentation',
    'reverse_channels',
    'segmentation_reconstruction',
    'reverse_segmentation'
]

augmentation_factory_methods = {
    'chr_augmentation': chr_augmentation,
    'reverse_channels': reverse_channels,
    'segmentation_reconstruction': segmentation_reconstruction,
    'reverse_segmentation': reverse_channels_segmentation_reconstruction
}

available_paradigm = [
    'Single', 
    'Cross',
    'LOSO'
]

################################## NORMALIZATIONs #############################################################
def _z_score_normalization(data, unique=False):
    if unique:
        mean = data.mean(dim=None, keepdim=True)
        std = data.std(dim=None, keepdim=True)
    else:
        mean = data.mean(dim=(0, 3), keepdim=True)
        std = data.std(dim=(0, 3), keepdim=True)
    return mean, std

def _min_max_normalization(data, unique=False):
    if unique:
        min_ = torch.amin(data, dim=None, keepdim=True)
        max_ = torch.amax(data, dim=None, keepdim=True)
    else:
        min_ = torch.amin(data, dim=(0, 3), keepdim=True)
        max_ = torch.amax(data, dim=(0, 3), keepdim=True)
    return min_, max_

def _percentile_normalization(data, unique=False):
    if unique:
        min_ = torch.quantile(data, q=0.05, dim=None, keepdim=True, interpolation='linear')
        max_ = torch.quantile(data, q=0.95, dim=None, keepdim=True, interpolation='linear')
    else:
        min_ = torch.quantile(data, q=0.05, dim=(0, 3), keepdim=True, interpolation='linear')
        max_ = torch.quantile(data, q=0.95, dim=(0, 3), keepdim=True, interpolation='linear')
    return min_, max_

def saved_normalizations(saved_path, mean=None, std=None, min_=None, max_=None):
    if mean != None:
        torch.save(mean, f'{saved_path}_mean_data.pt')
    if std != None:
        torch.save(std, f'{saved_path}_std_data.pt')
    if min_ != None:
        torch.save(min_, f'{saved_path}_min_data.pt')
    if max_ != None:
        torch.save(max_, f'{saved_path}_max_data.pt')

def load_normalizations(saved_path):
    mean, std, min_, max_ = None, None, None, None
    if os.path.exists(f'{saved_path}_mean_data.pt'):
        mean = torch.load(f'{saved_path}_mean_data.pt')
    if os.path.exists(f'{saved_path}_std_data.pt'):
        std = torch.load(f'{saved_path}_std_data.pt')
    if os.path.exists(f'{saved_path}_min_data.pt'):
        min_ = torch.load(f'{saved_path}_min_data.pt')
    if os.path.exists(f'{saved_path}_max_data.pt'):
        max_ = torch.load(f'{saved_path}_max_data.pt')
    return mean, std, min_, max_

def normalization_z_score_channels(data):
    mean, std = _z_score_normalization(data)
    return mean, std, None, None

def normalization_z_score_unique(data):
    mean, std = _z_score_normalization(data, True)
    return mean, std, None, None

def normalization_min_max_channels(data):
    min_, max_ = _min_max_normalization(data)
    return None, None, min_, max_
    
def normalization_min_max_unique(data):
    min_, max_ = _min_max_normalization(data, True)
    return None, None, min_, max_

def normalization_percentile_channels(data):
    min_, max_ = _percentile_normalization(data)
    return None, None, min_, max_
    
def normalization_percentile_unique(data):
    min_, max_ = _percentile_normalization(data, True)
    return None, None, min_, max_

def data_labels_from_subset(subset):
    data = torch.stack([subset[i][0] for i in range(len(subset))])
    labels = torch.stack([subset[i][1] for i in range(len(subset))])
    labels_subjects = torch.stack([subset[i][2] for i in range(len(subset))])
    return data, labels, labels_subjects
        
def normalize_subset(train_subset, val_subset, normalization_function):
    train_data, train_labels, train_labels_subjects = data_labels_from_subset(train_subset)
    val_data, val_labels, val_labels_subjects = data_labels_from_subset(val_subset)

    mean, std, min_, max_ = normalization_function(train_data)
    
    if mean != None:
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
    else:
        train_data = (train_data - min_)/(max_-min_)
        val_data = (val_data - min_)/(max_-min_)

    train_tensor = TensorDataset(train_data, train_labels, train_labels_subjects)
    val_tensor = TensorDataset(val_data, val_labels, val_labels_subjects)

    return train_tensor, val_tensor

def data_labels_from_subset_autoencoder(subset):
    data = torch.stack([subset[i][0] for i in range(len(subset))])
    labels = torch.stack([subset[i][1] for i in range(len(subset))])
    return data, labels
        
def normalize_subset_autoencoder(train_subset, val_subset, normalization_function):
    train_data, train_labels = data_labels_from_subset_autoencoder(train_subset)
    val_data, val_labels = data_labels_from_subset_autoencoder(val_subset)

    mean, std, min_, max_ = normalization_function(train_data)
    
    if mean != None:
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
    else:
        train_data = (train_data - min_)/(max_-min_)
        val_data = (val_data - min_)/(max_-min_)

    train_tensor = TensorDataset(train_data, train_labels)
    val_tensor = TensorDataset(val_data, val_labels)

    return train_tensor, val_tensor

normalization_factory_methods = {
    'Z_Score_channels': normalization_z_score_channels,
    'Z_Score_unique': normalization_z_score_unique,
    'Min_Max_channels': normalization_min_max_channels,
    'Min_Max_unique': normalization_min_max_unique,
    'Percentile_channels': normalization_percentile_channels,
    'Percentile_unique': normalization_percentile_unique
}

available_normalization = [
    'Z_Score_channels',
    'Z_Score_unique',
    'Min_Max_channels',
    'Min_Max_unique',
    'Percentile_channels',
    'Percentile_unique',
    'None'
]

########################## CREATE TENSORS ###########################
def create_tensors(dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    data_tensor = []
    data_list = dataset['data']
    for data in data_list:
        data_tensor.append(torch.tensor(data).float().unsqueeze(1))
    labels_tensor = []
    labels_list = dataset['labels']
    for labels in labels_list:
        labels_tensor.append(torch.tensor(labels))

    return data_tensor, labels_tensor

def create_tensors_subjects(dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    data_tensor = []
    data_list = dataset['data']
    for data in data_list:
        data_tensor.append(torch.tensor(data).float().unsqueeze(1))
    labels_tensor = []
    labels_subjects = []
    labels_list = dataset['labels']
    for subject, labels in enumerate(labels_list):
        labels_tensor.append(torch.tensor(labels))
        labels_subjects.append(torch.full_like(labels_tensor[subject], subject))
    
    return data_tensor, labels_tensor, labels_subjects


######################## FIX SEED #################################
def fix_seeds(seed=42):
    random.seed(seed)  # Per il modulo random standard di Python
    np.random.seed(seed)  # Per NumPy
    torch.manual_seed(seed)  # Per PyTorch su CPU
    torch.cuda.manual_seed(seed)  # Per PyTorch su una singola GPU
    torch.cuda.manual_seed_all(seed)  # Per PyTorch su più GPU

    # Imposta il determinismo nei backend di PyTorch
    torch.backends.cudnn.deterministic = True  # Per convoluzioni deterministiche
    torch.backends.cudnn.benchmark = False  # Disattiva le ottimizzazioni basate su benchmark

################################################  PLOT ##############################################################
def plot_losses(list_loss_train, list_loss_train_tasks, list_loss_train_subjects, 
                list_loss_validation, list_loss_validation_tasks, list_loss_validation_subjects, fold=0, path='Results/', reconstruction=False):
    
    min_index = list_loss_validation.index(min(list_loss_validation))
    min_value = list_loss_validation[min_index]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(len(list_loss_train)), list_loss_train, marker='o', label='Train Loss')
    plt.plot(range(len(list_loss_train_tasks)), list_loss_train_tasks, marker='o', label='Train Loss Tasks')
    plt.plot(range(len(list_loss_train_subjects)), list_loss_train_subjects, marker='o', label='Train Loss Reconstruction' if reconstruction else 'Train Loss Subjects')
    
    plt.plot(range(len(list_loss_validation)), list_loss_validation, marker='o', label='Validation Loss')
    plt.plot(range(len(list_loss_validation_tasks)), list_loss_validation_tasks, marker='o', label='Validation Loss Tasks')
    plt.plot(range(len(list_loss_validation_subjects)), list_loss_validation_subjects, marker='o', label='Validation Loss Reconstruction' if reconstruction else 'Validation Loss Subjects')
    
    plt.plot(min_index, min_value, 'gs', label='Minimum Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Train and Validation Loss per Epoch in fold {fold}')
    plt.legend()
    plt.savefig(f"./{path}_plot_losses_fold{fold}")
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, fold, name, accuracy):
    plt.figure(figsize=(10, 10))
    confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1, keepdims=True)
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 42}, cbar=False)
    plt.ylabel('True label', fontsize=32, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=32, fontweight='bold')
    plt.xticks(fontsize=32)  # modifica il valore come preferisci
    plt.yticks(fontsize=32)
    # plt.title(f'Confusion Matrix fold: {fold} - Accuracy: {accuracy:.2f}')
    plt.title(f'Balanced Accuracy: {accuracy:.2f}', fontsize=32)
    plt.savefig(f'./{name}_confusion_matrix_{fold}.png')
    plt.close()

def plot_training_complete(fold_performance, name, folds):
    print('Training complete')
    for fold, performance in enumerate(fold_performance):
        # print(f'Fold {fold + 1} - Loss: {performance[0]:.4f}, F1: {performance[1]}, Confusion Matrix: '
        #       f'{performance[2]}, Accuracy: {performance[3]:.4f}, Balanced Accuracy: {performance[4]:.4f}\n')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Fold {fold + 1} - Loss: {performance[0]:.3f}, Loss Tasks: {performance[1]:.3f}, Loss Subjects: {performance[2]:.3f}, '
                    f'F1 Tasks: {performance[3]}, F1 Subjects: {performance[4]}, '
                    f'Accuracy Tasks: {performance[5]:.3f}, Accuracy Subjects: {performance[6]:.3f}, '
                    f'Balanced Accuracy Tasks: {performance[7]:.3f}, Balanced Accuracy Subjects: {performance[8]:.3f}\n')
    avg_loss = np.mean([performance[0] for performance in fold_performance])
    avg_f1 = np.mean([performance[3] for performance in fold_performance])
    if folds > 1:
        accuracy = 0.0
        for performance in fold_performance:
            accuracy += performance[7]
        avg_accuracy = accuracy / folds
        print(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}, Average Accuracy: {avg_accuracy:.3f}\n')
            
def plot_training_complete_loso(fold_performance, name, folds):
    print('Training complete')
    for fold, performance in enumerate(fold_performance):
        # print(f'Fold {fold + 1} - Loss: {performance[0]:.4f}, F1: {performance[1]}, Confusion Matrix: '
        #       f'{performance[2]}, Accuracy: {performance[3]:.4f}, Balanced Accuracy: {performance[4]:.4f}\n')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Fold {fold + 1} - Loss: {performance[0]:.3f}, F1 Tasks: {performance[1]}, '
                    f'Accuracy Tasks: {performance[3]:.3f}, '
                    f'Balanced Accuracy Tasks: {performance[4]:.3f}\n')
    avg_loss = np.mean([performance[0] for performance in fold_performance])
    avg_f1 = np.mean([performance[3] for performance in fold_performance])
    if folds > 1:
        accuracy = 0.0
        for performance in fold_performance:
            accuracy += performance[4]
        avg_accuracy = accuracy / folds
        print(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}, Average Accuracy: {avg_accuracy:.3f}\n')

def plot_training_complete_autoencoder(fold_performance, name, folds):
    print('Training complete')
    for fold, performance in enumerate(fold_performance):
        # print(f'Fold {fold + 1} - Loss: {performance[0]:.4f}, F1: {performance[1]}, Confusion Matrix: '
        #       f'{performance[2]}, Accuracy: {performance[3]:.4f}, Balanced Accuracy: {performance[4]:.4f}\n')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Fold {fold + 1} - Loss: {performance[0]:.3f}, Loss Tasks: {performance[1]:.3f}, Loss Reconstruction: {performance[2]:.3f}, '
                    f'F1 Tasks: {performance[3]}, '
                    f'Accuracy Tasks: {performance[4]:.3f}, '
                    f'Balanced Accuracy Tasks: {performance[5]:.3f}\n')
    avg_loss = np.mean([performance[0] for performance in fold_performance])
    avg_f1 = np.mean([performance[3] for performance in fold_performance])
    if folds > 1:
        accuracy = 0.0
        for performance in fold_performance:
            accuracy += performance[5]
        avg_accuracy = accuracy / folds
        print(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}')
        with open(f'{name}_validation_log.txt', 'a') as f:
            f.write(f'Average Loss: {avg_loss:.3f}, Average F1: {avg_f1}, Average Accuracy: {avg_accuracy:.3f}\n')

########################################## CREATE DATALOADER #############################################
def create_data_loader(train_tensor, val_tensor, batch_size, num_workers):
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              prefetch_factor=3, persistent_workers=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            prefetch_factor=3, persistent_workers=True)
    return train_loader, val_loader

#################################### FIND BEST FOLD ############################################
def find_minum_loss(filename):
    # Apri il file e leggi le linee
    with open(filename, "r") as file:
        lines = file.readlines()
    
    # Inizializza variabili per tracciare il fold con la loss più bassa
    min_loss = float('inf')
    best_fold = None

    # Scorri le linee e cerca i valori di loss
    for line in lines:
        if "Loss:" in line and "Average" not in line:
            # Estrai il numero del fold e la loss
            parts = line.split()
            fold_num = int(parts[1])
            loss_value = float(parts[4].strip(','))

            # Aggiorna il minimo se la loss corrente è inferiore a quella minima trovata
            if loss_value < min_loss:
                min_loss = loss_value
                best_fold = fold_num

    # Stampa il risultato
    return best_fold

def find_max_f1(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

        # Inizializza variabili per tracciare il fold con la loss più bassa
    max_f1 = 0.0
    best_fold = None

    # Scorri le linee e cerca i valori di loss
    for line in lines:
        if "Loss:" in line and "Average" not in line:
            # Estrai il numero del fold e la loss
            parts = line.split()
            fold_num = int(parts[1])
            f1_value = float(parts[6].strip(','))

            # Aggiorna il minimo se la loss corrente è inferiore a quella minima trovata
            if f1_value > max_f1:
                max_f1 = f1_value
                best_fold = fold_num

    # Stampa il risultato
    return best_fold

################################# TRAIN AND VALIDATE ########################################
class JointCrossEntropyLoss(nn.Module):
    def __init__(self, lamd : float = 0.6) -> None:
        super().__init__()
        self.lamd = lamd

    def forward(self, out, label):
        end_out = out[0]
        branch_out = out[1]
        end_loss = F.nll_loss(end_out, label)
        branch_loss = [F.nll_loss(out, label).unsqueeze(0) for out in branch_out]
        branch_loss = torch.cat(branch_loss)
        loss = self.lamd * end_loss + (1 - self.lamd) * torch.sum(branch_loss)
        return loss


def validate(model, val_loader, criterion_tasks, criterion_subjects, alpha, device):
    model.eval()
    val_loss = 0.0
    val_loss_tasks = 0.0
    val_loss_subjects = 0.0
    all_preds_tasks = []
    all_preds_subjects = []
    all_labels_tasks = []
    all_labels_subjects = []
    with torch.no_grad():
        for x_raw_batch, labels_batch, subjects_batch in val_loader:
            x_raw_batch, labels_batch, subjects_batch = x_raw_batch.to(device), labels_batch.to(device), subjects_batch.to(device)
            output_tasks, output_subjects = model(x_raw_batch)
            loss_tasks = criterion_tasks(output_tasks, labels_batch)
            loss_subjects = criterion_subjects(output_subjects, subjects_batch)
            loss = ((1-alpha)*loss_tasks) + (alpha*loss_subjects)
            val_loss += loss.detach().item()
            val_loss_tasks += loss_tasks.detach().item()
            val_loss_subjects += loss_subjects.detach().item()

            # Ottenere le predizioni
            if isinstance(criterion_tasks, JointCrossEntropyLoss):
                output_tasks = output_tasks[0]
                output_subjects = output_subjects[0]
            
            _, preds_tasks = torch.max(output_tasks, 1)
            all_preds_tasks.extend(preds_tasks.cpu().numpy())
            all_labels_tasks.extend(labels_batch.cpu().numpy())
            # Ottenere le predizioni sui soggetti
            _, preds_subjects = torch.max(output_subjects, 1)
            all_preds_subjects.extend(preds_subjects.cpu().numpy())
            all_labels_subjects.extend(subjects_batch.cpu().numpy())

        avg_loss = val_loss / len(val_loader)
        avg_loss_tasks = val_loss_tasks / len(val_loader)
        avg_loss_subjects = val_loss_subjects / len(val_loader)
        
        f1_tasks = f1_score(all_labels_tasks, all_preds_tasks, average=None)
        f1_subjects = f1_score(all_labels_subjects, all_preds_subjects, average=None)
        accuracy_tasks = accuracy_score(all_labels_tasks, all_preds_tasks)
        accuracy_subjects = accuracy_score(all_labels_subjects, all_preds_subjects)
        balanced_accuracy_tasks = balanced_accuracy_score(all_labels_tasks, all_preds_tasks)
        balanced_accuracy_subjects = balanced_accuracy_score(all_labels_subjects, all_preds_subjects)
        # conf_matrix = confusion_matrix(all_labels, all_preds)
    return avg_loss, avg_loss_tasks, avg_loss_subjects, f1_tasks.tolist(), f1_subjects.tolist(), accuracy_tasks, accuracy_subjects, balanced_accuracy_tasks, balanced_accuracy_subjects

def validate_loso(model, val_loader, criterion_tasks, criterion_subjects, alpha, device):
    model.eval()
    all_preds_tasks = []
    all_preds_subjects = []
    all_labels_tasks = []
    all_labels_subjects = []
    with torch.no_grad():
        for x_raw_batch, labels_batch, subjects_batch in val_loader:
            x_raw_batch, labels_batch, subjects_batch = x_raw_batch.to(device), labels_batch.to(device), subjects_batch.to(device)
            output_tasks, output_subjects = model(x_raw_batch)

            # Ottenere le predizioni
            if isinstance(criterion_tasks, JointCrossEntropyLoss):
                output_tasks = output_tasks[0]
                output_subjects = output_subjects[0]
            
            _, preds_tasks = torch.max(output_tasks, 1)
            all_preds_tasks.extend(preds_tasks.cpu().numpy())
            all_labels_tasks.extend(labels_batch.cpu().numpy())
            # Ottenere le predizioni sui soggetti
            _, preds_subjects = torch.max(output_subjects, 1)
            all_preds_subjects.extend(preds_subjects.cpu().numpy())
        
        f1_tasks = f1_score(all_labels_tasks, all_preds_tasks, average=None)
        accuracy_tasks = accuracy_score(all_labels_tasks, all_preds_tasks)
        balanced_accuracy_tasks = balanced_accuracy_score(all_labels_tasks, all_preds_tasks)
        # conf_matrix = confusion_matrix(all_labels, all_preds)
    return f1_tasks.tolist(), accuracy_tasks, balanced_accuracy_tasks, all_preds_subjects

    


def validate_autoencoder(model, val_loader, criterion_tasks, criterion_reconstruction, alpha, device, masked,test=False):
    model.eval()
    val_loss = 0.0
    val_loss_tasks = 0.0
    val_loss_reconstruction = 0.0
    all_preds_tasks = []
    all_labels_tasks = []
    with torch.no_grad():
        for x_raw_batch, labels_batch in val_loader:
            x_raw_batch, labels_batch = x_raw_batch.to(device), labels_batch.to(device)
            if not test:
                x_raw_batch_masked = zero_segments(x_raw_batch, masked)
            else:
                x_raw_batch_masked = x_raw_batch.clone()
            output_tasks, output_reconstruction = model(x_raw_batch_masked)
            loss_tasks = criterion_tasks(output_tasks, labels_batch)
            loss_reconstruction = criterion_reconstruction(output_reconstruction, x_raw_batch)
            loss = ((1-alpha)*loss_tasks) + (alpha*loss_reconstruction)
            val_loss += loss.detach().item()
            val_loss_tasks += loss_tasks.detach().item()
            val_loss_reconstruction += loss_reconstruction.detach().item()

            # Ottenere le predizioni
            _, preds_tasks = torch.max(output_tasks, 1)
            all_preds_tasks.extend(preds_tasks.cpu().numpy())
            all_labels_tasks.extend(labels_batch.cpu().numpy())

        avg_loss = val_loss / len(val_loader)
        avg_loss_tasks = val_loss_tasks / len(val_loader)
        avg_loss_reconstruction = val_loss_reconstruction / len(val_loader)
        
        f1_tasks = f1_score(all_labels_tasks, all_preds_tasks, average=None)
        accuracy_tasks = accuracy_score(all_labels_tasks, all_preds_tasks)
        balanced_accuracy_tasks = balanced_accuracy_score(all_labels_tasks, all_preds_tasks)
        # conf_matrix = confusion_matrix(all_labels, all_preds)
    return avg_loss, avg_loss_tasks, avg_loss_reconstruction, f1_tasks.tolist(), accuracy_tasks, balanced_accuracy_tasks


def train_model(model, fold_performance, train_loader, val_loader, fold, criterion_tasks, 
                criterion_subjects, lr, epochs, device, augmentation, patience, alpha, checkpoint_flag):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    best_val_loss = float('inf')
    best_val_f1_tasks = 0.0
    best_val_f1_subjects = 0.0
    best_val_loss_tasks = 0.0
    best_val_loss_subjects = 0.0
    best_val_accuracy_tasks = 0.0
    best_val_accuracy_subjects = 0.0
    best_val_balanced_accuracy_tasks = 0.0
    best_val_balanced_accuracy_subjects = 0.0
    list_loss_train, list_loss_train_tasks, list_loss_train_subjects = [], [], []
    list_loss_validation, list_loss_validation_tasks, list_loss_validation_subjects = [], [], []
    early_stopping_counter = 0
    start_epoch = 0
    skip_fold_flag = False
    
    if checkpoint_flag and os.path.exists(f"{model.model_name_prefix}_checkpoint_fold{fold+1}.pth"):
        checkpoint = torch.load(f"{model.model_name_prefix}_checkpoint_fold{fold+1}.pth", weights_only=False)
        
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        best_val_loss_tasks = checkpoint['best_val_loss_tasks']
        best_val_loss_subjects = checkpoint['best_val_loss_subjects']
        best_val_f1_tasks = checkpoint['best_val_f1_tasks']
        best_val_f1_subjects = checkpoint['best_val_f1_subjects']
        best_val_accuracy_tasks = checkpoint['best_val_accuracy_tasks']
        best_val_accuracy_subjects = checkpoint['best_val_accuracy_subjects']
        best_val_balanced_accuracy_tasks = checkpoint['best_val_balanced_accuracy_tasks']
        best_val_balanced_accuracy_subjects = checkpoint['best_val_balanced_accuracy_subjects']
        early_stopping_counter = checkpoint['early_stopping_counter']
        list_loss_validation = checkpoint['list_loss_validation']
        list_loss_validation_tasks = checkpoint['list_loss_validation_tasks']
        list_loss_validation_subjects = checkpoint['list_loss_validation_subjects']
        list_loss_train = checkpoint['list_loss_train']
        list_loss_train_tasks = checkpoint['list_loss_train_tasks']
        list_loss_train_subjects = checkpoint['list_loss_train_subjects']
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['finish']:
            skip_fold_flag = True
            print('Skip Fold')

    for epoch in range(start_epoch, epochs, 1):
        if skip_fold_flag:
            break
        model.train()
        running_loss = 0.0
        running_loss_tasks = 0.0
        running_loss_subjects = 0.0

        for x_raw, label, subject in train_loader:
            x_raw, label, subject = x_raw.to(device), label.to(device), subject.to(device)
            if augmentation is not None:
                x_raw, label, subject = augmentation_factory_methods[augmentation](x_raw,label, subject)
            optimizer.zero_grad()
            output_tasks, output_subjects = model(x_raw)

            loss_tasks = criterion_tasks(output_tasks, label)
            loss_subjects = criterion_subjects(output_subjects, subject)
            loss = ((1-alpha)*loss_tasks) + (alpha*loss_subjects)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.detach().item()
            running_loss_tasks += loss_tasks.detach().item()
            running_loss_subjects += loss_subjects.detach().item()

        # calcolare la loss di validation
        (val_loss, val_loss_tasks, 
        val_loss_subjects, val_f1_tasks, 
        val_f1_subjects, val_accuracy_tasks, 
        val_accuracy_subjects, val_balanced_accuracy_tasks, 
        val_balanced_accuracy_subjects) = validate(model, val_loader, criterion_tasks, criterion_subjects, alpha, device)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_tasks = val_loss_tasks
            best_val_loss_subjects = val_loss_subjects
            best_val_f1_tasks = val_f1_tasks
            best_val_f1_subjects = val_f1_subjects
            best_val_accuracy_tasks = val_accuracy_tasks
            best_val_accuracy_subjects = val_accuracy_subjects
            best_val_balanced_accuracy_tasks = val_balanced_accuracy_tasks
            best_val_balanced_accuracy_subjects = val_balanced_accuracy_subjects
            early_stopping_counter = 0
            torch.save(model.state_dict(),
                       f"{model.model_name_prefix}_best_model_fold{fold+1}.pth")
        else:
            early_stopping_counter += 1
        
        checkpoint = {
            'epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_loss_tasks': best_val_loss_tasks,
            'best_val_loss_subjects': best_val_loss_subjects,
            'best_val_f1_tasks': best_val_f1_tasks,
            'best_val_f1_subjects': best_val_f1_subjects, 
            'best_val_accuracy_tasks': best_val_accuracy_tasks, 
            'best_val_accuracy_subjects': best_val_accuracy_subjects, 
            'best_val_balanced_accuracy_tasks': best_val_balanced_accuracy_tasks, 
            'best_val_balanced_accuracy_subjects': best_val_balanced_accuracy_subjects,
            'early_stopping_counter': early_stopping_counter,
            'list_loss_validation': list_loss_validation,
            'list_loss_validation_tasks': list_loss_validation_tasks,
            'list_loss_validation_subjects': list_loss_validation_subjects,
            'list_loss_train': list_loss_train,
            'list_loss_train_tasks': list_loss_train_tasks,
            'list_loss_train_subjects': list_loss_train_subjects,
            'model_state_dict': model.state_dict(),
            'finish': True if (early_stopping_counter >= patience) or (epoch+1==epochs) else False
        }
        torch.save(checkpoint, f"{model.model_name_prefix}_checkpoint_fold{fold+1}.pth")

        print(f"Epoch {epoch+1}/{epochs}, TLoss: {running_loss/len(train_loader):.3f}, VLoss: {val_loss:.3f}, "
              f"TLossTask: {running_loss_tasks/len(train_loader):.3f}, VLossTask: {val_loss_tasks:.3f}, "
             f"Validation F1 Score: {val_f1_tasks}, Validation Accuracy: {val_accuracy_tasks:.3f}, "
             f"Early Stopping Counter: {early_stopping_counter}/{patience}")

        list_loss_validation.append(val_loss)
        list_loss_validation_tasks.append(val_loss_tasks)
        list_loss_validation_subjects.append(val_loss_subjects)
        list_loss_train.append(running_loss / len(train_loader))
        list_loss_train_tasks.append(running_loss_tasks / len(train_loader))
        list_loss_train_subjects.append(running_loss_subjects / len(train_loader))
        
        plot_losses(list_loss_train, list_loss_train_tasks, list_loss_train_subjects, 
                list_loss_validation, list_loss_validation_tasks, list_loss_validation_subjects, 
                fold + 1, f"{model.model_name_prefix}")
        
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Save the loss values
    print("Min Train Loss: ", min(list_loss_train))
    print("Min Val Loss: ", min(list_loss_validation))
    fold_performance.append((best_val_loss, best_val_loss_tasks, best_val_loss_subjects,
                             best_val_f1_tasks, best_val_f1_subjects, 
                             best_val_accuracy_tasks, best_val_accuracy_subjects,
                             best_val_balanced_accuracy_tasks, best_val_balanced_accuracy_subjects))
    
    
def train_fine_tuning_model(model, fold_performance, train_loader, val_loader, fold, criterion,
                lr, epochs, device, augmentation, patience, checkpoint_flag):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_conf_matrix = np.zeros(shape=(2, 2))
    best_val_accuracy = 0.0
    best_val_balanced_accuracy = 0.0
    list_loss_train = []
    list_loss_validation = []
    early_stopping_counter = 0
    start_epoch = 0
    skip_fold_flag = False
    
    if checkpoint_flag and os.path.exists(f"{model.model_name_prefix}_checkpoint_fold{fold+1}.pth"):
        checkpoint = torch.load(f"{model.model_name_prefix}_checkpoint_fold{fold+1}.pth", weights_only=False)
        
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        best_val_f1 = checkpoint['best_val_f1']
        best_val_conf_matrix = checkpoint['best_val_conf_matrix']
        best_val_accuracy = checkpoint['best_val_accuracy']
        best_val_balanced_accuracy = checkpoint['best_val_balanced_accuracy']
        early_stopping_counter = checkpoint['early_stopping_counter']
        list_loss_validation = checkpoint['list_loss_validation']
        list_loss_train = checkpoint['list_loss_train']
        model.load_state_dict(checkpoint['model_state_dict'])
        skip_fold_flag = checkpoint['finish']
    
    for epoch in range(start_epoch, epochs, 1):
        if skip_fold_flag:
            print('Skip Fold')
            break
        model.train()
        running_loss = 0.0

        for x_raw, label, label_subject in train_loader:
            x_raw, label = x_raw.to(device), label.to(device)
            if augmentation is not None:
                x_raw, label, _ = augmentation_factory_methods[augmentation](x_raw,label,label_subject)
            optimizer.zero_grad()
            outputs, _ = model(x_raw)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()

        # calcolare la loss di validation
        val_loss, val_f1, val_conf_matrix, val_accuracy, val_balanced_accuracy = validate_fine_tuning(model, val_loader, criterion, device)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_val_conf_matrix = val_conf_matrix
            best_val_accuracy = val_accuracy
            best_val_balanced_accuracy = val_balanced_accuracy
            early_stopping_counter = 0
            torch.save(model.state_dict(),
                       f"{model.model_name_prefix}_best_model_fold{fold+1}.pth")
        else:
            early_stopping_counter += 1

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, "
             f"Validation F1 Score: {val_f1}, Validation Accuracy: {val_accuracy:.4f}, "
             f"Early Stopping Counter: {early_stopping_counter}/{patience}")
        
        checkpoint = {
            'epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_f1': best_val_f1,
            'best_val_conf_matrix': best_val_conf_matrix,
            'best_val_accuracy': best_val_accuracy,
            'best_val_balanced_accuracy': best_val_balanced_accuracy,
            'early_stopping_counter': early_stopping_counter,
            'list_loss_validation': list_loss_validation,
            'list_loss_train': list_loss_train,
            'model_state_dict': model.state_dict(),
            'finish': True if (early_stopping_counter >= patience) or (epoch+1==epochs) else False
        }
        torch.save(checkpoint, f"{model.model_name_prefix}_checkpoint_fold{fold+1}.pth")

        list_loss_validation.append(val_loss)
        list_loss_train.append(running_loss / len(train_loader))
        
        # Save the loss values
        plot_losses(list_loss_train, list_loss_train, list_loss_train, list_loss_validation, list_loss_validation, list_loss_validation, fold + 1, f"{model.model_name_prefix}")
        
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Min Train Loss: ", min(list_loss_train))
    print("Min Val Loss: ", min(list_loss_validation))
    fold_performance.append((best_val_loss, best_val_f1, best_val_conf_matrix, best_val_accuracy, best_val_balanced_accuracy))

def validate_fine_tuning(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_raw_batch, labels_batch, _ in val_loader:
            x_raw_batch, labels_batch = x_raw_batch.to(device), labels_batch.to(device)
            outputs, _ = model(x_raw_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.detach().item()

            # Ottenere le predizioni
            if isinstance(criterion, JointCrossEntropyLoss):
                outputs=outputs[0]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

        avg_loss = val_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average=None)
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
    return avg_loss, f1.tolist(), conf_matrix, accuracy, balanced_accuracy

############################################ AUTOENCODER ##################################################
def zero_segments(x, high=400):
    B,_,C,T = x.shape
    signal = x.clone()
    dim_segment = torch.randint(low=high-100, high=high, size=(1,))[0]
    batch_segments = torch.randint(high=(T-dim_segment), size=(B,))
    for elem in range(B):
        signal[elem,: ,: ,batch_segments[elem]:batch_segments[elem]+dim_segment] = 0
    return signal

def train_model_autoencoder(model, fold_performance, train_loader, val_loader, fold, class_weight,
                lr, epochs, device, augmentation, patience, alpha, masked):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_tasks = nn.CrossEntropyLoss(weight=class_weight)
    criterion_reconstruction = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_val_loss_tasks = float('inf')
    best_val_loss_reconstruction = float('inf')
    
    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    best_val_balanced_accuracy = 0.0
    list_loss_train, list_loss_train_tasks, list_loss_train_reconstruction = [], [], []
    list_loss_validation, list_loss_validation_tasks, list_loss_validation_reconstruction = [], [], []
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss_task = 0.0
        running_loss_reconstruction = 0.0

        for x_raw, label in train_loader:
            x_raw, label = x_raw.to(device), label.to(device)
            x_raw_masked = zero_segments(x_raw, masked)
            # if augmentation is not None:
            #     x_raw, label = augmentation_factory_methods[augmentation](x_raw,label)
            optimizer.zero_grad()
            output_tasks, output_reconstruction = model(x_raw_masked)
            loss_task = criterion_tasks(output_tasks, label)
            loss_reconstruction = criterion_reconstruction(output_reconstruction, x_raw)
            loss = ((1-alpha)*loss_task) + (alpha*loss_reconstruction)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach().item()
            running_loss_task += loss_task.detach().item()
            running_loss_reconstruction += loss_reconstruction.detach().item()

        # calcolare la loss di validation
        val_loss, val_loss_tasks, val_loss_reconstruction, val_f1, val_accuracy, val_balanced_accuracy = validate_autoencoder(model, val_loader, criterion_tasks, criterion_reconstruction, alpha, device, masked, False)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_tasks = val_loss_tasks
            best_val_loss_reconstruction = val_loss_reconstruction
            best_val_f1 = val_f1
            best_val_accuracy = val_accuracy
            best_val_balanced_accuracy = val_balanced_accuracy
            early_stopping_counter = 0
            torch.save(model.state_dict(),
                       f"{model.model_name_prefix}_best_model_fold{fold+1}.pth")
        else:
            early_stopping_counter += 1

        # print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, "
        #      f"Validation F1 Score: {val_f1}, Validation Accuracy: {val_accuracy:.4f}, "
        #      f"Early Stopping Counter: {early_stopping_counter}/{patience}")

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        list_loss_validation.append(val_loss), list_loss_validation_tasks.append(val_loss_tasks), list_loss_validation_reconstruction.append(val_loss_reconstruction)
        list_loss_train.append(running_loss / len(train_loader)), list_loss_train_tasks.append(running_loss_task / len(train_loader)), list_loss_train_reconstruction.append(running_loss_reconstruction / len(train_loader))

    # Save the loss values
    plot_losses(list_loss_train, list_loss_train_tasks, list_loss_train_reconstruction, list_loss_validation, list_loss_validation_tasks, list_loss_validation_reconstruction, fold + 1, f"{model.model_name_prefix}", True)
    print("Min Train Loss: ", min(list_loss_train))
    print("Min Val Loss: ", min(list_loss_validation))
    fold_performance.append((best_val_loss, best_val_loss_tasks, best_val_loss_reconstruction, best_val_f1, best_val_accuracy, best_val_balanced_accuracy))