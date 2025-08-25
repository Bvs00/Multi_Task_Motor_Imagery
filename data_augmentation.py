import random
import numpy as np
import torch


def chr_augmentation(data, labels):
    """
    Preso il dato di input [batch, 1, 3, 1000] devo invertire i canali e invertire la labels.
    
    """
    data_inverted = torch.zeros_like(data)
    data_inverted[:, :, [0, 2], :] = data[:, :, [2, 0], :]
    
    num_segments=8
    length_segment=data.shape[3]//num_segments
    
    new_data = torch.empty_like(data)
    
    for i in range(data.shape[0]):
        for seg_idx in range(num_segments):
            # Selezioniamo casualmente un campione dal batch originale
            random_sample = random.randint(0, data.shape[0] - 1)
            
            # Prendiamo il segmento corretto e lo assembliamo nel nuovo campione
            start = seg_idx * length_segment
            end = (seg_idx + 1) * length_segment
            
            # Assembliamo mantenendo la struttura [batch, 1, 3, 1000]
            new_data[i, :, :, start:end] = data[random_sample, :, :, start:end]

    return torch.cat([data, data_inverted, new_data]), torch.cat([labels, (1-labels), labels])

def reverse_channels(data, labels, subjects):
    """
    Preso il dato di input [batch, 1, 3, 1000] devo invertire i canali e invertire la labels.
    
    """
    data_inverted = torch.zeros_like(data)
    data_inverted[:, :, [0, 2], :] = data[:, :, [2, 0], :]
    
    return torch.cat([data, data_inverted]), torch.cat([labels, (1-labels)]), torch.cat([subjects, subjects])

def segmentation_reconstruction(data, labels, subjects, num_segments=8, num_augmentations=3):
    _, conv, ch, time = data.shape 
    full_data = data.clone()
    full_labels = labels.clone()
    full_subjects = subjects.clone()
    type_labels = torch.unique(labels)
    type_subjects = torch.unique(subjects)
    length_segment=time//num_segments
    
    # creo i nuovi dati a partire da gruppi di campioni che appartengono allo stesso soggetto e alla stessa classe.
    for subject in type_subjects:
        idx_subjects = subjects==subject
        if not idx_subjects.any():
            continue
        
        data_for_subject = data[idx_subjects]
        labels_for_subject = labels[idx_subjects]
    
        for label in type_labels:
            idx_labels = labels_for_subject==label
            if not idx_labels.any():
                continue
            
            data_for_label = data_for_subject[idx_labels]
            n_samples = data_for_label.size(0)
            num_samples_for_classes = n_samples*num_augmentations
            
            new_data = torch.empty((num_samples_for_classes, conv, ch, time), device=data.device)
            random_sample = torch.randint(0, n_samples, (num_samples_for_classes, num_segments), device=data.device)
            
            for seg_idx in range(num_segments):
                # Prendiamo il segmento corretto e lo assembliamo nel nuovo campione
                start = seg_idx * length_segment
                end = (seg_idx + 1) * length_segment
                # Assembliamo mantenendo la struttura [batch, 1, 3, 1000]
                new_data[:, :, :, start:end] = data_for_label[random_sample[:, seg_idx], :, :, start:end]
            
            full_data = torch.cat([full_data, new_data], dim=0)
            full_labels = torch.cat([full_labels, label.repeat(num_samples_for_classes)], dim=0)
            full_subjects = torch.cat([full_subjects, subject.repeat(num_samples_for_classes)], dim=0)
    
    idx_shuffled = torch.randperm(full_data.size(0))
    return full_data[idx_shuffled], full_labels[idx_shuffled], full_subjects[idx_shuffled]

def reverse_channels_segmentation_reconstruction(data, labels, subjects, num_segments=8, num_augmentations=3):
    data_aug, labels_aug, subjects_aug = reverse_channels(data, labels, subjects)
    final_data_aug, final_labels_aug, full_subjects_aug = segmentation_reconstruction(data_aug, labels_aug, subjects_aug, num_segments, num_augmentations)
    return final_data_aug, final_labels_aug, full_subjects_aug