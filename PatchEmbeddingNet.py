import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import math
    
################ PATCHEMBEDDING ######################
class PatchEmbeddingNet(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, 
                 channels=3, num_classes=2, model_name_prefix="PatchEmbeddingNet", samples=1000, subjects=9):
        super().__init__()
        f2 = D*f1
        self.channels = channels
        self.samples = samples
        self.model_name_prefix = model_name_prefix
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (channels, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.classification_tasks = nn.Linear(self.num_features_linear(), num_classes)
        self.classification_subjects = nn.Linear(self.num_features_linear(), subjects)
        
    
    def num_features_linear(self):
        x = torch.ones((1, 1, self.channels, self.samples))
        x = self.cnn_module(x)
        x = self.projection(x)
        return x.shape[-1]*x.shape[-2]
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        x = x.flatten(start_dim=1)
        
        out_tasks = self.classification_tasks(x)
        out_subjects = self.classification_subjects(x)
        
        return out_tasks, out_subjects


################ PATCHEMBEDDING SOFT ######################
class PatchEmbeddingNet_Soft(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, 
                 channels=3, num_classes=2, model_name_prefix="PatchEmbeddingNet_Soft", samples=1000, subjects=9):
        super().__init__()
        f2 = D*f1
        self.channels = channels
        self.samples = samples
        self.model_name_prefix = model_name_prefix
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (channels, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
        )
        self.task_branch = nn.Sequential(
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
        )
        self.subject_branch = nn.Sequential(
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.classification_tasks = nn.Linear(self.num_features_linear(), num_classes)
        self.classification_subjects = nn.Linear(self.num_features_linear(), subjects)
        
    
    def num_features_linear(self):
        x = torch.ones((1, 1, self.channels, self.samples))
        x = self.cnn_module(x)
        x = self.task_branch(x)
        x = self.projection(x)
        return x.shape[-1]*x.shape[-2]
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        
        x_task = self.task_branch(x)
        x_task = self.projection(x_task)
        x_task = x_task.flatten(start_dim=1)
        out_tasks = self.classification_tasks(x_task)
        
        x_subj = self.subject_branch(x)
        x_subj = self.projection(x_subj)
        x_subj = x_subj.flatten(start_dim=1)
        out_subjects = self.classification_subjects(x_subj)
        
        return out_tasks, out_subjects


####################### MASKED AUTOENCODER ############################
class PatchEmbeddingNet_Autoencoder(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, 
                 channels=3, num_classes=2, model_name_prefix="PatchEmbeddingNet", samples=1000, subjects=9):
        super().__init__()
        f2 = D*f1
        self.channels = channels
        self.samples = samples
        self.model_name_prefix = model_name_prefix
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (channels, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.decoder = nn.Sequential(
            Rearrange('b (h w) e -> b e h w', h=1),
            # Step 5 - Reverse of AvgPool2d((1, p2)) → Upsample
            nn.Upsample(size=(1, 125)),  # 15 → 125
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False),  # reverse of encoder conv
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # Step 3 - Reverse of AvgPool2d((1, p1))
            nn.Upsample(size=(1, 1000)),  # 125 → 1000
            nn.Dropout(dropout_rate),  # match encoder dropout

            # Step 2 - Reverse depthwise conv (1 → 3 electrodes)
            nn.ConvTranspose2d(f2, f1, (channels, 1), groups=1, bias=False),  # 1→3
            nn.BatchNorm2d(f1),
            nn.ELU(),

            # Step 1 - Reverse of temporal conv
            nn.Conv2d(f1, 1, (1, kernel_size), padding='same', bias=False),  # reduce to 1 channel
            nn.BatchNorm2d(1),
        )


        self.classification_tasks = nn.Linear(self.num_features_linear(), num_classes)
    
    def num_features_linear(self):
        x = torch.ones((1, 1, self.channels, self.samples))
        x = self.cnn_module(x)
        x = self.projection(x)
        return x.shape[-1]*x.shape[-2]
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        latent = self.projection(x)
        latent_cls = latent.flatten(start_dim=1)
        
        out_tasks = self.classification_tasks(latent_cls)
        out_reconstruction = self.decoder(latent)
        
        return out_tasks, out_reconstruction
