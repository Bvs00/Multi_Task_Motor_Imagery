import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import math
    
################ PATCHEMBEDDING ######################
class PatchEmbeddingNet(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, 
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
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
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

####################### MASKED AUTOENCODER ############################
class PatchEmbeddingNet_Autoencoder(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, 
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
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
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
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),  # reverse of encoder conv
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


################################ CTNET #################################################

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, number_channel=22, emb_size=16):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x
    
class MultiHeadAttention_CTNet(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    


# PointWise FFN
class FeedForwardBlock_CTNet(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ClassificationHead_CTNet(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd_CTNet(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock_CTNet(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd_CTNet(nn.Sequential(
                MultiHeadAttention_CTNet(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd_CTNet(nn.Sequential(
                FeedForwardBlock_CTNet(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder_CTNet(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock_CTNet(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=3,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )


# learnable positional embedding module        
class PositioinalEncoding_CTNet(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)        
        
   
        
# CTNet       
class CTNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.25,
                 flatten_eeg1 = 240,
                 num_classes = 2,
                 channels = 3,
                 model_name_prefix="CTNet",
                 subjects=9,
                 **kwargs):
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.number_class, self.number_channel = num_classes, channels
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        self.subjects=subjects
        print('self.number_channel', self.number_channel)
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1 = eeg1_f1,
                                              kernel_size = eeg1_kernel_size,
                                              D = eeg1_D,
                                              pooling_size1 = eeg1_pooling_size1,
                                              pooling_size2 = eeg1_pooling_size2,
                                              dropout_rate = eeg1_dropout_rate,
                                              )
        self.position = PositioinalEncoding_CTNet(emb_size, dropout=0.1)
        self.trans = TransformerEncoder_CTNet(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead_CTNet(self.flatten_eeg1 , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module
        self.classification_subjects = ClassificationHead_CTNet(self.flatten_eeg1 , self.subjects)
        
    def forward(self, x):
        cnn = self.cnn(x)

        #  positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        
        trans = self.trans(cnn)
        # residual connect
        features = cnn+trans
        
        out = self.classification(self.flatten(features))
        out_subject = self.classification_subjects(self.flatten(features))
        return out, out_subject
