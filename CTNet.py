import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import math


################################ CTNET #################################################

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, number_channel=22, emb_size=16):
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
                 kernel_size = 63,
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
                 eeg1_kernel_size = 63,
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


######## CTNet Soft-Sharing ###############

class CTNet_Soft(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 63,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.25,
                 flatten_eeg1 = 240,
                 num_classes = 2,
                 channels = 3,
                 model_name_prefix="CTNet_Soft",
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
        self.trans_subjects = TransformerEncoder_CTNet(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead_CTNet(self.flatten_eeg1 , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module
        self.classification_subjects = ClassificationHead_CTNet(self.flatten_eeg1 , self.subjects)
        
    def forward(self, x):
        cnn = self.cnn(x)

        #  positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        
        #task
        trans = self.trans(cnn)
        features = cnn+trans # residual connect
        out = self.classification(self.flatten(features))
        
        
        trans_subjects = self.trans_subjects(cnn)
        features_subject = cnn + trans_subjects
        out_subject = self.classification_subjects(self.flatten(features_subject))
        return out, out_subject
    

#################### CSETNet ####################
# SENet (Squeeze-and-Excitation Network) per l'attenzione sui canali
class SENet(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SENet, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc1 = nn.Linear(channels, channels // reduction)  # Riduzione dimensionale
        self.fc2 = nn.Linear(channels // reduction, channels)  # Espansione dimensionale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_attention=False):
        batch, feature_maps, _, _ = x.shape
        y = self.global_avg_pool(x).view(batch, feature_maps)  # Global Average Pooling
        y = F.relu(self.fc1(y))  # ReLU dopo la riduzione dimensionale
        y = self.sigmoid(self.fc2(y)).view(batch, feature_maps, 1, 1)  # Sigmoid e reshape
        if return_attention:
            return x * y, y
        return x * y  # Applicazione dei pesi ai canali originali
    
class PatchSEEmbedding(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, number_channel=22, emb_size=16):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000]
            SENet(f1),
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
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
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class BranchEEGSENetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=3,
                 f1 = 20,
                 kernel_size = 63,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchSEEmbedding(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )


class CSETNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 63,
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
        self.cnn = BranchEEGSENetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
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


#################### CBAMTNet ####################
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = output * x
        return output

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x
    
class PatchCBAMEmbedding(nn.Module):
    def __init__(self, f1=8, kernel_size=63, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.25, number_channel=22, emb_size=16):
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
            nn.Conv2d(f2, f2, (1, 15), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # cbam
            CBAM(f2, r=4),
            # 
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


class BranchEEGCBAMNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=3,
                 f1 = 20,
                 kernel_size = 63,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchCBAMEmbedding(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
        )


class CCBAMTNet(nn.Module):
    def __init__(self, heads=2, 
                 emb_size=16,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 8,
                 eeg1_kernel_size = 63,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.25,
                 flatten_eeg1 = 240,
                 num_classes = 2,
                 channels = 3,
                 model_name_prefix="CCBAMTNet",
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
        self.cnn = BranchEEGCBAMNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
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