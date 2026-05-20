import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import math


#################################################################################################################################
#################################################            MSVTNet            #################################################
#################################################################################################################################


class TSConv(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding='same', groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x += self.pe
        return x
        

class Transformer(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model, 
        nhead, 
        ff_ratio, 
        Pt = 0.5, 
        num_layers = 4, 
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_len + 1, d_model)

        dim_ff =  d_model * ff_ratio
        self.dropout = nn.Dropout(Pt)
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, Pt, batch_first=True, norm_first=False #era True ho modificato in False per un warining
        ), num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x):
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class ClsHead(nn.Sequential):
    def __init__(self, linear_in, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(linear_in, cls),
            nn.LogSoftmax(dim=1)
        )


class MSVTNet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTNet', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc),
                Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
            )
            for b in range(len(F))
        ])
        branch_linear_in = self._forward_flatten(cat=False)
        
        self.branch_head_task = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])
        self.branch_head_subject = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], subjects)
            for b in range(len(F))
        ])

        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat = True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head_task)]
        cx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head_subject)]
        x = torch.cat(x, dim=2)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, bx], [output_subject, cx]
        else:
            return output_task, output_subject
        
######################################  MSVTSENet ######################################
# SENet (Squeeze-and-Excitation Network) per l'attenzione sui canali
class SENet(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SENet, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc1 = nn.Linear(channels, channels // reduction)  # Riduzione dimensionale
        self.fc2 = nn.Linear(channels // reduction, channels)  # Espansione dimensionale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, feature_maps, _, _ = x.shape
        y = self.global_avg_pool(x).view(batch, feature_maps)  # Global Average Pooling
        y = F.relu(self.fc1(y))  # ReLU dopo la riduzione dimensionale
        y = self.sigmoid(self.fc2(y)).view(batch, feature_maps, 1, 1)  # Sigmoid e reshape
        return x * y  # Applicazione dei pesi ai canali originali

class MSVTSENet(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVTNet', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
        reduction = 2
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            TSConv(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_layer = SENet(D*sum(F), reduction)
        branch_linear_in = self._forward_flatten(cat=False)
        
        self.branch_head_task = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])
        self.branch_head_subject = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], subjects)
            for b in range(len(F))
        ])

        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat = True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x, return_attention=False):
        x = [tsconv(x) for tsconv in self.mstsconv]
        branch_task = [branch(self.rearrange(x[idx])) for idx, branch in enumerate(self.branch_head_task)]
        branch_subject = [branch(self.rearrange(x[idx])) for idx, branch in enumerate(self.branch_head_subject)]
        x = torch.cat(x, dim=1)
        x = self.se_layer(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, branch_task], [output_subject, branch_subject]
        else:
            return output_task, output_subject


################################### MSVT_SE_Net #######################################

class TSConv_SE(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc, reduction=2) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            SENet(F, reduction),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding='same', groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )



class MSVT_SE_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_SE_Net', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv_SE(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc),
                Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
            )
            for b in range(len(F))
        ])
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])
        self.branch_head_subject = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], subjects)
            for b in range(len(F))
        ])
        
        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat = True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        cx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head_subject)]
        x = torch.cat(x, dim=2)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, bx], [output_subject, cx]
        else:
            return output_task, output_subject


################################### MSVT_SE_SE_Net #######################################

class MSVT_SE_SE_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_SE_SE_Net', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
        reduction = 2,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'
        print(F)
        self.mstsconv = nn.ModuleList([
            TSConv_SE(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc, reduction)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_layer = SENet(D*sum(F), reduction=reduction)
        branch_linear_in = self._forward_flatten(cat=False)
        
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])
        self.branch_head_subject = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], subjects)
            for b in range(len(F))
        ])
        
        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat = True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        branch_task = [branch(self.rearrange(x[idx])) for idx, branch in enumerate(self.branch_head)]
        branch_subject = [branch(self.rearrange(x[idx])) for idx, branch in enumerate(self.branch_head_subject)]
        x = torch.cat(x, dim=1)
        x = self.se_layer(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, branch_task], [output_subject, branch_subject]
        else:
            return output_task, output_subject
        
        
############################

class Convolutional_SE_Block(nn.Sequential):
    def __init__(self, nCh, F, C1, D, P1, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            SENet(F),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc)
        )

class Convolutional_Block_SE(nn.Sequential):
    def __init__(self, nCh, F, C1, D, P1, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            SENet(F*D),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc)
        )

class Convolutional_Union_SE_Block(nn.Sequential):
    def __init__(self, F, C2, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(F, F, (1, C2), padding='same', groups=F, bias=False),
            nn.BatchNorm2d(F),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )
    

class MSVT_Custom_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_Custom_Net', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'
        print(F)
        self.mstsconv = nn.ModuleList([
            Convolutional_SE_Block(self.nCh, F[b], C1[b], D, P1, Pc)
            for b in range(len(F))
        ])
        self.convolutional_union = Convolutional_Union_SE_Block(D*sum(F), C2, P2, Pc)
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_layer = SENet(D*sum(F))
        branch_linear_in = self._forward_flatten(transformer=False)
        
        self.branch_head = ClsHead(branch_linear_in.shape[1], num_classes)    
        self.branch_head_subject = ClsHead(branch_linear_in.shape[1], subjects)
        
        # d_model è l'informazione di ogni token
        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore [17, 72]
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = torch.cat(x, dim=1)
        x = self.rearrange(self.convolutional_union(x))
        return x

    def _forward_flatten(self, transformer = True):
        x = self._forward_mstsconv()
        if transformer:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = x.flatten(start_dim=1, end_dim=-1)
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        branch_task = self.branch_head(self.rearrange(self.convolutional_union(torch.cat(x, dim=1))))
        branch_subject = self.branch_head_subject(self.rearrange(self.convolutional_union(torch.cat(x, dim=1))))
        x = torch.cat(x, dim=1)
        x = self.convolutional_union(x)
        x = self.se_layer(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, branch_task], [output_subject, branch_subject]
        else:
            return output_task, output_subject
        

class MSVT_Custom_SE_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_Custom_Net', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'
        print(F)
        self.mstsconv = nn.ModuleList([
            Convolutional_Block_SE(self.nCh, F[b], C1[b], D, P1, Pc)
            for b in range(len(F))
        ])
        self.convolutional_union = Convolutional_Union_SE_Block(D*sum(F), C2, P2, Pc)
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_layer = SENet(D*sum(F))
        branch_linear_in = self._forward_flatten(transformer=False)
        
        self.branch_head = ClsHead(branch_linear_in.shape[1], num_classes)    
        self.branch_head_subject = ClsHead(branch_linear_in.shape[1], subjects)
        
        # d_model è l'informazione di ogni token
        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore [17, 72]
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = torch.cat(x, dim=1)
        x = self.rearrange(self.convolutional_union(x))
        return x

    def _forward_flatten(self, transformer = True):
        x = self._forward_mstsconv()
        if transformer:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = x.flatten(start_dim=1, end_dim=-1)
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        branch_task = self.branch_head(self.rearrange(self.convolutional_union(torch.cat(x, dim=1))))
        branch_subject = self.branch_head_subject(self.rearrange(self.convolutional_union(torch.cat(x, dim=1))))
        x = torch.cat(x, dim=1)
        x = self.convolutional_union(x)
        x = self.se_layer(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, branch_task], [output_subject, branch_subject]
        else:
            return output_task, output_subject
        
        
class MSVT_SE_SE_SE_Net(nn.Module):
    def __init__(
        self, channels = 3, samples = 1000, num_classes = 2, model_name_prefix = 'MSVT_SE_SE_Net', subjects = 9,
        F = [9, 9, 9, 9], C1 = [15, 31, 63, 125], C2 = 15, D = 2, P1 = 8, P2 = 7, Pc = 0.3,
        nhead = 8,
        ff_ratio = 1,
        Pt = 0.5,
        layers = 2,
        b_preds = True,
        reduction = 2,
    ) -> None:
        super().__init__()
        self.model_name_prefix = model_name_prefix
        self.nCh = channels
        self.nTime = samples
        self.b_preds = b_preds
        self.subjects=subjects
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'
        print(F)
        self.se_layer_channels = SENet(self.nCh, reduction=reduction)
        self.mstsconv = nn.ModuleList([
            TSConv_SE(self.nCh, F[b], C1[b], C2, D, P1, P2, Pc, reduction)
            for b in range(len(F))
        ])
        self.rearrange = Rearrange('b d 1 t -> b t d')   # b x 18 x 1 x 17 e le feature maps diventano le nostra informazioni per ogni token (la lista di token diventa 17)
        self.se_layer = SENet(D*sum(F), reduction=reduction)
        branch_linear_in = self._forward_flatten(cat=False)
        
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], num_classes)
            for b in range(len(F))
        ])
        self.branch_head_subject = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], subjects)
            for b in range(len(F))
        ])
        
        seq_len, d_model = self._forward_mstsconv().shape[1:3] # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1] # type: ignore
        self.last_head_task = ClsHead(linear_in, num_classes)
        self.last_head_subject = ClsHead(linear_in, subjects)

    def _forward_mstsconv(self, cat = True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        x = [self.rearrange(x_i) for x_i in x]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat = True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = torch.permute(self.se_layer_channels(torch.permute(x, (0,2,1,3))), (0,2,1,3))
        x = [tsconv(x) for tsconv in self.mstsconv]
        branch_task = [branch(self.rearrange(x[idx])) for idx, branch in enumerate(self.branch_head)]
        branch_subject = [branch(self.rearrange(x[idx])) for idx, branch in enumerate(self.branch_head_subject)]
        x = torch.cat(x, dim=1)
        x = self.se_layer(x)
        x = self.rearrange(x)
        x = self.transformer(x)
        output_task = self.last_head_task(x)
        output_subject = self.last_head_subject(x)
        if self.b_preds:
            return [output_task, branch_task], [output_subject, branch_subject]
        else:
            return output_task, output_subject