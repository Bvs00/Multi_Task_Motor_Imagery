import argparse
import torch
from scipy.signal import freqz
from utils import available_network, find_minum_loss, network_factory_methods
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_model", type=str, default='MSVT_SE_SE_Net', help="Name of tensors that use", choices=available_network)
    parser.add_argument('--saved_path', type=str, default='Results_2B/Results_Alpha025/Results_SegRec/Results_Cross/Results_MSVT_SE_SE_Net_Wout_Aux')
    parser.add_argument('--saved_path_visualization', type=str, default='Visualization_Frequency_Convolutional_Kernels')
    parser.add_argument("--seed", type=int, default=42, help="Seed of initialization")
    parser.add_argument('--auxiliary_branch', type=str, default='False')
    parser.add_argument("--labels", type=int, default=2)
    parser.add_argument("--num_subjects", type=int, default=9)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else'cpu')
    parser.add_argument('--feature_maps', nargs='+', type=int, default=[9, 9, 9, 9])
    parser.add_argument('--waves', nargs='+', type=int, default=[0, 4, 8, 13, 30])
    parser.add_argument('--max_frequence', type=int, default=80)
    args = parser.parse_args()
    
    saved_path_visualization_frequency = f'{args.saved_path_visualization}/{args.name_model}'
    if not os.path.exists(saved_path_visualization_frequency):
        os.makedirs(saved_path_visualization_frequency)
    
    best_fold = find_minum_loss(f'{args.saved_path}/{args.name_model}_seed{args.seed}_validation_log.txt')
    extra_args = {'b_preds': args.auxiliary_branch, 'F': args.feature_maps} if 'MS' in args.name_model else {}
    model = (
            network_factory_methods[args.name_model](model_name_prefix=f'{args.saved_path}/{args.name_model}_seed{args.seed}',
                num_classes=args.labels, subjects=args.num_subjects,
                samples=args.samples, channels=args.channels, **extra_args)
        )
    model.to(args.device)
    model.load_state_dict(torch.load(f'{args.saved_path}/{args.name_model}_seed{args.seed}_best_model_fold{best_fold}.pth'))
    
    heatmap = np.zeros((36, args.max_frequence))
    # fig, axs = plt.subplots(9, 4, tight_layout=True, figsize=(6*4, 12))
    # fig.suptitle('Frequency response in convolutional kernels')
    count = 0
    for branch in range(4):
        if args.name_model == 'MSVT_SE_Net' or 'MSVTNet':
            conv = model.mstsconv[branch][0][0]
        else:
            conv = model.mstsconv[branch][0]
        b_matrix = conv.weight.data[:,0,0,:]
        for i in range(b_matrix.shape[0]):
            frequencies_w, frequencies_h = freqz(b_matrix[i].cpu().numpy(), fs=250, worN=125)
            heatmap[count]=np.abs(frequencies_h[:args.max_frequence])
            count+=1
            # axs[i][branch].plot(frequencies_w, np.abs(frequencies_h)) # 20 * np.log10(np.abs(frequencies_h) + 1e-12) per la magnitudine in dB
            # axs[i][branch].set_title(f'Branch {branch+1} - Feature Map {i+1}')
    plt.figure(tight_layout=True)
    # cmap=sns.color_palette("vlag", as_cmap=True)
    # cmap=sns.color_palette("icefire", as_cmap=True)
    # cmap=sns.diverging_palette(220, 20, as_cmap=True)
    # cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True)
    # cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    # cmap=sns.color_palette("Spectral", as_cmap=True)
    cmap=sns.color_palette("coolwarm", as_cmap=True)
    # cmap=sns.color_palette("viridis", as_cmap=True)
    # cmap=sns.color_palette("magma", as_cmap=True)
    # cmap=sns.color_palette("rocket", as_cmap=True)
    ax = sns.heatmap(heatmap, cmap=cmap) #coolwarm
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlim(0, heatmap.shape[1])   # asse X parte da 0
    ax.set_ylim(0, heatmap.shape[0])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    for x in args.waves:
        plt.axvline(x=x, c='black')
    x_ticks = np.arange(0,heatmap.shape[1]+1,25).tolist() # 0 = inizio, 34 = fine
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=45)
    
    y_ticks = []
    tmp = 0
    for y in args.feature_maps:
        tmp += y
        y_ticks.append(tmp)
        plt.axhline(y=tmp, c='black')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    plt.ylabel('Feature Maps', fontsize=20)
    plt.xlabel('Frequencies', fontsize=20)
    plt.title('Feature Map Frequencies', fontsize=20)
    plt.tight_layout(pad=0)
    plt.savefig(f'{saved_path_visualization_frequency}/Heatmap', bbox_inches="tight", pad_inches=0.05)
    plt.close()
        # plt.savefig(f'{saved_path_visualization_frequency}/Branch_{branch+1}.png')
    # plt.savefig(f'{saved_path_visualization_frequency}/Branches.png')
    
    
    