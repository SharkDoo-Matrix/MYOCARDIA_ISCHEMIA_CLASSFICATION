import os

import torch
import model.three_process_train as TrainModel
import dataset.mcg_dataset as MCGDataset
import model.backbone.resnet3d as ResNet3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# -------------------------- main  --------------------------
if __name__ == "__main__":
    # param config
    args = {
        'train_root': "",
        'val_root': "",
        'treat_csv_path': "",
        'model_backbone_dir':'log/resnet3d',
        'model_save_dir':'log/resnet3d',
        'save_path_stage1': "stage1_multi_t_best.pth",
        'save_path_stage2': "stage2_multi_t_best.pth",
        'save_path_final': "final_multi_t_best.pth",
        'device': device,
        'pretrained': False,

        # 数据参数
        'seq_len': 100,
        'batch_size': 8,  # 多维模型计算量大，建议batch_size小一些
        'contrast': True,

        # 训练参数
        'lr_stage1': 3e-5,
        'epochs_stage2': 20,
        'lr_stage2': 4e-5,
        'stage1_epoch': 0,
        'stage2_epoch': 0,
        'lr_stage3': 3e-5
    }
    train_loader, val_loader = MCGDataset.get_dataloader(args)
    # 数据路径
    args['train_loader'] = train_loader
    args['val_loader'] = val_loader

    # 模型定义
    args['backbone_class'] = ResNet3D.ResNet3D
    args['backbone_model'] = ResNet3D.get_resnet3d(args)

    # 启动训练
    TrainModel.train_multi_t_model(args)

