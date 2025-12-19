import ast
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -------------------------- dataset --------------------------
class VideoCausalDatasetMultiT(Dataset):
    def __init__(self, root_dir, treat_csv_path, transform=None,
                 sequence_length=100, is_val=False, random_sample=True, t_scaler=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.is_val = is_val
        self.random_sample = random_sample
        self.t_scaler = t_scaler

        # class mapping
        self.class_to_idx = {
            "healthy": 0, "mild_symptoms": 1, "severe_illness": 2
        }

        # load key_features
        self.feat_df = pd.read_csv(treat_csv_path)
        self.feat_df['feature_tensor'] = self.feat_df['feature_tensor'].apply(
            lambda x: self._parse_feat(x)
        )
        # filter invalid features
        self.feat_df = self.feat_df[
            self.feat_df['feature_tensor'].apply(lambda x: len(x) == 13)
        ].reset_index(drop=True)

        # process key_features
        self._process_t_features()

        # collect samples
        self.samples = self._collect_samples()

        self.is_test = is_test

        # validate
        if len(self.samples) == 0:
            raise ValueError(f"not find valid sample（root: {root_dir}, CSV: {treat_csv_path}）")
        print(f"loading {len(self.samples)} samples（{'valid_set' if is_val else 'train_set'}）")

    def _parse_feat(self, x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return [0.0] * 13
        return x if isinstance(x, list) else [0.0] * 13

    def _process_t_features(self):
        # all key_features
        all_t = np.array(self.feat_df['feature_tensor'].tolist())

        if not self.is_val:
            if self.t_scaler is None:
                self.t_scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # 避免边界值
            self.t_scaler.fit(all_t)
            self.feat_df['t_norm'] = self.t_scaler.transform(all_t).tolist()

        else:
            if self.t_scaler is None:
                raise ValueError("The validation set must be passed into the t_scaler fitted by the training set")
            self.feat_df['t_norm'] = self.t_scaler.transform(all_t).tolist()

    def _collect_samples(self):
        samples = []
        for cls_name, label in self.class_to_idx.items():
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Warning：class appendix {cls_dir} non_existing, step")
                continue
            for video_folder in os.listdir(cls_dir):

                video_path = os.path.join(cls_dir, video_folder)
                # validate frame nums
                frame_files = [f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg'))]
                if not (os.path.isdir(video_path) and len(frame_files) >= 10):
                    continue

                # match = self.feat_df[self.feat_df['name'].str.rstrip() == video_folder.rstrip()]
                match = self.feat_df[self.feat_df['name'].astype(str).str.rstrip() == str(video_folder).rstrip()]
                if len(match) != 1:
                    match = self.feat_df[self.feat_df['name'].astype(str).str.rstrip() == str(video_folder).rstrip()]
                    if len(match) != 1:
                        print(f"Warning：vides {video_folder} feature match error, skip")
                        continue
                t_norm = match.iloc[0]['t_norm']
                samples.append((video_path, label, np.array(t_norm)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, t_norm = self.samples[idx]

        # load frames
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg'))], key=lambda x: int(x.split('.')[0]))
        T = len(frame_files)

        if T >= self.sequence_length:
            if self.random_sample and not self.is_val:
                indices = np.random.choice(T, self.sequence_length, replace=False)
                indices.sort()
            else:
                indices = np.linspace(0, T - 1, self.sequence_length, dtype=int)
        else:
            indices = np.random.choice(T, self.sequence_length, replace=True)
            indices.sort()
        # load frames and convert format
        frames = []
        for i in indices:
            img = Image.open(os.path.join(video_path, frame_files[i])).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

        # convert T and label format  转换T和标签格式
        t = torch.tensor(t_norm, dtype=torch.float32)  # [13]
        label = torch.tensor(label, dtype=torch.long)  # [1]

        # validate labels
        assert 0 <= label.item() < 3, f"invalid label: {label.item()}"

        if self.is_test:
            return frames, t, label, video_path
        return frames, t, label


def get_dataloader(args):
    # data preprocessing
    mean_vals = [0.5653775930404663, 0.6673181056976318, 0.391217440366745]
    std_vals = [0.38859298825263977, 0.370522141456604, 0.3847532868385315]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])
    train_dataset = VideoCausalDatasetMultiT(
        root_dir=args['train_root'],
        treat_csv_path=args['treat_csv_path'],
        transform=train_transform,
        sequence_length=args['seq_len'],
        is_val=False,
        random_sample=True
    )
    t_scaler = train_dataset.t_scaler

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])
    val_dataset = VideoCausalDatasetMultiT(
        root_dir=args['val_root'],
        treat_csv_path=args['treat_csv_path'],
        transform=val_transform,
        sequence_length=args['seq_len'],
        is_val=True,
        random_sample=False,
        t_scaler=t_scaler
    )


    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader