import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
import os

def visualize_3d_attention(model, video_frames, t_feature, device, save_path='video_attention.gif'):
    model.eval()
    model.to(device)

    video_frames = video_frames.unsqueeze(0).to(device)  # 添加批次维度 [1, C, T, H, W]
    t_feature = t_feature.unsqueeze(0).to(device)  # [1, 13]

    with torch.no_grad():
        outputs, attn_weights = model(video_frames, t_feature)

        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)
            attn_weights = attn_weights.squeeze()

        attn_normalized = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)
        attn_value = attn_normalized.cpu().numpy()

        frames = video_frames.squeeze(0).cpu().numpy()  # [C, T, H, W]
        frames = frames.transpose(1, 0, 2, 3)  # [T, C, H, W]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if frames.shape[1] == 3:  # RGB
            img = frames[0].transpose(1, 2, 0)
        else:
            img = frames[0][0]

        im1 = ax1.imshow(img)
        ax1.set_title('Video Frame')
        ax1.axis('off')

        h, w = img.shape[:2]
        attention_map = np.ones((h, w)) * attn_value
        im2 = ax2.imshow(attention_map, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Attention Map (Value: {attn_value:.4f})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        frame_text = ax1.text(0.05, 0.95, f'Frame 0/{frames.shape[0] - 1}',
                              transform=ax1.transAxes, color='white',
                              fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        def update(frame_idx):
            if frames.shape[1] == 3:  # RGB
                img = frames[frame_idx].transpose(1, 2, 0)
            else:
                img = frames[frame_idx][0]

            im1.set_array(img)

            frame_text.set_text(f'Frame {frame_idx}/{frames.shape[0] - 1}')

            return im1, frame_text

        ani = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)

        ani.save(save_path, writer='pillow', fps=5)
        plt.close()

        pred = torch.argmax(outputs, dim=1)
        print(f"Predicted class: {pred.item()}")
        print(f"Attention weight: {attn_weights.item():.4f}")
        print(f"Animation saved to: {save_path}")

        return attn_weights.item(), ani


# 使用示例
def visualize_single_video_attention(model, dataset, idx, device, save_dir='attention_visualizations'):
    os.makedirs(save_dir, exist_ok=True)

    if dataset.is_test:
        frames, t, label, video_path = dataset[idx]
        video_name = os.path.basename(video_path)
    else:
        frames, t, label = dataset[idx]
        video_name = f"sample_{idx}"

    save_path = os.path.join(save_dir, f"{video_name}_attention.gif")
    attn_value, animation = visualize_3d_attention(model, frames, t, device, save_path)

    return attn_value, animation


import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_attention_sample(model, loader, device, num_frames=10, sample_index=0):
    model.eval()

    frames, t, label = next(iter(loader))
    frames, t, label = frames.to(device), t.to(device), label.to(device)

    frames_0 = frames[sample_index].unsqueeze(0)  # [1, C, T, H, W]
    t_0 = t[sample_index].unsqueeze(0)            # [1, 13]
    label_0 = label[sample_index].unsqueeze(0)    # [1]

    with torch.no_grad():
        output_0, attn_weights_0 = model(frames_0, t_0)

    true_label = label_0.item()
    pred_label = output_0.argmax(dim=1).item()
    print(f"sample {sample_index}: true label = {true_label}, pred = {pred_label}")

    attn_map = attn_weights_0[0].mean(dim=0).squeeze(-1).cpu().numpy()  # [T]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    fig, axes = plt.subplots(1, num_frames, figsize=(2 * num_frames, 4))
    for i in range(num_frames):
        img = frames_0[0, :, i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        img = (img - img.min()) / (img.max() - img.min())

        gray = np.mean(img, axis=-1)
        heat = gray * attn_map[i]

        axes[i].imshow(heat, cmap="gray")
        axes[i].axis("off")

    plt.suptitle(f"Attention Map - True: {true_label}, Pred: {pred_label}", fontsize=16)
    plt.show()
