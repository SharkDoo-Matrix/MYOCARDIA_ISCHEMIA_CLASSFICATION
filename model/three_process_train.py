import copy

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from .model_classifier import CrossAttentionClassifier, get_classifier
import pandas as pd


import train_loss


# train Step1: backbone
def train_step1_resnet(model, train_loader, val_loader, device, num_epochs=50, learning_rate=0.001, save_dir='models_new1_dataset'):
    model = model.to(device)

    os.makedirs(save_dir, exist_ok=True)

    criterion = set_criterion(train_dataset=train_loader.dataset, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (volumes, _, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            volumes, labels = volumes.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for volumes, _, labels in val_loader:
                volumes, labels = volumes.to(device), labels.to(device)
                outputs, _ = model(volumes)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy
            }, os.path.join(save_dir, 'best_resnet_model.pth'))
            print(f'New best model saved with accuracy: {accuracy:.2f}%')

    model.load_state_dict(best_model_wts)

    return model

# train Step2: CDFEM
def train_step2_full(full_model, train_loader, val_loader, device, num_epochs=20, learning_rate=0.0001, save_dir='models_new1_dataset', contrast=False):
    full_model = full_model.to(device)

    os.makedirs(save_dir, exist_ok=True)

    for param in full_model.parameters():
        param.requires_grad = True

    criterion = set_criterion(train_dataset=train_loader.dataset, device=device)
    optimizer = optim.Adam(full_model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []

    best_acc = 0.0
    best_model_wts = copy.deepcopy(full_model.state_dict())
    best_classification_report = ""

    for epoch in range(num_epochs):
        full_model.train()
        running_loss = 0.0
        contrast_loss = 0.0
        lamb = 0.05
        if epoch % 10 == 0 and lamb < 1.0:
            lamb = lamb + 0.1

        for i, (volumes, low_features, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            volumes, low_features, labels = volumes.to(device), low_features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, loss_ctr = full_model(volumes, low_features)
            loss = criterion(outputs, labels)
            if contrast:

                loss = loss + lamb * loss_ctr
                contrast_loss += loss_ctr.item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        full_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for volumes, low_features, labels in val_loader:
                volumes, low_features, labels = volumes.to(device), low_features.to(device), labels.to(device)
                outputs, _ = full_model(volumes, low_features)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        avg_loss = running_loss / len(train_loader)
        avg_contrast_loss = contrast_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)

        class_report = classification_report(all_labels, all_preds, zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
        print(f'contrast: {avg_contrast_loss:.4f}')
        print("Classification Report:")
        print(class_report)
        print("Confusion Matrix:")
        print(conf_matrix)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(full_model.state_dict())
            best_classification_report = class_report
            torch.save({
                'epoch': epoch,
                'model_state_dict': full_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            }, os.path.join(save_dir, 'best_full_model.pth'))
            print(f'New best model saved with accuracy: {accuracy:.2f}%')

    full_model.load_state_dict(best_model_wts)

    print("Best Model Classification Report:")
    print(best_classification_report)

    return full_model

def set_criterion(train_dataset, device):
    criterion = nn.CrossEntropyLoss()

    return criterion

def evaluate_model(model, val_loader, device, model_save_dir):
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    video_paths = []
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_attn_weights = [] 

    with torch.no_grad():
        for volumes, low_features, labels, video_path in val_loader:

            volumes, low_features, labels = (
                volumes.to(device),
                low_features.to(device),
                labels.to(device)
            )

            outputs, attn_weights = model(volumes, low_features)

        
            probabilities = torch.softmax(outputs, dim=1)  # 形状: [batch_size, num_classes]

            
            max_prob, predicted = torch.max(probabilities, dim=1)  # max_prob: [batch_size], predicted: [batch_size]


            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            video_paths.extend(np.array(video_path))  
            all_labels.extend(labels.cpu().numpy())  
            all_predictions.extend(predicted.cpu().numpy())  
            all_probabilities.extend(max_prob.cpu().numpy())  
            all_attn_weights.append(attn_weights.cpu())  

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1', 'Class 2']))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    cm = confusion_matrix(all_labels, all_predictions)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)

    for i in range(len(precision)):
        print(f'Class {i} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}')

    # sava results
    results_df = pd.DataFrame({
        "Sample_Name": video_paths,
        "True_Label": all_labels,
        "Predicted_Label": all_predictions,
        "Probability": all_probabilities,
    })
    results_df.to_csv(os.path.join(model_save_dir, "test_results.csv"), index=False)  # 不保存 DataFrame 索引

    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Average Precision: {np.mean(precision):.4f}")
    print(f"Average Recall: {np.mean(recall):.4f}")
    print(f"Average F1-score: {2 * np.mean(precision) * np.mean(recall) / (np.mean(precision) + np.mean(recall)):.4f}")

    return all_labels, all_predictions, all_attn_weights

def load_best_model(model_class, model_path, device, feature_model=None, pretrained=True, contrast=False, **kwargs):

    if feature_model is not None:

        model = model_class(feature_model, contrast=contrast, **kwargs)
    else:

        model = model_class(**kwargs)

    if pretrained:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        print(f"Loaded best model from {model_path}")
        print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")

    return model

def train_multi_t_model(args):
    train_loader = args['train_loader']
    val_loader = args['val_loader']

    device = args['device']

    resnet_model = args.get('backbone_model')
    full_model = args.get('full_model')

    stage1_epoch = args.get('stage1_epoch', 0)
    stage2_epoch = args.get('stage2_epoch', 0)

    model_backbone_dir = args.get('model_backbone_dir', 'models')
    model_save_dir = args.get('model_save_dir', 'models')
    os.makedirs(model_save_dir, exist_ok=True)

    resnet_model_path = os.path.join(model_backbone_dir, args.get('resnet_model_name', 'best_resnet_model.pth'))
    full_model_path = os.path.join(model_save_dir, args.get('full_model_name', 'best_full_model.pth'))

    # Step 1
    if stage1_epoch > 0:
        if resnet_model is None:
            raise ValueError("ResNet model is required for Stage 1 training but not provided.")
        print("Step 1: Training ResNet3D...")
        resnet_model = train_step1_resnet(
            resnet_model,
            train_loader,
            val_loader,
            device = device,
            num_epochs=stage1_epoch,
            save_dir=model_backbone_dir
        )
        args['resnet_model'] = resnet_model

    # Step 2
    print("Step 2: Training Attention Part...")
    best_resnet_model = load_best_model(
        args.get('backbone_class'),
        resnet_model_path,
        device = device,
        num_classes=3,
        in_channels=3
    )
    
    full_model = get_classifier(best_resnet_model, high_dim_size=2048, low_dim_size=13, num_classes=3)
    if stage2_epoch > 0:
        full_model = train_step2_full(
            full_model,
            train_loader,
            val_loader,
            device = device,
            num_epochs=stage2_epoch,
            save_dir=model_save_dir,
            contrast=args['contrast']
        )

    print("Final Evaluation...")
    
    # 
    best_full_model = load_best_model(
        CrossAttentionClassifier,
        full_model_path,
        feature_model=best_resnet_model,
        high_dim_size=2048,
        low_dim_size=13,
        num_classes=3,
        device=device,
    )

    test_labels, test_predictions, attention_weights = evaluate_model(best_full_model, val_loader, device, model_save_dir)

    from .evaluate import evaluate_single_attention
    evaluate_single_attention.visualize_attention_sample(full_model, val_loader, device, num_frames=10)

    if args.get('visualize_attention', False):
        plt.figure(figsize=(10, 6))
        plt.imshow(attention_weights[0].cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.title('Attention Weights')
        plt.colorbar()
        attention_plot_path = os.path.join(model_save_dir, args.get('attention_plot_name', 'attention_weights.png'))
        plt.savefig(attention_plot_path)
        if args.get('show_attention_plot', False):
            plt.show()
        plt.close()

    return full_model, test_labels, test_predictions, attention_weights

