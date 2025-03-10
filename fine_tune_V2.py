import segmentation_models_pytorch as smp
import numpy as np
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
import torchvision.transforms as transforms
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
import random
from tqdm import tqdm
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

name_classes = np.array(['non-fire','fire'], dtype=str)
epsilon = 1e-14

def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

# 新增：自定义CNN模型作为比较基准
class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CustomCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 新增：基于编码器特征的自定义架构
class EncoderFeatureModel(nn.Module):
    def __init__(self, encoder_model, num_classes=2):
        super(EncoderFeatureModel, self).__init__()
        self.encoder = encoder_model
        
        # 提取编码器各层特征时使用的钩子
        self.features = []
        
        # 用于处理提取的特征的自定义CNN层
        self.custom_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )
        
    def register_hooks(self):
        # 注册钩子来捕获特定层的输出
        self.features = []
        def hook(module, input, output):
            self.features.append(output)
        
        # 举例：为ResNet50的几个关键层注册钩子
        # 实际使用时需要根据具体模型结构调整
        self.encoder.layer1[2].conv3.register_forward_hook(hook)
        self.encoder.layer2[3].conv3.register_forward_hook(hook)
        self.encoder.layer3[5].conv3.register_forward_hook(hook)
        self.encoder.layer4[2].conv3.register_forward_hook(hook)
        
    def forward(self, x):
        # 清空之前的特征
        self.features = []
        
        # 通过编码器获取特征
        _ = self.encoder(x)
        
        # 使用最后一层特征作为输入
        x = self.features[-1]
        
        # 通过自定义层处理特征
        output = self.custom_layers(x)
        
        return output, self.features

def get_arguments():
    parser = argparse.ArgumentParser()
    
    # 新增：指定分割模型类型
    parser.add_argument("--model_type", type=str, default='unet', 
                      choices=['unet', 'deeplabv3plus', 'pspnet', 'fpn', 'customcnn', 'encoderfeature'],
                      help="Type of segmentation model to use.")
    parser.add_argument("--encoder_name", type=str, default='resnet50',
                      choices=['resnet50', 'resnet101', 'efficientnet-b4', 'timm-regnety_016', 'resnext50_32x4d'],
                      help="Name of encoder backbone to use.")
    parser.add_argument("--use_activation_maps", action='store_true',
                      help="Whether to visualize activation maps during validation.")
                      
    #dataset
    parser.add_argument("--freeze_epochs", type=int, default=3000,   
                    help="Number of steps (or epochs) before unfreezing encoder.")
    parser.add_argument("--data_dir", type=str, default='/Users/d5826/desktop/milestone/Sen2Fire/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--val_list", type=str, default='./dataset/val.txt',
                        help="val list file.")         
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")               
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")   
    # for this fine-tuning, use mode 2: RGB for now      
    parser.add_argument("--mode", type=int, default=4,
                        help="input type (0-all_bands, 1-all_bands_aerosol,...).")           

    #network
    parser.add_argument("--batch_size", type=int, default=8,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="base learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--weight", type=float, default=10,
                        help="ce weight.")
    # for fine-tuning, unfreeze step
    parser.add_argument("--unfreeze_step", type=int, default=2000,
                        help="Number of steps before unfreezing U Net Encoder weights.")
    
    # 新增：渐进式解冻策略
    parser.add_argument("--progressive_unfreeze", action='store_true',
                        help="Whether to progressively unfreeze encoder layers.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./Exp/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

modename = ['all_bands',                        #0
            'all_bands_aerosol',                #1
            'rgb',                              #2
            'rgb_aerosol',                      #3
            'swir',                             #4
            'swir_aerosol',                     #5
            'nbr',                              #6
            'nbr_aerosol',                      #7   
            'ndvi',                             #8
            'ndvi_aerosol',                     #9 
            'rgb_swir_nbr_ndvi',                #10
            'rgb_swir_nbr_ndvi_aerosol',]       #11

# 新增：获取模型方法
def get_model(args):
    if args.model_type == 'customcnn':
        return CustomCNN(in_channels=3, num_classes=args.num_classes)
    elif args.model_type == 'encoderfeature':
        # 先创建编码器模型
        base_model = smp.Unet(
            encoder_name=args.encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.num_classes,
            encoder_depth=5
        )
        # 创建基于编码器特征的模型
        model = EncoderFeatureModel(base_model.encoder, num_classes=args.num_classes)
        model.register_hooks()
        return model
    else:
        # 使用smp库中的模型
        if args.model_type == 'unet':
            return smp.Unet(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=args.num_classes
            )
        elif args.model_type == 'deeplabv3plus':
            return smp.DeepLabV3Plus(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=args.num_classes
            )
        elif args.model_type == 'pspnet':
            return smp.PSPNet(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=args.num_classes
            )
        elif args.model_type == 'fpn':
            return smp.FPN(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=args.num_classes
            )

# 新增：可视化Grad-CAM激活图
def visualize_activation_maps(model, image, target, save_path):
    # 使用GradCAM来获取激活图
    target_layers = [model.encoder.layer4[-1]]  # 对于ResNet，选择最后一层作为目标层
    
    # 为不同的模型结构调整目标层
    if isinstance(model, CustomCNN):
        target_layers = [model.encoder[-2]]
    elif isinstance(model, EncoderFeatureModel):
        target_layers = [model.encoder.layer4[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 获取类别激活图
    grayscale_cam = cam(input_tensor=image, target_category=1)  # 目标类别为"fire"
    
    # 将输入图像转换为可视化格式
    input_image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    # 标准化到[0,1]范围
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    
    # 将激活图叠加在输入图像上
    visualization = show_cam_on_image(input_image, grayscale_cam[0])
    
    # 创建子图显示原始图像、激活图和标签
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title('Activation Map')
    axes[1].axis('off')
    
    # 显示真值标签
    target_vis = target.squeeze().cpu().numpy()
    axes[2].imshow(target_vis, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 新增：可视化特征图函数
def visualize_feature_maps(features, save_path):
    # 选择要显示的特征图数量
    num_maps = min(16, features[-1].shape[1])
    
    # 创建画布
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    # 获取最后一层的特征图
    feature_map = features[-1].squeeze().cpu().detach().numpy()
    
    # 显示前num_maps个特征图
    for i in range(num_maps):
        feat_map = feature_map[i]
        # 标准化到[0,1]范围
        feat_map = (feat_map - feat_map.min()) / (max(feat_map.max() - feat_map.min(), 1e-8))
        axes[i].imshow(feat_map, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = get_arguments()
    snapshot_dir = args.snapshot_dir+'model_'+args.model_type+'_encoder_'+args.encoder_name+'_input_'+modename[args.mode]+'/weight_'+str(args.weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    
    # 创建可视化目录
    vis_dir = os.path.join(snapshot_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        
    f = open(snapshot_dir+'Training_log.txt', 'w')

    input_size_train = (512, 512)

    cudnn.enabled = True
    cudnn.benchmark = True
    init_seeds()

    # 获取选定的模型
    model = get_model(args)
    model = model.to(device)
    print(model)

    # Resnet expects a particular size
    # 创建归一化函数，稍后手动应用于图像而不是作为transform参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 修复: 移除 transform 参数
    train_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.train_list, max_iters=args.num_steps_stop*args.batch_size,
                    mode=args.mode),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.val_list, mode=args.mode),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    test_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.test_list, mode=args.mode),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # interpolation for the probability maps and labels 
    interp = nn.Upsample(size=(input_size_train[1], input_size_train[0]), mode='bilinear')
    
    hist = np.zeros((args.num_steps_stop,4))
    F1_best = 0.    

    class_weights = [1, args.weight]
    L_seg = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    
    # 添加Dice Loss和BCE Loss
    L_dice = smp.losses.DiceLoss(mode="binary")
    L_bce = nn.BCEWithLogitsLoss()

    # 根据模型类型确定参数组和冻结策略
    if args.model_type in ['customcnn']:
        # 自定义CNN不需要冻结
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.model_type == 'encoderfeature':
        # 冻结编码器，只训练自定义层
        for param in model.encoder.parameters():
            param.requires_grad = False
            
        optimizer = optim.Adam([
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.custom_layers.parameters(), "lr": args.learning_rate}
        ], weight_decay=args.weight_decay)
    else:
        # 对于预训练模型，冻结编码器
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
            
            optimizer = optim.Adam([
                {"params": model.encoder.parameters(), "lr": 1e-5},
                {"params": [p for n, p in model.named_parameters() if not n.startswith('encoder')], "lr": args.learning_rate}
            ], weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for batch_index, train_data in enumerate(train_loader):
        print(batch_index)
        if batch_index == args.num_steps_stop:
            break
        tem_time = time.time()
        adjust_learning_rate(optimizer, args.learning_rate, batch_index, args.num_steps)
        model.train()
        optimizer.zero_grad()
        
        patches, labels, _, _ = train_data
        # 手动应用归一化
        patches = patches.to(device)
        # 对每个batch中的每个图像应用归一化
        for i in range(patches.size(0)):
            patches[i] = normalize(patches[i])
          
        labels = labels.to(device).long()
        
        # 根据模型类型处理前向传播
        if args.model_type == 'encoderfeature':
            pred, features = model(patches)
        else:
            pred = model(patches)
            
        pred_interp = interp(pred)
        
        # 组合损失：交叉熵 + Dice Loss
        L_seg_value = L_seg(pred_interp, labels)
        L_dice_value = L_dice(F.softmax(pred_interp, dim=1)[:, 1:], labels == 1)
        
        # 组合损失
        loss = 0.7 * L_seg_value + 0.3 * L_dice_value
        
        _, predict_labels = torch.max(pred_interp, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = labels.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        batch_miou = np.nanmean(metrics_batch, axis=0)  
        batch_oa = np.sum(lbl_pred==lbl_true)*1./len(lbl_true.reshape(-1))
                    
        hist[batch_index,0] = loss.item()
        hist[batch_index,1] = batch_oa
        hist[batch_index,2] = batch_miou
        
        loss.backward()
        optimizer.step()

        hist[batch_index,-1] = time.time() - tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d Time: %.2f Batch_OA = %.2f Batch_mIoU = %.2f Loss = %.3f'%(batch_index+1,args.num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,2])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
            f.write('Iter %d/%d Time: %.2f Batch_OA = %.2f Batch_mIoU = %.2f Loss = %.3f\n'%(batch_index+1,args.num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,2])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
            f.flush() 
        
        # 处理解冻策略
        if args.model_type not in ['customcnn']:
            # 渐进式解冻
            if args.progressive_unfreeze:
                if batch_index == int(args.unfreeze_step * 0.25):
                    print("\nUnfreezing encoder layer 4...\n")
                    for param in model.encoder.layer4.parameters():
                        param.requires_grad = True
                
                elif batch_index == int(args.unfreeze_step * 0.5):
                    print("\nUnfreezing encoder layer 3...\n")
                    for param in model.encoder.layer3.parameters():
                        param.requires_grad = True
                
                elif batch_index == int(args.unfreeze_step * 0.75):
                    print("\nUnfreezing encoder layer 2...\n")
                    for param in model.encoder.layer2.parameters():
                        param.requires_grad = True
                
                elif batch_index == args.unfreeze_step:
                    print("\nUnfreezing all encoder layers...\n")
                    for param in model.encoder.parameters():
                        param.requires_grad = True
            # 常规解冻
            elif batch_index == args.unfreeze_step:
                print("\nUnfreezing encoder for fine-tuning...\n")
                if hasattr(model, 'encoder'):
                    for param in model.encoder.parameters():
                        param.requires_grad = True

        # evaluation per 500 iterations
        if (batch_index+1) % 500 == 0:            
            print('Validating..........')  
            f.write('Validating..........\n')  

            model.eval()
            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
            IoU = np.zeros((args.num_classes, 1))
            
            tbar = tqdm(val_loader)
            for idx, batch in enumerate(tbar):  
                image, label, _, _ = batch
                label = label.squeeze().numpy()
                image = image.float().to(device)
                
                # 手动应用归一化
                for i in range(image.size(0)):
                    image[i] = normalize(image[i])
                
                with torch.no_grad():
                    if args.model_type == 'encoderfeature':
                        pred, features = model(image)
                    else:
                        pred = model(image)
                
                # 可视化激活图和特征图
                if args.use_activation_maps and idx < 10:  # 只为前10个样本生成激活图
                    act_map_path = os.path.join(vis_dir, f'activation_map_{batch_index}_{idx}.png')
                    visualize_activation_maps(model, image, label, act_map_path)
                    
                    # 如果是EncoderFeature模型，可视化特征图
                    if args.model_type == 'encoderfeature':
                        feat_map_path = os.path.join(vis_dir, f'feature_map_{batch_index}_{idx}.png')
                        visualize_feature_maps(features, feat_map_path)

                _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                       
                               
                TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all
            for i in range(args.num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
            
                if i==1:
                    print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
                    print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
                    print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
                    print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))   
                    f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P * 100))
                    f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R * 100))            
                    f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i] * 100))              
                    f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i] * 100))   
                
            mF1 = np.mean(F1)   
            mIoU = np.mean(IoU)           
            print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
            f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))
                
            if F1[1]>F1_best:
                F1_best = F1[1]
                # save the models        
                print('Save Model')   
                f.write('Save Model\n')   
                model_name = 'best_model.pth'
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name))
    
    saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
    model.load_state_dict(saved_state_dict)

    print('Testing..........')  
    f.write('Testing..........\n')  

    model.eval()
    TP_all = np.zeros((args.num_classes, 1))
    FP_all = np.zeros((args.num_classes, 1))
    TN_all = np.zeros((args.num_classes, 1))
    FN_all = np.zeros((args.num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((args.num_classes, 1))
    IoU = np.zeros((args.num_classes, 1))
    
    tbar = tqdm(test_loader)
    for idx, batch in enumerate(tbar):  
        image, label, _, _ = batch
        label = label.squeeze().numpy()
        image = image.float().to(device)
        
        # 手动应用归一化
        for i in range(image.size(0)):
            image[i] = normalize(image[i])
        
        with torch.no_grad():
            if args.model_type == 'encoderfeature':
                pred, features = model(image)
            else:
                pred = model(image)

        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()                       
                        
        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(args.num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
        IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
    
        if i==1:
            print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
            print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
            print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
            print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))   
            f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P * 100))
            f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R * 100))            
            f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i] * 100))              
            f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i] * 100))   
        
    mF1 = np.mean(F1)   
    mIoU = np.mean(IoU)           
    print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
    f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))        
    f.close()
    saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
    np.savez(snapshot_dir+'Precision_'+str(int(P * 10000))+'Recall_'+str(int(R * 10000))+'F1_'+str(int(F1[1] * 10000))+'_hist.npz',hist=hist) 
    
if __name__ == '__main__':
    main()



