"""
Architecture Med3D ResNet3D-18 avec Transfer Learning
Adapté pour la détection d'anévrismes intracrâniens

Basé sur: https://github.com/Tencent/MedicalNet

Auteur: Romain
Date: Octobre 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict

# ============================================================================
# CONFIGURATION
# ============================================================================

if sys.platform == 'win32':
    PROJECT_DIR = r"E:\Etudes\M2_UPC\Intelligence_Artificielle\projet_kaggle"
else:
    PROJECT_DIR = "/mnt/e/Etudes/M2_UPC/Intelligence_Artificielle/projet_kaggle"

PRETRAINED_PATH = os.path.join(PROJECT_DIR, "pretrained_models", "resnet_18_23dataset.pth")


# ============================================================================
# BLOCS RÉSIDUELS 3D (compatible Med3D)
# ============================================================================

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution avec padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    """
    Bloc résiduel basique pour ResNet-18
    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


# ============================================================================
# RESNET 3D
# ============================================================================

class ResNet3D(nn.Module):
    """
    ResNet 3D pour imagerie médicale
    Compatible avec les weights Med3D pré-entraînés
    """
    
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type='B',
        widen_factor=1.0,
        n_classes=1
    ):
        super(ResNet3D, self).__init__()
        
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        
        # Stem
        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Stages résiduel
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        
        # Global pooling et classification
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        
        # Initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        
        out = torch.cat([out.data, zero_pads], dim=1)
        return out
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = lambda x: self._downsample_basic_block(x, planes * block.expansion, stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion)
                )
        
        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        
        # Stages résiduel
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling et classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# ============================================================================
# FONCTION DE CRÉATION DU MODÈLE
# ============================================================================

def generate_resnet18_med3d(n_classes=1, n_input_channels=1, **kwargs):
    """
    Crée un ResNet3D-18 compatible avec Med3D
    
    Args:
        n_classes: Nombre de classes (1 pour binary classification)
        n_input_channels: Nombre de channels d'input (1 pour grayscale)
    
    Returns:
        model: ResNet3D-18
    """
    model = ResNet3D(
        BasicBlock,
        [2, 2, 2, 2],  # ResNet-18: [2,2,2,2] blocs par stage
        block_inplanes=[64, 128, 256, 512],
        n_input_channels=n_input_channels,
        n_classes=n_classes,
        **kwargs
    )
    return model


# ============================================================================
# CHARGEMENT DES WEIGHTS PRÉ-ENTRAÎNÉS
# ============================================================================

def load_pretrained_weights(model, pretrained_path=PRETRAINED_PATH, strict=False):
    """
    Charge les weights Med3D pré-entraînés
    
    Args:
        model: Modèle ResNet3D
        pretrained_path: Chemin vers le fichier .pth
        strict: Si False, ignore les layers incompatibles (comme fc)
    
    Returns:
        model: Modèle avec weights chargés
    """
    
    print(f"\n📥 Chargement des weights pré-entraînés...")
    print(f"   Fichier: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        print(f"   ❌ Fichier introuvable!")
        print(f"   💡 Lance d'abord: python install_med3d.py")
        return model
    
    # Charger le checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extraire le state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Nettoyer les noms des keys (enlever 'module.' si présent)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    # Supprimer la dernière couche fc (incompatible)
    # Med3D a fc avec 400 classes, nous on veut 1 classe
    if 'fc.weight' in new_state_dict:
        del new_state_dict['fc.weight']
    if 'fc.bias' in new_state_dict:
        del new_state_dict['fc.bias']
    
    # Charger les weights (strict=False pour ignorer fc)
    model.load_state_dict(new_state_dict, strict=False)
    
    print(f"   ✅ Weights pré-entraînés chargés")
    print(f"   ⚠️  Dernière couche (fc) initialisée aléatoirement")
    
    return model


# ============================================================================
# FREEZE/UNFREEZE DES COUCHES
# ============================================================================

def freeze_layers(model, freeze_until='layer3'):
    """
    Freeze les couches pour fine-tuning
    
    Args:
        model: ResNet3D
        freeze_until: 'layer1', 'layer2', 'layer3', ou 'layer4'
                      Freeze tout jusqu'à (inclus) cette layer
    
    Stratégie:
        - Freeze stem + layer1 + layer2 + layer3 (features générales)
        - Unfreeze layer4 + fc (features spécifiques anévrismes)
    """
    
    print(f"\n🔒 Freeze des couches jusqu'à {freeze_until}...")
    
    # Freeze stem
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    
    # Freeze layers
    layers_to_freeze = []
    if freeze_until == 'layer1':
        layers_to_freeze = ['layer1']
    elif freeze_until == 'layer2':
        layers_to_freeze = ['layer1', 'layer2']
    elif freeze_until == 'layer3':
        layers_to_freeze = ['layer1', 'layer2', 'layer3']
    elif freeze_until == 'layer4':
        layers_to_freeze = ['layer1', 'layer2', 'layer3', 'layer4']
    
    for layer_name in layers_to_freeze:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = False
        print(f"   🔒 {layer_name} frozen")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n   📊 Paramètres:")
    print(f"      Total: {total_params:,}")
    print(f"      Entraînables: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"      Frozen: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)")


def unfreeze_all(model):
    """
    Unfreeze toutes les couches
    """
    print(f"\n🔓 Unfreeze de toutes les couches...")
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ {trainable_params:,} paramètres entraînables")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def build_med3d_model(pretrained=True, freeze_until='layer3', n_classes=1):
    """
    Construit le modèle Med3D complet avec transfer learning
    
    Args:
        pretrained: Charger les weights pré-entraînés
        freeze_until: Jusqu'où freeze ('layer1', 'layer2', 'layer3', 'layer4')
        n_classes: Nombre de classes de sortie
    
    Returns:
        model: ResNet3D-18 prêt pour fine-tuning
    """
    
    print("=" * 80)
    print("🧠 CRÉATION DU MODÈLE MED3D RESNET3D-18")
    print("=" * 80)
    
    # Créer le modèle
    model = generate_resnet18_med3d(n_classes=n_classes, n_input_channels=1)
    
    print(f"\n✅ Architecture créée")
    print(f"   Type: ResNet3D-18")
    print(f"   Input: (batch, 1, D, H, W)")
    print(f"   Output: (batch, {n_classes})")
    
    # Charger les weights pré-entraînés
    if pretrained:
        model = load_pretrained_weights(model)
    
    # Freeze des couches
    if pretrained and freeze_until is not None:
        freeze_layers(model, freeze_until=freeze_until)
    
    # Ajouter sigmoid à la fin pour binary classification
    if n_classes == 1:
        model.sigmoid = nn.Sigmoid()
        print(f"\n✅ Sigmoid ajouté pour binary classification")
    
    print("\n" + "=" * 80)
    
    return model


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    
    print("🧪 Test de l'architecture Med3D\n")
    
    # Créer le modèle
    model = build_med3d_model(pretrained=True, freeze_until='layer3')
    
    # Test avec un batch fictif
    print("\n🔍 Test avec un batch fictif...")
    batch_size = 2
    fake_input = torch.randn(batch_size, 1, 96, 192, 192)
    
    print(f"   Input shape: {fake_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(fake_input)
        output_sigmoid = model.sigmoid(output)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output (logits): {output.squeeze()}")
    print(f"   Output (sigmoid): {output_sigmoid.squeeze()}")
    
    print("\n✅ Architecture testée avec succès !")
