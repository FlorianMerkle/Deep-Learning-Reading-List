import torch
import torchvision

def load_imagenette(path, bs=32):
    train_transforms = torchvision.transforms.Compose([
        #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        #torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.RandomRotation(20),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transforms= torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_path=path
    imagenette_train = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=train_transforms
    )
    val_path=path
    imagenette_val = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=val_transforms
    )

    train_loader = torch.utils.data.DataLoader(imagenette_train,
                                              batch_size=bs,
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(imagenette_val,
                                              batch_size=bs,
                                              shuffle=True)
    return train_loader, val_loader

def load_torchvision_dataset(dataset, batchsize=512, data_augmentation=False):
    if data_augmentation == True:
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(32+6),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),        
        ])
    if data_augmentation == False:
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),        
        ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    if dataset == 'MNIST':
        train = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test = torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transforms, download=True)
        test = torchvision.datasets.CIFAR10('./data', train=False, transform=val_transforms, download=True)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batchsize,
        pin_memory=True,
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batchsize,
        pin_memory=True,
        shuffle=False,
        num_workers=2
    )
    return train_loader, test_loader