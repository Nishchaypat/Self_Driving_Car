# utilities
import os
import sys
import logging
import argparse
import datetime
import itertools
import torch

#tensorboard
from torch.utils.tensorboard import SummaryWriter

# dataset and preprocessing
from torch.utils.data import DataLoader, ConcatDataset
from vision.datasets.open_images import ImagesDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

# learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# networks
from vision.utils.misc import Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.config import mobilenetv1_ssd_config

#loss
from vision.nn.multibox_loss import MultiboxLoss

from evaluators import *

DEFAULT_PRETRAINED_MODEL='models/mobilenet-v1-ssd-mp-0_675.pth'

arg_params = {}

arg_params["dataset_type"] = "open_images"
arg_params["datasets"] = ["/scratch/Workshop/all_images"]
arg_params["balance_data"] = True
arg_params["net"] = "mb1-ssd"
arg_params["resolution"] = 300
arg_params["freeze_base_net"] = True
arg_params["freeze_net"] = True
arg_params["mb2_width_mult"] = 1.0
arg_params["base_net"] = ""
arg_params["pretrained_ssd"] = "models/mobilenet-v1-ssd-mp-0_675.pth "
arg_params["resume"] = "models/mb1-ssd-Epoch-99-Loss-7.836331605911255.pth"
arg_params["lr"] = 0.01
arg_params["momentum"] = 0.9
arg_params["weight_decay"] = 5e-4
arg_params["gamma"] = 0.1
arg_params["base_net_lr"] = 0.001
arg_params["extra_layers_lr"] = None
arg_params["scheduler"] = "cosine"
arg_params["milestones"] = "80,100"
arg_params["t_max"] = 100
arg_params["batch_size"] = 4
arg_params["num_epochs"] = 100
arg_params["num_workers"] = 2
arg_params["validation_epochs"] = 1
arg_params["validation_mean_ap"] = True
arg_params["debug_steps"] = 10
arg_params["use_cuda"] = True
arg_params["checkpoint_folder"] = "/scratch/sshrestha8/Workshop/Day4/saving_directories/checkpoints"
arg_params["log_level"] = "info"
arg_params["tensorboard"] = "/scratch/sshrestha8/Workshop/Day4/saving_directories/tensorboard"
# arg_params["data_root"] = "/scratch/sshrestha8/Workshop/pytorch-ssd/data/traffic"



class DotDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

args = DotDict(arg_params)


logging.basicConfig(stream=sys.stdout, level=getattr(logging, args.log_level.upper(), logging.INFO),
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

tensorboard = SummaryWriter(log_dir=os.path.join(args.tensorboard, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA...")

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    
    train_loss = 0.0
    train_regression_loss = 0.0
    train_classification_loss = 0.0
    
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    
    num_batches = 0
    
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_regression_loss += regression_loss.item()
        train_classification_loss += classification_loss.item()
        
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}/{len(loader)}, " +
                f"Avg Loss: {avg_loss:.4f}, " +
                f"Avg Regression Loss {avg_reg_loss:.4f}, " +
                f"Avg Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

        num_batches += 1
        
    train_loss /= num_batches
    train_regression_loss /= num_batches
    train_classification_loss /= num_batches
    
    logging.info(
        f"Epoch: {epoch}, " +
        f"Training Loss: {train_loss:.4f}, " +
        f"Training Regression Loss {train_regression_loss:.4f}, " +
        f"Training Classification Loss: {train_classification_loss:.4f}"
    )
     
    tensorboard.add_scalar('Loss/train', train_loss, epoch)
    tensorboard.add_scalar('Regression Loss/train', train_regression_loss, epoch)
    tensorboard.add_scalar('Classification Loss/train', train_classification_loss, epoch)

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    
    # make sure that the checkpoint output dir exists
    if args.checkpoint_folder:
        args.checkpoint_folder = os.path.expanduser(args.checkpoint_folder)

        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
            
    # select the network architecture and config     
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
        config.set_image_size(args.resolution)
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # create data transforms for train/test/val
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    # load datasets (could be multiple)
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'open_images':

            dataset = ImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
        
    # create training dataset
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
                           
    # create validation dataset                           
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == 'open_images':
        val_dataset = ImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    logging.info("Validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
                      
    # create the network
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    # prepare eval dataset (for mAP computation)
    # if args.validation_mean_ap:
    #     if args.dataset_type == 'open_images':
    #         eval_dataset = ImagesDataset(dataset_path, dataset_type="test")
    #     eval = MeanAPEvaluator(eval_dataset, net, arch=args.net, eval_dir=os.path.join(args.checkpoint_folder, 'eval_results'))
        
    # freeze certain layers (if requested)
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    # load a previous model checkpoint (if requested)
    timer.start("Load Model")
    
    if args.resume:
        logging.info(f"Resuming from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained SSD {args.pretrained_ssd}")
        
        if not os.path.exists(args.pretrained_ssd) and args.pretrained_ssd == DEFAULT_PRETRAINED_MODEL:
            os.system(f"wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O {DEFAULT_PRETRAINED_MODEL}")

        net.init_from_pretrained_ssd(args.pretrained_ssd)
        
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    # move the model to GPU
    net.to(DEVICE)

    # define loss function and optimizer
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
                             
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
                                
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    # set learning rate policy
    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # train for the desired number of epochs
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    
    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
                    
            tensorboard.add_scalar('Loss/val', val_loss, epoch)
            tensorboard.add_scalar('Regression Loss/val', val_regression_loss, epoch)
            tensorboard.add_scalar('Classification Loss/val', val_classification_loss, epoch)
    
    
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")

    logging.info("Task done, exiting program.")
    tensorboard.close()
