import os
import torch
import warnings
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, build_model, nms
from utils.metrics import Mean, AveragePrecision
import random
import datetime
import matplotlib.pyplot as plt
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, unnormalize
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-db", "--dataset", required=False, default="C:/Users/monhe/Downloads/datasets/", help="Link to database")
ap.add_argument("-mi", "--model", required=False, help="Model index [0, 1, 2]")
ap.add_argument("-is", "--img_size", required=False, default=320)
ap.add_argument("-bs", "--batch_size", required=False, default=32)
args = ap.parse_args()

label_names = [
    '000', '001', '003', '004', '007', '008','009','023', 
    '025', '028', '035', '040', '042', '051', '052', '053'
]

if args.model:
    if int(args.model) < 0 or int(args.model) > 2:
        raise ValueError("Model index should be 0, 1 or 2")
class PrepareDataset():
    def __init__(self, dataset_path):
        self.root = dataset_path
        self.splits = ['train', 'val', 'test']
        self.start()

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        file_root = tree.getroot()
        boxes, classes, difficulties = [], [], []

        for object in file_root.iter('object'):
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text) - 1
            ymin = int(bndbox.find('ymin').text) - 1
            xmax = int(bndbox.find('xmax').text) - 1
            ymax = int(bndbox.find('ymax').text) - 1
            boxes.append([xmin, ymin, xmax, ymax])

            label = object.find('name').text.lower().strip()
            classes.append(label)

            difficulty = int(object.find('difficult').text == '1')
            difficulties.append(difficulty)
        return boxes, classes, difficulties

    def save_as_json(self, basename, dataset):
        filename = os.path.join(os.path.dirname(__file__), basename)
        print("Saving %s ..." % filename)
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)

    def start(self):
        for split in self.splits:
            json_file = self.root + f'{split}.json'
            if not os.path.exists(json_file):
                dataset = []
                training_file = os.path.join(self.root, f'divide/{split}.txt')
                with open(training_file) as f:
                    ids = [line.strip() for line in f.readlines()]
                for id in tqdm(ids, desc=f"{split}"):
                    image_path = os.path.join(self.root, 'images', id + '.jpg')
                    annotation_path = os.path.join(self.root, 'annotations', id + '.xml')
                    boxes, classes, difficulties = self.parse_annotation(annotation_path)
                    classes = [label_names.index(c) for c in classes]
                    dataset.append(
                        {
                            'image': os.path.abspath(image_path),
                            'boxes': boxes,
                            'classes': classes,
                            'difficulties': difficulties
                        }
                    )
                self.save_as_json(json_file, dataset)
        print("Dataset Complete.")

class CheckpointManager(object):
    def __init__(self, logdir, model, optim, scaler, scheduler, best_score):
        self.epoch = 0
        self.logdir = logdir
        self.model = model
        self.optim = optim
        self.scaler = scaler
        self.scheduler = scheduler
        self.best_score = best_score

    def save(self, filename):
        data = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_score': self.best_score
        }
        torch.save(data, os.path.join(self.logdir, filename))

    def restore(self, filename):
        data = torch.load(os.path.join(self.logdir, filename))
        self.model.load_state_dict(data['model_state_dict'])
        self.optim.load_state_dict(data['optim_state_dict'])
        self.scaler.load_state_dict(data['scaler_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
        self.epoch = data['epoch']
        self.best_score = data['best_score']

    def restore_lastest_checkpoint(self):
        if os.path.exists(os.path.join(self.logdir, 'last.pth')):
            self.restore('last.pth')
            print("Restore the last checkpoint.")

def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']

def train_step(images, true_boxes, true_classes, difficulties, model, optim, amp, scaler, metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]
    difficulties = [x.to(device) for x in difficulties]
    optim.zero_grad()
    with autocast(enabled=amp):
        preds = model(images)
        loss = model.compute_loss(preds, true_boxes, true_classes)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])
    det_boxes, det_scores, det_classes = nms(*model.decode(preds))
    metrics['APs'].update(det_boxes, det_scores, det_classes, true_boxes, true_classes, difficulties, True)

def test_step(images, true_boxes, true_classes, difficulties, model, amp, metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]
    difficulties = [x.to(device) for x in difficulties]
    with autocast(enabled=amp):
        preds = model(images)
        loss = model.compute_loss(preds, true_boxes, true_classes)
    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])
    det_boxes, det_scores, det_classes = nms(*model.decode(preds))
    metrics['APs'].update(det_boxes, det_scores, det_classes, true_boxes, true_classes, difficulties)

def train_model(input_size, config_path, results_path, model_name, device, train_json, val_json, label_names, batch_size):
    cfg = config_path + f'{model_name}.yaml'
    workers = 4
    resume = True
    no_amp = 1
    val_period = 1
    
    cfg = load_config(cfg)
    enable_amp = (not no_amp)
    logdir = results_path + f'{input_size}/{model_name}/'
    
    if os.path.exists(logdir) and (not resume):
        raise ValueError("Log directory %s already exists. Specify --resume "
                         "in command line if you want to resume the training."
                         % logdir)

    model = build_model(cfg, input_size, label_names, device)
    model.to(device)
    
    train_loader = create_dataloader(
        train_json,
        batch_size = batch_size,
        image_size = input_size,
        image_mean = cfg.image_mean,
        image_stddev = cfg.image_stddev,
        augment = True,
        shuffle = True,
        num_workers = workers)
    
    val_loader = create_dataloader(
        val_json,
        batch_size = batch_size,
        image_size = input_size,
        image_mean = cfg.image_mean,
        image_stddev = cfg.image_stddev,
        num_workers = workers)

    # Criteria
    optim = getattr(torch.optim, cfg.optim.pop('name'))(model.parameters(), **cfg.optim)
    scaler = GradScaler(enabled=enable_amp)
    scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.pop('name'))(
        optim,
        **cfg.scheduler
    )
    metrics = {
        'loss': Mean(device),
        'APs': AveragePrecision(len(label_names), cfg.recall_steps, device)
    }

    # Checkpointing
    ckpt = CheckpointManager(logdir, model=model, optim=optim, scaler=scaler, scheduler=scheduler, best_score=0.)
    ckpt.restore_lastest_checkpoint()

    # TensorBoard writers
    writers = {
        'train': SummaryWriter(os.path.join(logdir, 'train')),
        'val': SummaryWriter(os.path.join(logdir, 'val'))
    }

    # Kick off
    print("=" * 20)
    print(model_name.upper())
    for epoch in range(ckpt.epoch + 1, cfg.epochs + 1):
        print("-" * 20)
        print(f"Epoch: {epoch}/{cfg.epochs} ({str(datetime.datetime.now()).split('.')[0]})")
        
        # Train
        model.train()
        metrics['loss'].reset()
        metrics['APs'].reset()
        if epoch == 1:
            warnings.filterwarnings(
                'ignore',
                ".*call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`.*"  # noqa: W605
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=0.001,
                total_iters=min(1000, len(train_loader))
            )
        pbar = tqdm(train_loader, bar_format="{l_bar}{bar:20}{r_bar}", desc="Training")
        for (images, true_boxes, true_classes, difficulties) in pbar:
            train_step(images,
                true_boxes,
                true_classes,
                difficulties = difficulties,
                model = model,
                optim = optim,
                amp = enable_amp,
                scaler = scaler,
                metrics = metrics,
                device = device)
            lr = get_lr(optim)
            pbar.set_postfix(loss='%.5f' % metrics['loss'].result, lr=lr)

            if epoch == 1:
                warmup_scheduler.step()
        APs = metrics['APs'].result
        mAP50 = APs[:, 0].mean()
        mAP = APs.mean()
        if mAP > ckpt.best_score:
            ckpt.best_score = mAP
        print("mAP@[0.5]: %.3f" % mAP50)
        print("mAP@[0.5:0.95]: %.3f (best: %.3f)" % (mAP, ckpt.best_score))
        writers['train'].add_scalar('Loss', metrics['loss'].result, epoch)
        writers['train'].add_scalar('Learning rate', get_lr(optim), epoch)
        writers['train'].add_scalar('mAP@[0.5]', mAP50, epoch)
        writers['train'].add_scalar('mAP@[0.5:0.95]', mAP, epoch)
        scheduler.step()
        
        # Validation
        if epoch % val_period == 0:
            model.eval()
            ckpt.best_score = 0.00
            metrics['loss'].reset()
            metrics['APs'].reset()
            pbar = tqdm(val_loader, bar_format="{l_bar}{bar:20}{r_bar}", desc=f"Validation ({str(datetime.datetime.now()).split('.')[0]})")
            with torch.no_grad():
                for (images, true_boxes, true_classes, difficulties) in pbar:
                    test_step(images,
                        true_boxes,
                        true_classes,
                        difficulties,
                        model = model,
                        amp = enable_amp,
                        metrics = metrics,
                        device = device
                    )
                    pbar.set_postfix(loss='%.5f' % metrics['loss'].result)
            APs = metrics['APs'].result
            mAP50 = APs[:, 0].mean()
            mAP = APs.mean()
            if mAP > ckpt.best_score:
                ckpt.best_score = mAP
                ckpt.save('best.pth')
            print("mAP@[0.5]: %.3f" % mAP50)
            print("mAP@[0.5:0.95]: %.3f (best: %.3f)" % (mAP, ckpt.best_score))
            writers['val'].add_scalar('Loss', metrics['loss'].result, epoch)
            writers['val'].add_scalar('Learning rate', get_lr(optim), epoch)
            writers['val'].add_scalar('mAP@[0.5]', mAP50, epoch)
            writers['val'].add_scalar('mAP@[0.5:0.95]', mAP, epoch)
        ckpt.epoch += 1
        ckpt.save('last.pth')

    writers['train'].close()
    writers['val'].close()

if __name__ == '__main__':
    root = os.getcwd().replace('\\', '/') + '/'
    config_path = root + 'configs/'
    model_names = [x.split('.')[0] for x in os.listdir(config_path) if x.__contains__('yaml')]
    
    device = 'cpu'
    FROM_DEFAULT_SIZE = False
    FROM_DEFAULT_BATCH = False
    input_sizes = [128, 256, 320, 512]
    batch_sizes = [8, 16, 32, 64]

    train_json = args.dataset + 'train.json'
    val_json = args.dataset + 'val.json'
    
    for img_size in input_sizes:
        if FROM_DEFAULT_SIZE: 
            if img_size != int(args.img_size):
                continue
        print('IMAGE SIZE : ' + str(img_size) + 'x' + str(img_size))

        for batch_size in batch_sizes:
            if FROM_DEFAULT_BATCH:
                if batch_size != int(args.batch_size):
                    continue
            print('BATCH SIZE : ' + str(batch_size))
            results_path = f'results{batch_size}/'

            if len([x for x in os.listdir(args.dataset) if x.__contains__('.json')]) != 3:
                PrepareDataset(args.dataset)            
            
            for model_name in model_names:
                if args.model: 
                    if model_name != model_names[int(args.model)]:
                        continue
            
                train_model(
                    img_size,
                    config_path, 
                    results_path, 
                    model_name, 
                    device, 
                    train_json, 
                    val_json, 
                    label_names,
                    batch_size
                )