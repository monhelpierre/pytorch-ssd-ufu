import os
import cv2
import torch
import numpy as np
import random
from math import floor
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, build_model, nms
from utils.constants import COLOR
import datetime
from utils.metrics import AveragePrecision
import argparse
import logging
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-db", "--dataset", required=False, default="C:/datasets/", help="Link to database")
ap.add_argument("-mi", "--model", required=False, help="Model index [0, 1, 2]")
ap.add_argument("-vp", "--video", required=False, help="Video path to make detection")
ap.add_argument("-ip", "--image", required=False, help="Image path to make detection")
ap.add_argument("-sp", "--save", required=False, help="Path to save detection")
ap.add_argument("-map", "--precision", required=False, help="Path to save detection")
args = ap.parse_args()
       
def calulate_mAP(model, dataloader, cfg, label_names, device, no_amp=True):
    metric = AveragePrecision(len(label_names), cfg.recall_steps)
    metric.reset()
    pbar = tqdm(dataloader, bar_format="{l_bar}{bar:30}{r_bar}")
    with torch.no_grad():
        for (images, true_boxes, true_classes, difficulties) in pbar:
            images = images.to(device)
            true_boxes = [x.to(device) for x in true_boxes]
            true_classes = [x.to(device) for x in true_classes]
            difficulties = [x.to(device) for x in difficulties]
            with autocast(enabled=(not no_amp)):
                preds = model(images)
            det_boxes, det_scores, det_classes = nms(*model.decode(preds))
            metric.update(det_boxes, det_scores, det_classes, true_boxes, true_classes, difficulties)    
    APs = metric.result
    print("ID   Class            AP@[0.5]     AP@[0.5:0.95]")
    for i, (ap, name) in enumerate(zip(APs, label_names)):
        print("%-5d%-20s%-13.3f%.3f" % (i, name, ap[0], ap.mean()))
    print("mAP@[0.5]: %.3f" % APs[:, 0].mean())
    print("mAP@[0.5:0.95]: %.3f" % APs.mean())

def read_image(cfg, image_path=None, frame=None):
    size = (cfg.input_size, cfg.input_size)
    if image_path == None:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize(size)
    else:
        image = Image.open(image_path)
        image = image.resize(size)  
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(device)
    return image

def detect_from_video(cfg, video_path, save_path, model, logging, threshold, no_amp=True, fps = 300):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_file = save_path + 'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (cfg.input_size, cfg.input_size))  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, nb_found, _ = model.detect('Video frame', read_image(cfg, frame=frame), label_names, logging, threshold, no_amp)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        if nb_found > 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            frame = cv2.convertScaleAbs(frame, alpha=(255.0)) 
            video_writer.write(frame)
        frame_count -= 1
        seconds = round(frame_count / fps)
        cv2.putText(frame, f'FPS: {int(fps)}   {datetime.timedelta(seconds=seconds)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2) 
        cv2.imshow('Video', frame)
        if cv2.waitKey(int(1000 / fps) ) & 0xFF == ord('q'):
            break
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

def get_split_test(dataset_path, testing_file):
    with open(dataset_path + testing_file) as f:
        ids = [line.strip() for line in f.readlines()]
    return [dataset_path + 'images/' + id + '.png' for id in ids]       

def convert_to_onnx(model, size, path):
    model.eval()
    dummy_input = torch.randn(1, 3, size, size) 
    onnx_path = path + "/model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path)
        
def process_image(cfg, image_path, save_path, label_names, logging):
    image_name = image_path.split('/')[-1]
    image, nb_found, _ = model.detect(image_name, read_image(cfg, image_path=image_path), label_names, logging)        
    if nb_found > 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = save_path + 'result-' + image_name.split('/')[-1]
        image = cv2.convertScaleAbs(image, alpha=(255.0))
        Image.fromarray(image).save(filename)

#=========================================================================================================

if __name__ == '__main__':
    workers = 4
    device = 'cpu'
    no_amp = True
    threshold = 0.5
    
    root = os.getcwd() + '/'
    results_path = root + 'results/'
    config_path = root + 'configs/'
    test_path = root + 'test/images/'
    log_path = root + 'logs/'
    onedrivepath = 'C:/Users/monhe/OneDrive - Universidade Federal de Uberlândia/test/'
    model_names = [x.split('.')[0] for x in os.listdir(config_path) if x.__contains__('yaml')]
    label_names = [
        '000', '001', '003', '004', '007', '008','009','023', 
        '025', '028', '035', '040', '042', '051', '052', '053'
    ]
    
    test_json = args.dataset + 'test.json'
    images_path_list = get_split_test(args.dataset, 'divide/test.txt')
    filename = log_path + 'testing_' + (str(datetime.datetime.now()).split('.')[0]).replace(':', '_')
    
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
              
    for model_name in model_names:
        if args.model:
            if model_name != model_names[int(args.model)]:
                continue
            
        #print('Number of parameters : ' + str(sum(p.numel() for p in model.parameters())))
        #print('Number of trainable parameters : ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            
        cfg = config_path + f'{model_name}.yaml'

        if os.path.exists(cfg):
            pth = results_path + f'{model_name}/best.pth'
            cfg = load_config(cfg)
            model = build_model(cfg, label_names)
            model.to(device)
            model.eval()

            if os.path.exists(pth):
                print('Loaded from pretrained model')
                model.load_state_dict(torch.load(pth)['model_state_dict'])
            
            logging.info("=-------------------")
            logging.info(model_name)
            print(model_name)
            logging.info("=-------------------\n")

            if args.save:
                save_path = args.save + '/' + model_name + '/'
            else:
                save_path = onedrivepath + 'image/' + model_name + '/' 

            if args.video:
                if os.path.exists(args.video):
                    print('Detection from video.')
                    save_path = save_path.replace('/image/', '/video/')
                    detect_from_video(cfg, args.video, save_path, model, logging, threshold, no_amp)
            elif args.image:
                print('Detection from single image.')
                process_image(cfg, args.image, save_path, label_names, logging)
            elif args.precision:
                calulate_mAP(model, dataloader, cfg, label_names, device)
            else:
                print('Detection from test set images.')
                dataloader = create_dataloader(
                    test_json,
                    batch_size=cfg.batch_size,
                    image_size=cfg.input_size,
                    image_mean=cfg.image_mean,
                    image_stddev=cfg.image_stddev,
                    num_workers=workers
                )
                for image_path in images_path_list:
                    process_image(cfg, image_path, save_path, label_names, logging)
        else:
            print(f"Please check if conf file and save path exist for {model_name}")
            
#python test.py -mi 0 -dataset "C:/datasets/"
#python test.py -mi 0 --video "C:/Users/monhe/Videos/4K Video Downloader/DRIVING IN BRAZIL Corupá-SC to São Francisco do Sul-SC.mp4" --save "C:/Users/monhe/OneDrive/Desktop"
#python test.py -mi 0 --image "C:/Users/monhe/OneDrive/Pictures/test6.jpg" --save "C:/Users/monhe/OneDrive/Desktop"
#python test.py -mi 0 --video "C:/Users/monhe/OneDrive/Desktop/videos/01 Brazilian traffic laws and signs. My Brazilian Friends driving in São Paulo City SP.mp4" --save "C:/Users/monhe/OneDrive/Desktop"
#python test.py -mi 0 --video "C:/Users/monhe/OneDrive/Downloads/Sao Paulo 4K - Driving Downtown - Brazil.mp4"
#python test.py -mi 0 --video "C:/Users/monhe/OneDrive/Downloads/1.mp4" --save "C:/Users/monhe/OneDrive/Downloads/"