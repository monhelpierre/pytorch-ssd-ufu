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
ap.add_argument("-db", "--dataset", required=False, default="C:/Users/monhe/Downloads/datasets/", help="Link to database")
ap.add_argument("-mi", "--model", required=False, help="Model index [0, 1, 2]")
ap.add_argument("-is", "--img_size", required=False, default=320)
ap.add_argument("-bs", "--batch_size", required=False, default=32)
ap.add_argument("-vp", "--video", required=False, help="Video path to make detection")
ap.add_argument("-ip", "--image", required=False, help="Image path to make detection")
ap.add_argument("-sp", "--save", required=True, help="Path to save detection")
ap.add_argument("-map", "--precision", required=False, help="Path to save detection")
args = ap.parse_args()
       
def calulate_mAP(model, dataloader, cfg, label_names, device, no_amp=True):
    metric = AveragePrecision(len(label_names), cfg.recall_steps, device)
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

def read_image(cfg, image_path=None, frame=None, input_size=320):
    size = (input_size, input_size)
    if image_path == None:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize(size)
    else:
        image = Image.open(image_path)
        image = image.resize(size)  
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(device)
    return image

def detect_from_video(cfg, video_path, save_path, model, logging, threshold, input_size, no_amp=True, fps = 300):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_file = save_path + 'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (input_size, input_size))  
    
    count_name = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, nb_found, _ = model.detect('Video frame', read_image(cfg, frame=frame, input_size=input_size), label_names, logging, threshold, no_amp)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        frame_count -= 1
        seconds = round(frame_count / fps)
        cv2.putText(frame, f'FPS: {int(fps)}   {datetime.timedelta(seconds=seconds)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2) 
        
        if nb_found > 0:
            count_name += 1
            frame = cv2.convertScaleAbs(frame, alpha=(255.0)) 
            video_writer.write(frame)
            cv2.imwrite(save_path + str(count_name) + '.jpg', frame)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(int(1000 / fps) ) & 0xFF == ord('q'):
            break
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

def get_split_test(dataset_path, testing_file):
    with open(dataset_path + testing_file) as f:
        ids = [line.strip() for line in f.readlines()]
    return [dataset_path + 'images/' + id + '.jpg' for id in ids]       

def convert_to_onnx(model, size, path):
    model.eval()
    dummy_input = torch.randn(1, 3, size, size) 
    onnx_path = path + "/model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path)
        
def process_image(cfg, image_path, save_path, label_names, logging, input_size, batch_size):
    image_name = image_path.split('/')[-1]
    image, nb_found, _ = model.detect(image_name, read_image(cfg, image_path=image_path, input_size=input_size), label_names, logging)        
    if nb_found > 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = save_path + f'result-{input_size}-{batch_size}-' + image_name.split('/')[-1]
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

    model_names = [x.split('.')[0] for x in os.listdir(config_path) if x.__contains__('yaml')]
    label_names = [
        '000', '001', '003', '004', '007', '008','009','023', 
        '025', '028', '035', '040', '042', '051', '052', '053'
    ]
    
    test_json = args.dataset + 'test.json'
    if args.precision:
        images_path_list = get_split_test(args.dataset, 'divide/test.txt')
    filename = log_path + 'testing_' + (str(datetime.datetime.now()).split('.')[0]).replace(':', '_')
    
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    FROM_DEFAULT_SIZE = True
    FROM_DEFAULT_BATCH = True
    input_sizes = [128, 256, 320, 512]
    batch_sizes = [8, 16, 32, 64]
              
    for input_size in input_sizes:
        if FROM_DEFAULT_SIZE:
            if input_size != args.img_size:
                continue
        print('IMAGE SIZE : ' + str(input_size) + 'x' + str(input_size))

        for batch_size in batch_sizes:
            if FROM_DEFAULT_BATCH:
                if batch_size != int(args.batch_size):
                    continue
            print('BATCH SIZE : ' + str(batch_size))
            #results_path = f'results{batch_size}/'
            results_path = f'results/'
            
            for model_name in model_names:
                if args.model:
                    if model_name != model_names[int(args.model)]:
                        continue

                best = ''   
                cfg = config_path + f'{model_name}.yaml'

                if os.path.exists(cfg):
                    cfg = load_config(cfg)
                    pth = results_path + f'{input_size}/{model_name}/best.pth'
                    
                    model = build_model(cfg, input_size, label_names, device)
                    model.to(device)
                    model.eval()

                    nb_layers = 0
                    for name, m in model.named_modules():
                        if len(list(m.named_children())) > 1: 
                            nb_layers += 1
                    nb_layers -= 2

                    print(f'The {model_name} model contains {nb_layers} layers.')
                    print('Number of parameters : ' + str(sum(p.numel() for p in model.parameters())))
                    print('Number of trainable parameters : ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

                    if os.path.exists(pth):
                        print('Loaded from pretrained model')
                        model.load_state_dict(torch.load(pth)['model_state_dict'])
                    else:
                        continue
                    
                    logging.info("=-------------------")
                    logging.info(model_name)
                    print(model_name)
                    logging.info("=-------------------\n")

                    save_path = args.save + '/' + model_name + '/'

                    if args.video:
                        if os.path.exists(args.video):
                            print('Detection from video.')
                            save_path = save_path.replace('/image/', '/video/')
                            detect_from_video(cfg, args.video, save_path, model, logging, threshold, input_size, no_amp)
                    elif args.image:
                        print('Detection from single image.')
                        process_image(cfg, args.image, save_path, label_names, logging, input_size, batch_size)
                    elif args.precision:
                        dataloader = create_dataloader(
                            test_json,
                            batch_size=batch_size,
                            image_size=input_size,
                            image_mean=cfg.image_mean,
                            image_stddev=cfg.image_stddev,
                            num_workers=workers
                        )
                        calulate_mAP(model, dataloader, cfg, label_names, device)
                    else:
                        print('Detection from test set images.')
                        dataloader = create_dataloader(
                            test_json,
                            batch_size=batch_size,
                            image_size=input_size,
                            image_mean=cfg.image_mean,
                            image_stddev=cfg.image_stddev,
                            num_workers=workers
                        )
                        for image_path in images_path_list:
                            process_image(cfg, image_path, save_path, label_names, logging, input_size, batch_size)
                else:
                    print(f"Please check if conf file and save path exist for {model_name}")
                
#python test.py -mi 0 -dataset "C:/datasets/"
#python test.py -mi 0 --video "D://DATAS//Videos//Sao Paulo 4K - Driving Downtown - Brazil.mp4" --save "C:/Users/monhel/Downloads"
#python test.py -mi 0 --image "C:/Users/monhel/OneDrive/Desktop/mobilenetV2_ssdlite/test6.jpeg" --save "C:/Users/monhel/OneDrive/Desktop"