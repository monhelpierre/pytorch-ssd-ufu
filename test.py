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
ap.add_argument("-mi", "--model", required=False, help="Model index [0, 1, 2]")
args = vars(ap.parse_args())
       
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
            
    APs, recalls = metric.result
    print("ID   Class            AP@[0.5]     AP@[0.5:0.95]")
    for i, (ap, name) in enumerate(zip(APs, label_names)):
        print("%-5d%-20s%-13.3f%.3f" % (i, name, ap[0], ap.mean()))
    print("mAP@[0.5]: %.3f" % APs[:, 0].mean())
    print("mAP@[0.5:0.95]: %.3f" % APs.mean())

def read_image(path=None, frame=None):
    size = (cfg.input_size, cfg.input_size)
    if path == None:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize(size)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = Image.open(image_path)
        image = image.resize(size)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(device)
    return image

def detect_from_video(video_path, save_path, model, logging, threshold, no_amp=True, fps = 300):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cpt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = read_image(frame=frame)
        frame, nb_found, detection = model.detect('Video frame', image, label_names, logging, threshold, no_amp)
        
        if save_path and nb_found > 0:
            frame = cv2.convertScaleAbs(frame, alpha=(255.0))
            cv2.imwrite(save_path + str(cpt) + '.png', frame)
            cpt += 1
        
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
        cv2.imshow('Video', frame)
        
        # Exit if 'q' key is pressed
        if cv2.waitKey(int(1000 / fps) ) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
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
    
def show_selected_class(path='C:/Users/monhe/OneDrive/Desktop/signs/'):
    cpt = 0
    NB_COLUMN = 4
    images = [x for x in os.listdir(path) if x.__contains__('png')]
    NB_LINE = len(images)//NB_COLUMN
                
    plt.figure(figsize=(16, 8))
    for image_name in images:
        image_path = path + image_name
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(NB_LINE, NB_COLUMN, cpt+1)
        image = cv2.resize(image, (32, 32))
        plt.imshow(image)
        plt.title(image_name.split('.')[0])
        plt.axis("off")
        cpt += 1
    plt.show()
        
#=========================================================================================================

if __name__ == '__main__':
    workers = 4
    device = 'cpu'
    root = os.getcwd() + '/'
    results_path = root + 'results/'
    config_path = root + 'configs/'
    dataset_path = 'C:/datasets/'
    test_path = root + 'test/images/'
    log_path = root + 'logs/'
    model_names = [x.split('.')[0] for x in os.listdir(config_path) if x.__contains__('yaml')]

    label_names = [
        '000', '001', '003', '004', '007', '008','009','023', 
        '025', '028', '035', '040', '042', '051', '052', '053'
    ]
    
    #show_selected_class()
    save = True
    show = False
    no_amp = True
    threshold = 0.5
    test_json = dataset_path + 'test.json'
    image_path = dataset_path + 'images/000010.png' 
    video_path = root + 'test/videos/DRIVING IN BRAZIL_ Curitiba(PR) to São Paulo(SP).mp4'
    video_path = 'C:/Users/monhe/Videos/4K Video Downloader/DRIVING IN BRAZIL Campinas-SP to Tocos de Moji.mp4'
    images_path_list = get_split_test(dataset_path, 'divide/test.txt')
   
    if len(images_path_list) < 0:   
        tmp = []
        nb_taken = 0
        print('Loading 1 image for every class...')
        while nb_taken != 8:
            image = random.choice(images_path_list)
            if image not in tmp and not image.__contains__('@'):
                tmp.append(image)
                nb_taken += 1
        images_path_list = tmp
        print('Done Loading.')
        print(f'{len(images_path_list)} images for testing...')
    
    save_path = 'C:/Users/monhe/OneDrive/Desktop/signs/test/'
    video_save_path = 'C:/Users/monhe/OneDrive/Desktop/signs/video/'
    
    #extracted_images = [x for x in os.listdir('C:/datasets/images/') if not x.__contains__('@')]
    #print('Number of extracted images : ' + str(len(extracted_images)))
    
    filename = log_path + 'testing_' + (str(datetime.datetime.now()).split('.')[0]).replace(':', '_')
        
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
                 
    for model_name in model_names:
        if args['model']:
            if model_name != model_names[int(args['model'])]:
                continue
            
        cfg = config_path + f'{model_name}.yaml'

        if os.path.exists(cfg): 
              
            pth = results_path + f'{model_name}/best.pth'
            cfg = load_config(cfg)
            model = build_model(cfg, label_names)
            model.to(device)
            model.eval()
            
            logging.info("=-------------------")
            logging.info(model_name)
            logging.info("=-------------------\n")
            
            #print(model)
            #print('Number of parameters : ' + str(sum(p.numel() for p in model.parameters())))
            #print('Number of trainable parameters : ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            #pth = pth.replace('pytorch-ssd-ufu', 'backup')
                
            if os.path.exists(pth):
                print('Loading pretrained model...')
                model.load_state_dict(torch.load(pth)['model_state_dict'])
            
            FROM_IMAGES = False
            
            if not FROM_IMAGES:        
                #--------------------------------------------------------
                #convert_to_onnx(model, cfg.input_size, results_path + model_name)
                #calulate_mAP(model, dataloader, cfg, label_names, device)
                detect_from_video(video_path, video_save_path, model, logging, threshold, no_amp)
                #--------------------------------------------------------
            else:                
                dataloader = create_dataloader(
                    test_json,
                    batch_size=cfg.batch_size,
                    image_size=cfg.input_size,
                    image_mean=cfg.image_mean,
                    image_stddev=cfg.image_stddev,
                    num_workers=workers
                )

                detectImages = []
                
                save_path_file = save_path + model_name + '/'
                if not os.path.exists(save_path_file):
                    os.mkdir(save_path_file)
                
                cpt = 1
                for image_name in images_path_list:
                    if os.path.exists(test_path + image_name):
                        image_path = test_path + image_name
                    else:
                        image_path = image_name 
                        
                    image = read_image(image_path)
                    image, nb_found, detection = model.detect(image_name, image, label_names, logging)
                    #detectImages.append(image)
                    
                    filename = save_path_file + image_name.split('/')[-1]
                    if nb_found > 0 and save:
                        image = cv2.convertScaleAbs(image, alpha=(255.0))
                        cv2.imwrite(filename, image)
                               
                if len(detectImages) > 0:         
                    cpt = 0
                    NB_COLUMN = 4
                    NB_LINE = len(detectImages)//NB_COLUMN
                    plt.figure(figsize=(16, 8))
                    
                    for image in detectImages:
                        ax = plt.subplot(NB_LINE, NB_COLUMN, cpt+1)
                        image = cv2.resize(image, (800, 800))
                        plt.imshow(image)
                        plt.title('Image ' + str(cpt+1))
                        plt.axis("off")
                        cpt += 1
                    plt.show()
            
        else:
            print(f"Please check if conf file exist for {model_name}")