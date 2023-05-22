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
from utils.constants import HEX_COLORS
import datetime
from utils.metrics import AveragePrecision
import pafy

def calulate_mAP(model, dataloader, cfg, class_names, device, no_amp=True):
    metric = AveragePrecision(len(class_names), cfg.recall_steps)
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
    for i, (ap, name) in enumerate(zip(APs, class_names)):
        print("%-5d%-20s%-13.3f%.3f" % (i, name, ap[0], ap.mean()))
    print("mAP@[0.5]: %.3f" % APs[:, 0].mean())
    print("mAP@[0.5:0.95]: %.3f" % APs.mean())
   
def get_color(cls=None):
    if cls:
        return tuple(int(HEX_COLORS[cls.cpu().numpy()][i:i+2], 16) for i in (1, 3, 5))
    else:
        return (0, 255, 0)
    
def get_label(cls, score):
    score_value = (str(floor(float(score.cpu().numpy()) * 100))) + '%'
    label =  class_names[cls.cpu().numpy()] + '-' + score_value
    return label, get_color()
 
def detection(image, model, model_name, threshold, show, no_amp):
    nb_found = 0
    with torch.no_grad():
        with autocast(enabled=(not no_amp)):
            preds = model(image)
    
    det_boxes, det_scores, det_classes = nms(*model.decode(preds))
    image = cv2.cvtColor(image[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
    
    found = False
    for box, score, cls in zip(det_boxes[0], det_scores[0], det_classes[0]):
        if score > threshold:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            label, color =  get_label(cls, score)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            found = True
            nb_found += 1
    
    if found:
        if show:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Detection with {model_name}")
            plt.show()
        print(label)
    else:
        if show:
            print(f'No object found in image with {model_name}.')
            
    return image, nb_found
    
def detect_from_single_image(image_path, model_name, model, threshold, show=True, no_amp=True):
    image = read_image(path=image_path)
    return detection(image, model, model_name, threshold, show, no_amp)

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

def detect_from_video(video_path, model, model_name, threshold, show=False, no_amp=True, desired_fps = 200):
    if not video_path.__contains__('mp4'):
        video = pafy.new(video_path)
        video_path = video.getbest(preftype="mp4")
    
    cap = cv2.VideoCapture(video_path)
    fps = desired_fps if desired_fps > 0 else cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = read_image(frame=frame)
        frame = detection(image, model, model_name, threshold, show, no_amp)[0]
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(), 2)
        cv2.imshow('Video', frame)
        
        # Exit if 'q' key is pressed
        if cv2.waitKey(int(1000 / fps) ) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

def make_images_list(testing_file):
    with open(testing_file) as f:
        ids = [line.strip() for line in f.readlines()]
    return [root + 'datasets/images/' + id + '.png' for id in ids]       

def convert_to_onnx(model, size, path):
    model.eval()
    dummy_input = torch.randn(1, 3, size, size) 
    onnx_path = path + "/model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path)
        
#=========================================================================================================

if __name__ == '__main__':
    device = 'cpu'
    root = os.getcwd() + '/'
    results_path = root + 'results/'
    config_path = root + 'configs/'
    dataset_path = root + 'datasets/'
    test_path = root + 'test/images/'
    model_names = [x.split('.')[0] for x in os.listdir(config_path) if x.__contains__('yaml')]

    class_names = [
        '000000', '000001', '000003', '000004', '000007', '000008','000009','000023', 
        '000025', '000028', '000035', '000040', '000042', '000051', '000052', '000053'
    ]
    
    threshold = 0.5
    test_json = dataset_path + 'test.json'
    images_path_list = os.listdir(test_path)
    image_path = dataset_path + 'images/000010.png' 
    video_path = root + 'test/videos/DRIVING IN BRAZIL_ Curitiba(PR) to São Paulo(SP).mp4'
    video_path = 'C:/Users/monhe/Videos/4K Video Downloader/DRIVING IN BRAZIL Paranagua-PR to Curitiba.mp4'
    video_path = 'C:/Users/monhe/Videos/4K Video Downloader/Brazilian Traffic Signs.mp4'
    
    images_path_list = make_images_list(dataset_path + 'divide/test.txt')

    tmp = []
    nb_taken = 0
    while nb_taken != 9:
        image = random.choice(images_path_list)
        if image not in tmp:
            tmp.append(image)
            nb_taken += 1
    images_path_list = tmp
                
    for model_name in model_names:

        cfg = config_path + f'{model_name}.yaml'
        pth = results_path + f'{model_name}/best.pth'

        if os.path.exists(cfg):
            workers = 4
            device = 'cpu'
            cfg = load_config(cfg)

            model = build_model(cfg, class_names)
            model.to(device)
            model.eval()

            #pth = pth.replace('pytorch-ssd-ufu', 'backup')
            if os.path.exists(pth):
                print('Loading pretrained model...')
                model.load_state_dict(torch.load(pth)['model_state_dict'])

            dataloader = create_dataloader(
                test_json,
                batch_size=cfg.batch_size,
                image_size=cfg.input_size,
                image_mean=cfg.image_mean,
                image_stddev=cfg.image_stddev,
                num_workers=workers
            )

            lenght = 70
            print("=" * lenght)
            print(f"Testing with {model_name} ({str(datetime.datetime.now()).split('.')[0]})")
            print("=" * lenght)
            
            #--------------------------------------------------------
            #convert_to_onnx(model, cfg.input_size, results_path + model_name)
            #calulate_mAP(model, dataloader, cfg, class_names, device)
            #detect_from_test_images(model, dataloader, cfg, class_names, device)
            video_path = "https://www.youtube.com/watch?v=QGtPj10K3aA"
            detect_from_video(video_path, model, model_name, threshold)
            #--------------------------------------------------------

            START = False

            if START:
                save = True
                show = False
                no_amp = True
                detectImages = []
                #images_path_list = os.listdir(test_path)
                    
                for image_name in images_path_list[0:30]:
                    if os.path.exists(test_path + image_name):
                        image_path = test_path + image_name
                    else:
                        image_path = image_name     
                    image, nb_found = detect_from_single_image(image_path, model_name, model, threshold, show=show, no_amp=no_amp)
                    detectImages.append(image)
                            
                cpt = 0
                NB_COLUMN = 3
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