import os
import cv2
import shutil
import datetime
import imutils
from PIL import Image

path = [
        'C:/Users/monhe/OneDrive/Documents/pytorch-ssd-ufu/',
        'C:/Users/monhe/OneDrive - Universidade Federal de Uberlândia/ufu/pytorch-ssd-ufu/'
]

root = path[1] + 'datasets/'
root2 = 'C:/Users/monhe/OneDrive/Documents/datasets/'

def extract_images_from_videos(base_path):
    images_path_ = base_path + 'new-images/'
    videos_path_ = 'C:/Users/monhe/Videos/4K Video Downloader/'
    cpt = 0

    for vname in os.listdir(videos_path_):
        startTime = datetime.datetime.now()
        cap = cv2.VideoCapture(videos_path_ + vname)
        saveImg = 10
        imgCnt = 0
        frameCnt = 0
        fCnt = 0
        startTime = datetime.datetime.now()
        print('Extracting images from {} video file...'.format(vname))
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            frame = imutils.resize(frame, width=800)
            frameCnt = frameCnt + 1
            fCnt = fCnt + 1

            if frameCnt == saveImg:
                cpt += 1
                frameCnt = 0
                imgCnt = imgCnt + 1
                cv2.imwrite(f"{images_path_ + str(cpt)}.png", frame)
            
        if frameCnt == 0:
            print("Unable to process.\r\n")
            continue
        endTime = datetime.datetime.now()
        diff = endTime - startTime
        sec = diff.seconds
        fps = round(fCnt/sec, 3)
        print("Total FPS {}".format(fps))
        print("Total Img saved {}".format(imgCnt))
    print("Total of all Img saved {}".format(len(os.listdir(images_path_))))
    
if __name__ == '__main__':
    #extract_images_from_videos(root2)
    #savepath = 'C:/Users/monhe/Downloads/draft/'
    
    #for image_name in os.listdir(root + 'images'):
        #im1 = Image.open(f'{root}images/{image_name}')
        #im1.save(f"{savepath}{image_name.replace('png', 'jpg')}")
        
        """
        if not image_name.__contains__('@'):
            original = root + 'images/' + image_name
            target =  root2 + 'images/' + image_name
            
            original_ann = root + 'annotations/' + image_name.replace('jpg', 'xml')
            target_ann = root2 + 'annotations/' + image_name.replace('jpg', 'xml')
            
            if not os.path.exists(target):
                shutil.copyfile(original, target)
            
            if not os.path.exists(target_ann):
                shutil.copyfile(original_ann, target_ann)
        """