import pygame
import time
import sys
import os
import RPi.GPIO as GPIO

#*****************************************************Detection imports****************************************************************************************************************
#mg967-ds983-sz392
#ECE5725: Design with Embedded Systems
#Team Members
#- Madhav Gupta (mg967)
#- Dhruv Sharma (ds983)
#- Shenhua Zhou (sz392)

#Wednesday Group
#Demo date: 05/13/2023



# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolov5 detect --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolov5 detect --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
#import os
import platform
import sys
import numpy as np
#import time
from pathlib import Path
from multiprocessing import Pool
from picamera2 import Picamera2, Preview

import torch

picam2 = Picamera2() #object for calling picamera2 methods
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

#from flask import Flask, send_from_directory
#from flask_cors import CORS
#*****************************************************Detection imports end here****************************************************************************************************************


#*****************************************************Detection function definitions****************************************************************************************************************

#Image capture function
def capture_image():
	camera_config = picam2.create_preview_configuration({"size" : (808, 606)})
	picam2.configure(camera_config)
	picam2.start()
	time.sleep(2)
	captured_images =  picam2.capture_file("/home/pi/Project/exp/data/images/take3.jpg")
	picam2.stop()
	captured_images =  picam2.start_and_capture_file("/home/pi/Project/exp/data/images/take1.jpg")    

def segment_into_four(img):
    #size
    (h,w) = img.shape[:2]

    #section1
    w1 = w//4
    sec1 = img[0:h, 0:w1]
    cv2.imwrite('/home/pi/Project/fragmentation/section1.jpg', sec1)
     #section2
    w2 = w//2
    sec2 = img[0:h, w1:w2]
    cv2.imwrite('/home/pi/Project/fragmentation/section2.jpg', sec2)
    #section3
    w3 = w1 + w2
    sec3 = img[0:h, w2:w3]
    cv2.imwrite('/home/pi/Project/fragmentation/section3.jpg', sec3)
    #section4
    w4 = w
    sec4 = img[0:h, w3:w4]
    cv2.imwrite('/home/pi/Project/fragmentation/section4.jpg', sec4)


#Detection run function
@smart_inference_mode()
def run(
        weights='/home/pi/Project/exp/best.pt',
        #weights='yolov5s.pt',  # model path or triton URL
        source=ROOT / '/home/pi/Project/fragmentation',  # file/dir/URL/glob/screen/0(webcam)
        #image_path = None,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=None,  # inference size (height, width)
        img=None,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='/home/pi/runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = "/home/pi/Project/fragmentation"
    # = {'chair': 0, 'occupied': 0}
    availability_data = []
    
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    if imgsz is None and img is None:
        imgsz = 640
    elif img is not None:
        imgsz = img

    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
        
   
        

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #Load Image
    #if image_path is not None:
    #    im0 = cv2.imread(image_path)
    #    im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
    #    im = im0
    #    im = np.ascontiguousarray(im)
    #    im = torch.from_numpy(im).to(device)
    #else:
    #    im = None
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        print('path:',path)
        each_availability_data = []
        filename = path
        filename = os.path.basename(filename)
        sectionTuple = os.path.splitext(filename)
        #print("section name is:", sectionTuple)
        which_area = sectionTuple[0]
        each_availability_data.append(which_area)
        
        print("area it parsed through was:" , which_area)
        
        #print('im:', im)
        #print('im0s', im0s)
        #print('vid_cap', vid_cap)
        #print('s', s)
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            print(det)
            det_class0 = det[det[:,-1]==0]
            det_class1 = det[det[:,-1]==1]            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        each_availability_data.append(len(det_class0)) #number of chairs
        each_availability_data.append(len(det_class1)) #number of occupied
        availability_data.append(each_availability_data)
        #print(res)
    #print(availability_data)
    return availability_data
        
        
        
        
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '/home/pi/Project/fragmentation', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
#*********************************************************************************************************************************************************************
#Main wrapper for run detection and feed the txt files with availability data
#*********************************************************************************************************************************************************************

def main():
    opt = parse_opt()
    capture_image() #saving full image and writing in path exp/data/images
    img = cv2.imread("/home/pi/Project/exp/data/images/take3.jpg") #reading captured image
    #fragmentation starts here
    segment_into_four(img)
    #run YOLOv5
    availability_data = run(**vars(opt))
    print(availability_data)
    #set availability
    section_1_availability = []
    section_2_availability = []
    section_3_availability = []
    section_4_availability = []
    section_1_availability = availability_data[0]
    section_2_availability = availability_data[1]
    section_3_availability = availability_data[2]
    section_4_availability = availability_data[3]
    #create text files appropriately
    if(section_1_availability[1] >= 1): 
        print("area available in section1")
        section1_file = open(r"/home/pi/Project/webpage/section1.txt","w+")
        section1_file.write("area available here")
        section1_file.close()
    else:
        section1_file = open(r"/home/pi/Project/webpage/section1.txt","w+")
        section1_file.write("area unavailable here")
        section1_file.close()
        
    if(section_2_availability[1] >= 1):
        print("area available in section2")
        section2_file = open(r"/home/pi/Project/webpage/section2.txt","w+")
        section2_file.write("area available here")
        section2_file.close()
    else:
        section2_file = open(r"/home/pi/Project/webpage/section2.txt","w+")
        section2_file.write("area unavailable here")
        section2_file.close()
    if(section_3_availability[1] >= 1):
        print("area available in section3")
        section3_file = open(r"/home/pi/Project/webpage/section3.txt","w+")
        section3_file.write("area available here")
        section3_file.close()
    else:
        section3_file = open(r"/home/pi/Project/webpage/section3.txt","w+")
        section3_file.write("area unavailable here")
        section3_file.close()
    if(section_4_availability[1] >= 1):
        print("area available in section4")
        section4_file = open(r"/home/pi/Project/webpage/section4.txt","w+")
        section4_file.write("area available in here")
        section4_file.close()
    else:
        section4_file = open(r"/home/pi/Project/webpage/section4.txt","w+")
        section4_file.write("area unavailable here")
        section4_file.close()
    if( (section_1_availability[1] == 0) and (section_2_availability[1] == 0) and (section_3_availability[1] == 0) and (section_4_availability[1] == 0)):
        print("sorry no space available, check back in sometime :(")
        #unavailability_file = open(r"/home/pi/Project/exp/unavailability_file.txt","w+")
        #unavailability_file.write("No area available :(, please check back in sometime")
        #unavailability_file.close()
    #else:
        #print("system might be broken, after all its a class project :(")
        
        
#*********************************************************************************************************************************************************************
#Main wrapper ENDS HERE
#*********************************************************************************************************************************************************************

           
           
           
           
#*********************************************************************************************************************************************************************
#GPIO setups and Pygame display
#*********************************************************************************************************************************************************************
GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.IN, pull_up_down = GPIO.PUD_UP)
GPIO.setup(17, GPIO.IN, pull_up_down = GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down = GPIO.PUD_UP)

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')

pygame.init()
size = width, height = 320, 240

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode(size)
start_detect = False

def GPIO17_callback(channel):
	global start_detect
	start_detect = True
	
	print("Start pressed")
	
	
def GPIO22_callback(channel):
	global start_detect
	start_detect = False
    #q=1
	print("Stop pressed")	
	
#def GPIO27_callback(channel):
	#print("27 pressed")	
	
#Detection functions and program

GPIO.add_event_detect(17, GPIO.FALLING, callback=GPIO17_callback, bouncetime=500)
GPIO.add_event_detect(22, GPIO.FALLING, callback=GPIO22_callback, bouncetime=500)

#GPIO.add_event_detect(27, GPIO.FALLING, callback=GPIO27_callback, bouncetime=500)
myfont = pygame.font.Font(None, 36)
colr = WHITE

#mybutton = {'Start': (190, 200), 'Quit': (50, 200)}
start_time = time.time()
q=0
while q == 0:
	time.sleep(0.1)
	for event in pygame.event.get():
		if event.type==pygame.QUIT: 
			GPIO.cleanup()
			sys.exit()
	if start_detect:
		print("shut the fuck up")
		colr = (0, 0, 255)
		main()
		#time.sleep(10)
	elif not start_detect:
		colr = WHITE
		print("I am out")
	screen.fill(BLACK)
	text_surface_quit = myfont.render('Quit', True, WHITE)
	quit_rect = text_surface_quit.get_rect(center = (290, 220))
	screen.blit(text_surface_quit, quit_rect)
	text_surface_stop = myfont.render('Stop', True, WHITE)
	stop_rect = text_surface_stop.get_rect(center = (290, 100))
	screen.blit(text_surface_stop, stop_rect)
	text_surface_start = myfont.render('Start', True, colr)
	start_rect = text_surface_start.get_rect(center = (290, 40))
	screen.blit(text_surface_start, start_rect)
	pygame.display.flip()
    #if (not GPIO.input(22)):
     #   start_detect = False
      #  print("Stop pressed")	
	if (not GPIO.input(27) or time.time()-start_time>300):
		q=1
		#GPIO.cleanup()
		pygame.quit()

GPIO.cleanup()
sys.exit()	


#if __name__ == "__main__":
	#main()
        #time.sleep(10)
        
        #if(not GPIO.input(27)):
           # sys.exit()
	
