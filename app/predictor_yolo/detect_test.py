# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import time
import shutil
import sys
from pathlib import Path
from numpy import random
from PIL import Image

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from app.predictor_yolo.models.common import DetectMultiBackend
from app.predictor_yolo.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from app.predictor_yolo.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,)
from app.predictor_yolo.utils.plots import Annotator, colors, save_one_box
from app.predictor_yolo.utils.torch_utils import select_device, time_sync
from app.com_ineuron_utils.utils import encodeImageIntoBase64


class Detection():
    def __init__(self, filename):
        self.weights = "./app/predictor_yolo/best.pt"
        self.conf = float(0.3)
        self.source = "./app/predictor_yolo/data/images"
        self.img_size = int(416)
        self.save_dir = "./app/predictor_yolo/data/output"
        self.view_img = False
        self.save_txt = False
        self.device = 'cpu'
        self.augment = True
        self.agnostic_nms = True
        self.conf_thres = float(0.3)
        self.iou_thres = float(0.45)
        self.classes = 0
        self.save_conf = True
        self.update = True
        self.filename = filename
        self.line_thickness = 3

    def detect(self):
        #source = str(source)
        out, source, weights, view_img, save_txt, imgsz, line_thickness = \
            self.save_dir, self.source, self.weights, self.view_img, self.save_txt, self.img_size, self.line_thickness
        webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

        #set_logging()
        device = select_device(self.device)
        if os.path.exists(out):  # output dir
            shutil.rmtree(out)  # delete dir
        os.makedirs(out)  # make new dir
        half = device.type != 'cpu'  # half precision only supported on CUDA

        #Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        model = DetectMultiBackend(weights, device=device, dnn=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        #stride, names, pt = model.stride, model.names, model.pt
        if half:
            model.half()  # to FP16

        save_img = True
        # dataset = LoadImages(source, img_size=imgsz)
        dataset = LoadImages(source + '/', img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

        # Get names and colors
        # names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        '''for path, img, im0, vid_cap, s in dataset:
            print(f'path:{path}, img:{img}, im0:{im0}, vid_cap:{vid_cap}, s:{s}')'''
        # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
        for path, img, im0s, vid_cap, s in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img[None]

            # Inference
            t1 = time_sync()
            # visualize = increment_path(out, mkdir=True)
            pred = model(img, augment=self.augment)

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       agnostic=self.agnostic_nms, multi_label=True)
            t2 = time_sync()

            # Apply Classifier
            '''if classify:
                pred = apply_classifier(pred, modelc, img, im0s)'''

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, conf, *xywh) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line) + '\n') % line)

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            print(label)
                            print(xyxy)
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))

                # Print time (inference + NMS)
                im0 = annotator.result()
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                if save_img:
                    if dataset.mode == 'image':

                        im = im0[:, :, ::-1]
                        # im = Image.fromarray(im0)

                        # im.save("/app/predictor_yolo/data/output/output.jpg")
                        cv2.imwrite(out + "/output.jpg", im0)
                    else:
                        print("Video Processing Needed")

        if save_txt or save_img:
            print('Results saved to %s' % Path(out))

            print('Done. (%.3fs)' % (time.time() - t0))

        return "Done"

    def detect_action(self):
        with torch.no_grad():
            self.detect()
        #bgr_image = cv2.imread("/app/predictor_yolo/data/output/output.jpg")
        # im_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('color_img.jpg', im_rgb)
        print('in detect action')
        opencodedbase64 = encodeImageIntoBase64("output.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        return result


'''def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='./input.jpg', help='input image to detect')
    opt = parser.parse_args()
    #opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(opt.images)
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))
    detector = Detection(filename=opt.images)
    detector.detect_action()
    #run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)'''
