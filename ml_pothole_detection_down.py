# Modified version to save label location in pothole.csv
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
"""
import os
import platform
import sys
from pathlib import Path

import torch
import pandas as pd

# Initializa parameters and flags
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def detect_pothole(
        weights=ROOT / 'vTwo.pt',  # model path or triton URL
        source= 3,  # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=True,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / '',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    txt_path = str(ROOT / Path('pothole_detected%s.csv'% source))
    print("text path at:", txt_path)
    # f = open(txt_path, "w")
    # f.write("0,0,0,0")
    # f.close()

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)
   

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
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
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # # Write results
                # f = open(txt_path, 'w')
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # Write to file
                    # width = abs(xyxy[2] - xyxy[0])
                    # height = abs(xyxy[3] - xyxy[1])
                    # xCenter = (xyxy[2] - xyxy[0])//2
                    # yCenter = (xyxy[3] - xyxy[1])//2
                    xCenter = xywh[0]
                    yCenter = xywh[1]
                    xywh_pd = pd.DataFrame(xywh)
                    xywh_pd.to_csv(txt_path, float_format = '%g', header = False, index = False)
                    # f.write("%i, %i, %i, %i"%(xCenter, yCenter,  xywh[2],  xywh[3]))
                    # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                # f.close()
            # if not detected, clear file
            # else:
            #     f = open(txt_path, "w")
            #     f.write("0, 0, 0, 0")
            #     f.close()
                  
            # Stream result
            im0 = annotator.result()

            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        if len(det):
            LOGGER.info(f"{s}{len(det)} at x={xCenter} y={yCenter} Time{dt[1].dt * 1E3:.1f}ms")

    return xywh

model = ROOT / 'vTwo.pt'
webcam = 2
detect_pothole(weights = model, source = webcam)


