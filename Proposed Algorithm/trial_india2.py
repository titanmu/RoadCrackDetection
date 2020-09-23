import argparse

from utils.datasets import *
import pandas as pd 
from utils.utils import *

import os

def vis_box(x, img, path, np_array, color=None, label=None, line_thickness=None):
    
    clss = ['D00','D10','D20','D40','D43','D44','D50']
    print (clss.index(label.split(' ')[0]))
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    x1,y1,x2,y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    # np_array.append([x1,y1,x2-x1,y2-y1,label.split(' ')[-1],clss.index(label.split(' ')[0]),1,1])
    np_array.append([x1,y1,x2,y2,label.split(' ')[-1],clss.index(label.split(' ')[0]),1,1])
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # cv2.imshow('',img)
    # cv2.waitKey(1000)
    return np_array,img

def detect(source, save_img=True):
    weights = '/home/ai/bigDataComp/yolov5/weights/pave/test_prev/last_india_new.pt'        #Pass the correct path of the model
    view_img = True
    save_txt = False 
    imgsz  = 640
    device = ''
    half = False
    augment=False
    iou_thres = 0.5
    # classes = None;conf_thres = 0.2;agnostic_nms = False
    classes = None;conf_thres = 0.15;agnostic_nms = False    #VM

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device)

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    # df=pd.DataFrame(columns=['x','y','w','h','score','cls','frame','time'])
    np_array = []
    for path, img, im0s,vc in dataset:        

        # print (ftime)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   fast=True, classes=classes, agnostic=agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        np_array,imgd = vis_box(xyxy, im0, path, np_array, label=label, color=colors[int(cls)], line_thickness=3)
            else:
                imgd = im0
    
    return np_array,imgd

 
# def filename1(file1):
#     with open ('submissions.txt', 'w') as file2:
#         file2.write()






with torch.no_grad():
    
    clss = ['D00','D10','D20','D40','D43','D44','D50']
    out_cls = ['D00','D10','D20','D40']
    # cnts = ['India','Japan','Czech']
    cnts = ['India']
    for cnt in cnts:
        img_source =os.path.join('../Trainer/test2',cnt,'images')           #Pass the Path of the test2 dataset
        # img_source = '../Trainer/test1/India/images'
        imgs = os.listdir(img_source)
        for imgcur in imgs:
            vid_source = os.path.join(img_source,imgcur)
            np_array, imgd = detect(vid_source)
            annts_list = []
            for annts in np_array:
                if clss[annts[5]] in out_cls:
                    annts_list.append([annts[5]+1,annts[0],annts[1],annts[2],annts[3]])
            
            if len(annts_list)>0:
                sub_list = [imgcur]+np.concatenate(annts_list).tolist()
                # print ("final output:",np.array(sub_list))
                # print("LENGTH", len(sub_list))
                with open('ttt3_test2.txt','a+') as writer:
                    txt=str(sub_list[0])+","
                    for i in range(1,len(sub_list)):
                        txt+=str(sub_list[i])+' '
                    txt+='\n'
                    writer.write(txt)
                # filename1=np.array(sub_list #vm
                # write to submission.txt

                # converter()  #vm

            else:
                print ("IMGCURL:",[imgcur])
                with open('ttt3_test2.txt','a+') as writer:
                    txt=str(imgcur)+","+"\n"
                    writer.write(txt)
                


