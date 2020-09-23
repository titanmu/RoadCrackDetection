import argparse

from utils.datasets import *
import pandas as pd 
from utils.utils import *


def show_annotations(input_folder,cimg):
    cimg_path = os.path.join(input_folder,cimg)
    cannt_path = os.path.join(input_folder,cimg.replace('.jpg','.txt'))
    if  os.path.isfile(cannt_path):
        img_file__ = cv2.imread(cimg_path);
        annt_file = np.loadtxt(cannt_path)

        img_h, img_w, _ = img_file__.shape
        img_w, img_h
        dims = annt_file.ndim
        annt_list = []
        if dims > 1:
            for cbox in annt_file:
                # print (cbox)
                clsv = cbox[0]
                cbox = cbox[1:]
                
                x11, y11 = int((cbox[0] - cbox[2]/2)*img_w), int((cbox[1] - cbox[3]/2)*img_h)
                x22, y22 = int((cbox[0] + cbox[2]/2)*img_w), int((cbox[1] + cbox[3]/2)*img_h)
                img_file__ = cv2.rectangle(img_file__, (x11, y11), (x22, y22), (0,255,0), 3)
                annt_list.append([clsv,x11,y11,x22,y22])
        else:
            clsv=annt_file[0]
            cbox = annt_file[1:]
            x11, y11 = int((cbox[0] - cbox[2]/2)*img_w), int((cbox[1] - cbox[3]/2)*img_h)
            x22, y22 = int((cbox[0] + cbox[2]/2)*img_w), int((cbox[1] + cbox[3]/2)*img_h)
            annt_list.append([clsv,x11,y11,x22,y22])
            # img_file__ = cv2.rectangle(img_file__, (x11, y11), (x22, y22), (0,255,0), 3)
        # cv2.imshow('',img_file__)
        # cv2.waitKey(0)
    else:
        annt_list = []
    return annt_list

def vis_box(x, img, path, np_array, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    # img_name = os.path.basename(path)
    # ann_list = show_annotations('training_img_annt',img_name)
    # for cann in ann_list:
    #     cv2.rectangle(img, (cann[1],cann[2]), (cann[3],cann[4]), (255,255,255), thickness=5, lineType=cv2.LINE_AA)
    
    clss = ['D00','D10','D20','D40','D44']
    print (clss.index(label.split(' ')[0]))
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    x1,y1,x2,y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    np_array.append([x1,y1,x2-x1,y2-y1,label.split(' ')[-1],clss.index(label.split(' ')[0]),1,1])
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imshow('',img)
    cv2.waitKey(500)
    return np_array 

def detect(source, save_img=True):

    weights = 'weights/pave/best.pt'
    view_img = True
    save_txt = False 
    imgsz  = 640
    device = ''
    half = False
    augment=False
    iou_thres = 0.1
    classes = None;conf_thres = 0.2;agnostic_nms = False

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
                        # print ('saving images')
                        np_array = vis_box(xyxy, im0, path, np_array, label=label, color=colors[int(cls)], line_thickness=3)
    
        # print (df.tail())
        # print (len(np_array))
    return np_array

with torch.no_grad():
    # cur_det_folder = os.path.join('../../paper/output', 'detections','yolov5')
    # if not (os.path.isdir(cur_det_folder)):
        # os.makedirs(cur_det_folder)
    
    img_source = '/media/ai/New Volume1/dev/aicity/pavement/test1/Japan/images'
    # img_source = 'vids'
    imgs = os.listdir(img_source)
    for imgcur in imgs:
        vid_source = os.path.join(img_source,imgcur)
        np_array = detect(vid_source)
    # df=pd.DataFrame(np_array, columns=['x','y','w','h','score','cls','frame','time'])

    # df.to_csv(os.path.join(det_folder, vidname+'.csv'),index=False,header=True)

    
    