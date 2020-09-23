import os, cv2, shutil, random, toml, json, math
import pandas as pd 
import xml.etree.ElementTree as ET
import numpy as np
from detect_di import detect
from scipy import ndimage
from skimage import transform


def testGauss(x_in, y_in):

    def gauss():
        n = round(len(x_in)/3)
        sigma = 10
        r = range(-int(n / 2), int(n / 2) + 1)
        return [1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]

    b = gauss()
    x_filt = filters.convolve1d(x_in, np.array(b)/sum(b))
    y_filt = filters.convolve1d(y_in, np.array(b) / sum(b))
    return x_filt, y_filt

def get_iou(a, b, epsilon=1e-5):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def get_gt(input_folder,cimg):
    cimg_path = os.path.join(input_folder,cimg)
    cannt_path = os.path.join(input_folder,cimg.replace('.jpg','.txt'))
    img_file__ = cv2.imread(cimg_path)
    annt_file = np.loadtxt(cannt_path)
    img_h, img_w, _ = img_file__.shape
    img_w, img_h
    dims = annt_file.ndim
    dets_info = []
    if dims > 1:
        for cbox in annt_file:
            cls=cbox[0]
            cbox = cbox[1:]
            x11, y11 = int((cbox[0] - cbox[2]/2)*img_w), int((cbox[1] - cbox[3]/2)*img_h)
            x22, y22 = int((cbox[0] + cbox[2]/2)*img_w), int((cbox[1] + cbox[3]/2)*img_h)
            dets_info.append([x11,y11,x22,y22,cls])
    else:
        cls=annt_file[0]
        cbox = annt_file[1:]
        x11, y11 = int((cbox[0] - cbox[2]/2)*img_w), int((cbox[1] - cbox[3]/2)*img_h)
        x22, y22 = int((cbox[0] + cbox[2]/2)*img_w), int((cbox[1] + cbox[3]/2)*img_h)
        img_file__ = cv2.rectangle(img_file__, (x11, y11), (x22, y22), (0,255,0), 3)
        dets_info.append([x11,y11,x22,y22,cls])
    return dets_info

def get_tfps(dets_vals, dets_gt):
    tps_gt = []; tps_dets = []; fps_dets = []
    for cur_det in dets_vals:
        box_info = []; iou_val = []; box_info_gt = []
        for cur_gt in dets_gt:
            cur_iou = get_iou(cur_det, cur_gt)
            if cur_iou>0.4 and (cur_det[-1]==cur_gt[-1]):
                # if len(iou_val)>0:
                if iou_val:
                    if cur_iou>iou_val:
                        iou_val = cur_iou
                        box_info = cur_det
                        box_info_gt = cur_gt
                else:
                    iou_val = cur_iou
                    box_info = cur_det
                    box_info_gt = cur_gt
        tps_dets.append(box_info)
        tps_gt.append(box_info_gt)
        if not(box_info):
            fps_dets.append(cur_det)
        # print (iou_val)
    return tps_dets, tps_gt, fps_dets

def get_fns(dets_vals, dets_gt):
    tps_gt = []; tps_dets = []; fns_dets = []
    for cur_det in dets_gt:
        box_info = []; iou_val = []; box_info_gt = []
        for cur_gt in dets_vals:
            cur_iou = get_iou(cur_det, cur_gt)
            if cur_iou>0.4 and (cur_det[-1]==cur_gt[-1]):
                iou_val = cur_iou
                box_info = cur_det
                box_info_gt = cur_gt
        tps_dets.append(box_info)
        tps_gt.append(box_info_gt)
        if not(box_info):
            # print ('false_negative')
            fns_dets.append(cur_det)
    return fns_dets

def vis_acc(df):
    df_imgs = df['filename'].unique()
    for df_img in df_imgs:
        df_filt = df[df['filename']==df_img]
        fn = df_filt.values[0][0]
        # print (fn)
        frame = cv2.imread(os.path.join(imgs,fn))
        for data in df_filt.values:
            img_file, x1,y1,x2,y2, clsv = data
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
        cv2.imshow('',frame)
        cv2.waitKey(1)

def plot_heatmap(image,coords_fns, title):

    frame = cv2.imread(image)
    x = np.zeros((frame.shape[0], frame.shape[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,title,(50,20), font, 1,(0,0,0), 1, 0)
    for cur_coord in coords_fns:
        # print (cur_coord)
        if not (math.isnan(cur_coord[1])):
            # x[ int(cur_coord[1]):int(cur_coord[3]),int(cur_coord[0]):int(cur_coord[2])] = 1
            x[ int(cur_coord[1]):int(cur_coord[3]),int(cur_coord[0]):int(cur_coord[2])] += 1
    heat_map = ndimage.filters.gaussian_filter(x, sigma=3)
    height = frame.shape[0]
    width = frame.shape[1]
    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))
    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    output = frame.copy()
    alpha = 0.7

    normalized_heat_map = 255*normalized_heat_map
    normalized_heat_map = cv2.applyColorMap(normalized_heat_map.astype(np.uint8),cv2.COLORMAP_JET)
    frame = cv2.addWeighted(frame, alpha,  normalized_heat_map, 1 - alpha, 0, normalized_heat_map)
    return frame


def display(preds, imgs,img):
    obj_list = ['car', 'motorcycle', 'truck','singl-unit']
    det_list = []
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            
            cur_row = [img, x1,y1,x2,y2,obj_list.index(obj)]
            det_list.append(cur_row)
            # df.loc[len(df)] = cur_row
  
    return det_list

def detec_cars(imgs,img_path):
    # df = pd.DataFrame(columns = ['x','y','w','h','score','cls'])
    df=pd.DataFrame(columns=['filename','x1','y1','x2','y2','cls'])
    # df.to_csv(os.path.join(det_folder, vidname+'.csv'),index=False,header=True)
    for img in imgs:
        frame = os.path.join(img_path,img)
        frame = cv2.imread(frame)
        ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        # result
        out = invert_affine(framed_metas, out)
        df_list = display(out, ori_imgs,img)
        for clist in df_list:
            if len(clist) == 6:
                df.loc[len(df)] = clist
    return df



input_folder = '../../paper/eval/test_data'
img_path = '../../paper/eval/imgs'
all_fldrs = os.listdir(img_path)
# print (all_fldrs)
for fldr in all_fldrs:
    test_imgs = os.listdir(os.path.join(img_path,fldr))
    # print (fldr)
    df=pd.DataFrame(columns=['filename','x1','y1','x2','y2','cls'])
    for cur_img in test_imgs:
        df_cur = detect(os.path.join(img_path,fldr, cur_img))
        # print (df)
        df = pd.concat([df,df_cur],axis=0)
    u_images = df['filename'].unique()
    all_fps = []; all_fns = []; all_tps = []

    for fname in u_images:
        # print (fname)
        df_filt = df[df['filename']==fname]
        dets_vals = df_filt[['x1','y1','x2','y2','cls']].values.tolist()
        dets_gt = get_gt(input_folder,fname)
        tps_dets, tps_gt,fps_dets = get_tfps(dets_vals, dets_gt)
        if len(fps_dets)>0:
            for fps_det in fps_dets:
                all_fps.append([fname] +fps_det)
        fns_dets = get_fns(dets_vals, dets_gt)
        if len(fns_dets)>0:
            for fns_det in fns_dets:
                all_fns.append([fname] +fns_det)
        if len(tps_dets)>0:
            for tps_det in tps_dets:
                all_tps.append([fname] +tps_det)
    # print (tps_det)
        ## compute iou --> get tps, fps, fns, etc
    df_fps = pd.DataFrame(all_fps,columns = ['filename','x1','y1','x2','y2','cls'])
    df_fns = pd.DataFrame(all_fns,columns = ['filename','x1','y1','x2','y2','cls'])
    df_tps = pd.DataFrame(all_tps,columns = ['filename','x1','y1','x2','y2','cls'])
    # vis_acc(df_fns)
    # df_fps.to_csv('test.csv',index=False)
    coords_fns = df_fns[['x1','y1','x2','y2']].values
    coords_fps = df_fps[['x1','y1','x2','y2']].values
    coords_tps = df_tps[['x1','y1','x2','y2']].values
    image = 'eval/imgs_/br-cam-008.streams_1cae233e-d5b9-4f1a-9407-d6b398604c30_02757.jpg'
    image = os.path.join(img_path,fldr,test_imgs[0])
    # print (coords)
    img_fns = plot_heatmap(image,coords_fns,'false negatives')
    img_fps = plot_heatmap(image,coords_fps, 'false positives')
    img_tps = plot_heatmap(image,coords_tps, 'true positives')
    final_img = cv2.hconcat((img_fns, img_fps, img_tps))
    cv2.imshow('Final', final_img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.imwrite(fldr + '_heatmap.jpg',final_img)


