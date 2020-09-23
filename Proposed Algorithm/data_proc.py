import os, cv2
import pandas as pd 

dets_df = pd.read_csv('dets.csv')
vid_path = '1x.avi'
# print (dets_df.head())
cap = cv2.VideoCapture(vid_path)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # get all detections on current frame
        cur_dets = dets_df[dets_df['frame'] == cur_frame].values
        for det in cur_dets:
            [x1,y1,x2,y2,cls,frm] = det
            ## to crop image use this code
            # crop_img = frame[y1:y2, x1:x2]

            ## plot detections on image
            frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
        cv2.imshow('',frame)
        if cv2.waitKey(100) == ord('q'):
            break
            # print (det)
cap.release()
cv2.destroyAllWindows() 