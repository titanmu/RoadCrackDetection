import os

# os.system('python detect.py --source d3.mp4 --weights weights/di/best.pt --view-img')
os.system('python detect.py --source d3.mp4 --weights weights/yolov5l.pt --view-img')
# os.system('python detect.py --source d3.mp4 --weights weights/di/last.pt --view-img')
# os.system('python detect.py --source d3.mp4 --weights weights/yolov5s.pt --view-img')
# os.system('python di_automl.py --source d8.mp4 --weights weights/yolov5s.pt --view-img')