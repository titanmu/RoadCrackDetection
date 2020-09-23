import os 
# os.system('python train_main.py --data data/pave.yaml --cfg models/yolov5x_pave.yaml --batch-size 8 --resume')
# os.system('python train_main.py --data data/pave.yaml --cfg models/yolov5x_pave.yaml --batch-size 6 --device 0 --resume')  #Japan_Czech

os.system('python train_main.py --data data/pave.yaml --cfg models/yolov5m_pave.yaml --batch-size 8 --device 0 --resume' )  #Japan_Czech

# os.system('python train_main.py --data data/pave_india.yaml --cfg models/yolov5x_india.yaml --batch-size 6 --device 0 --resume')  #India

# os.system('python train_main.py -pyth-data data/pave.yaml --cfg models/yolov5x_pave.yaml --weights '' --batch-size 8')

## test model
# run submissions.py
# os.system('python detect.py --weights weights/pave/last.pt --source ../Trainer/test1/Japan/images --conf-thres 0.2')