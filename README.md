# faster-rcnn

pytorch faster rcnn의 경우 800*800으로 resizing을 해준다. 
model 자체가 위에 맞게 training 되었으므로 2048을 그대로 쓸시 pretraining의 의미가 없어질 수 있고,
해상도를 낮추거나 pretraining을 다시 하는 식의 방법을 고민해야 한다.
해상도를 줄일 수록 batch수가 늘어날 수 있다.

anchor, feature pyramid 등 내부 동작을 정확히 이해한 후에
lbp 2048 image를 어떻게 처리할지 정확히 판단하자.

data parallel을 쓸시 batch norm이 불완전..
distributed data parallel을 써야 한다.


python -W ignore -m torch.distributed.launch --nproc_per_node=6 --master_addr=192.168.40.242 --master_port=50019 --use_env train.py --model fasterrcnn_resnet201_fpn --sync-bn

python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_addr=192.168.40.242 --master_port=50019 --use_env train_lbp.py --model fasterrcnn_resnet50_fpn --sync-bn --pretrained | tee outputs/lbp_train_resnet50.log
