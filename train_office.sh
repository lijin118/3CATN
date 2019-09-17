#!/usr/bin/env bash
#w2a
python train_image.py --gpu_id 0  --cyc_loss_weight 0.05 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office/webcam_list.txt  --t_dset_path data/office/amazon_list.txt --source webcam --target  amazon

#w2d
python train_image.py --gpu_id 0  --cyc_loss_weight 0.05 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office/webcam_list.txt  --t_dset_path data/office/dslr_list.txt --source webcam --target  dslr

#a2w
python train_image.py --gpu_id 0  --cyc_loss_weight 0.05 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office/amazon_list.txt  --t_dset_path data/office/webcam_list.txt --source amazon --target  webcam

#a2d
python train_image.py --gpu_id 0  --cyc_loss_weight 0.05 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office/amazon_list.txt  --t_dset_path data/office/dslr_list.txt --source amazon --target  dslr

#d2w
python train_image.py --gpu_id 0  --cyc_loss_weight 0.05 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office/dslr_list.txt  --t_dset_path data/office/webcam_list.txt --source dslr --target  webcam

#d2a
python train_image.py --gpu_id 0  --cyc_loss_weight 0.05 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office/dslr_list.txt  --t_dset_path data/office/amazon_list.txt --source dslr --target  amazon
