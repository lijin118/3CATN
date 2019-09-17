#!/usr/bin/env bash
python train_uspsmnist.py --epochs 30 --task USPS2MNIST --cla_plus_weight 0.3 --cyc_loss_weight 0.05 --weight_in_loss_g 1,0.01,1,1

python train_uspsmnist_pixel.py --epochs 30 --task MNIST2USPS --cla_plus_weight 0.3 --cyc_loss_weight 0.05 --weight_in_loss_g 1,0.01,1,1