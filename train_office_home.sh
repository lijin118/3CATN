#!/usr/bin/env bash

#a2c
python train_image.py --gpu_id 0  --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office-home/Art.txt  --t_dset_path data/office-home/Clipart.txt --dset office-home --source Art --target  Clipart

#a2p
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Art.txt  --t_dset_path data/office-home/Product.txt --dset office-home --source Art --target  Product

#a2r
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Art.txt  --t_dset_path data/office-home/Real_World.txt --dset office-home --source Art --target  Real_World

#c2a
python train_image.py --gpu_id 0  --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office-home/Clipart.txt  --t_dset_path data/office-home/Art.txt --dset office-home --source Clipart --target  Art

#c2p
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Clipart.txt  --t_dset_path data/office-home/Product.txt --dset office-home --source Clipart --target  Product

#c2r
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Clipart.txt  --t_dset_path data/office-home/Real_World.txt --dset office-home --source Clipart --target  Real_World

#p2a
python train_image.py --gpu_id 0  --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office-home/Product.txt  --t_dset_path data/office-home/Art.txt --dset office-home --source Product --target  Art

#p2c
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Product.txt  --t_dset_path data/office-home/Clipart.txt --dset office-home --source Product --target  Clipart

#p2r
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Product.txt  --t_dset_path data/office-home/Real_World.txt --dset office-home --source Product --target  Real_World

#r2a
python train_image.py --gpu_id 0  --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1 --s_dset_path data/office-home/Real_World.txt  --t_dset_path data/office-home/Art.txt --dset office-home --source Real_World --target  Art

#r2c
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Real_World.txt  --t_dset_path data/office-home/Clipart.txt --dset office-home --source Real_World --target  Clipart

#r2p
python train_image.py --gpu_id 0 --cyc_loss_weight 0.005 --cla_plus_weight 0.1  --weight_in_lossG 1,0.01,0.1,0.1  --s_dset_path data/office-home/Real_World.txt  --t_dset_path data/office-home/Product.txt --dset office-home --source Real_World --target  Product

