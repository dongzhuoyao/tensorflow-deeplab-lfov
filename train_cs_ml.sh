CUDA_VISIBLE_DEVICES=0  python train.py --data_dir /home/guest1/Documents/cityscapes  \
--data_list ./cityscapes_dataset/train.txt \
--input_size 641,641 \
--restore_from  ./model.ckpt-pretrained \
--save_pred_every 500