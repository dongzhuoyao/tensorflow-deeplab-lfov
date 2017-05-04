CUDA_VISIBLE_DEVICES=1  python train.py --data_dir /media/F/hutao/dataset/ADEChallengeData2016  \
--data_list ./ade20k_dataset/train.txt \
--input_size 641,641 \
--restore_from  ./model.ckpt-pretrained \
--save_pred_every 500