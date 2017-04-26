CUDA_VISIBLE_DEVICES=1  git pull && python train.py --data_dir /media/F/hutao/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012  \
--restore_from  ./model.ckpt-pretrained \
--save_pred_every 50