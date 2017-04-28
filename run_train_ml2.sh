CUDA_VISIBLE_DEVICES=1,2  python train.py --data_dir /media/F/hutao/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012  \
--restore_from  ./model.ckpt-pretrained \
--attentionbranch_restore_from ./snapshots/model.ckpt-17400
--save_pred_every 50