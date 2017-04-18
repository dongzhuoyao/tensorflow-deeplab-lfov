python train.py --data_dir /cdata/hut/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012 \
--restore_from  ./model.ckpt-init \
--recurrent_times 5 \
--save_pred_every 50