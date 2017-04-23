git pull && python train.py --data_dir /cdata/hut/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012 \
--restore_from  ./model.ckpt-pretrained \
--save_pred_every 50