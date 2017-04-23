ps -ef|grep tensorboard|grep -v grep|cut -c 9-15|xargs kill -9 || true

rm -rf summary/* || true

nohup tensorboard --logdir ./summary --port 7777 > tensorboard.log & || true

