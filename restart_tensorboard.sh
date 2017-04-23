ps -ef|grep tensorboard|grep -v grep|cut -c 9-15|xargs kill -9

rm -rf summary/*

tensorboard --logdir ./summary --port 7777

