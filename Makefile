download:
	wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip .
	unzip ./traffic-signs-data

train:
	nohup python train.py &