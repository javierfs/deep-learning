#! /usr/bin/env python

import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
b_size=int(sys.argv[1])
iters=int(sys.argv[2])
learning_rate=int(sys.argv[3])
epoch_var=int(sys.argv[4])
reg_var=int(sys.argv[5])
reg_lambda = 10**-reg_var

from tf_unet import unet, util, image_util

data_provider = image_util.ImageDataProvider("../Images/*.tif")
result_path = "itr"+str(iters)+"_epo"+str(epoch_var)+"_bat"+str(b_size)+"_lrn"+str(learning_rate)+"_reg10e-"+str(reg_var)
output_path = "results_last/" + result_path

net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2,cost_kwargs=dict(regularizer=reg_lambda))
trainer = unet.Trainer(net,batch_size=b_size, optimizer="adam")
path = trainer.train(data_provider,output_path, training_iters=iters, epochs=epoch_var, write_graph=True)
logging.info("worked")
