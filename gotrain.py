from tf_unet import unet, util, image_util
import os
import random
import numpy as np

train_iters = 20
num_epochs = 150
feat_roots = 16
num_layers = 1
learning_rate = 1e-3
batch_size = 1
optimizer = "adam"

reg_lambda = 1e-6
n_classes = 2

#preparing data loading
data_provider = image_util.ImageDataProvider("/home/Kenneth/DL-Lung-Segmentation/img/train/*", data_suffix = "t1ce.jpg", mask_suffix = "seg.jpg")

output_path = "./output"
net = unet.Unet(layers=num_layers, features_root=feat_roots, channels=1, n_class=n_classes,  cost = "dice_coefficient", cost_kwargs=dict(regularizer=reg_lambda))
trainer = unet.Trainer(net, optimizer=optimizer, batch_size = batch_size)
trainer.train(data_provider, output_path, training_iters=train_iters, epochs=num_epochs)


