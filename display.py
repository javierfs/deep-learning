from tf_unet import unet, util, image_util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#preparing data loading
data_provider = image_util.ImageDataProvider("../../../../Images/*.tif")

x_test, y_test = data_provider(1)

path = 'model.cpkt';

net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
prediction = net.predict(path, x_test)

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].imshow(x_test[0,...,0], aspect="auto", cmap='gray')
ax[1].imshow(prediction[0,...,1], aspect="auto", cmap='gray')
fig.savefig('results2.tif')   # save the figure to file