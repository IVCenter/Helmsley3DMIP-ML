from model import *
from data import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
data_gen_args = dict(rotation_range=0.2,
					width_shift_range=0.05,
					height_shift_range=0.05,
					shear_range=0.05,
					zoom_range=0.05,
					horizontal_flip=True,
					fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','mri_image_2016','mri_label_2016',data_gen_args,save_to_dir = None)
model = unet_lrelu()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=20000,epochs=5,callbacks=[model_checkpoint])
