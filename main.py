from model import *
from data import *

# enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# extend dataset
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# remember to modify file path
myGene = trainGenerator(4, '/home/work/user-job-dir/unet/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('/home/work/user-job-dir/unet/unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=30,callbacks=[model_checkpoint])

testGene = testGenerator('/home/work/user-job-dir/unet/test')
results = model.predict_generator(testGene,5,verbose=1)
saveResult('/home/work/user-job-dir/unet/result', results)


# save to OBS when training on modelarts
import moxing as mox
mox.file.copy('/home/work/user-job-dir/unet/unet_membrane.hdf5',
              's3://mlproj/unet/unet_membrane.hdf5')
mox.file.copy_parallel('/home/work/user-job-dir/unet/result',
              's3://mlproj/unet/result')