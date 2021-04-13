from helper_function_new import *
from resunet import *
import os
import pathlib

data_num = ''
e_name = ''
crop_size = 2048

X_train = ''
Y_train = ''
X_val = ''
Y_val = ''

exp_name = e_name+'_'+data_num+'_'+crop_size
save_int_name = exp_name+'_int.h5'
save_name = exp_name + '.h5'
steps_per_epoch = 100
epochs_no = 50
val_steps = 50
img_size = (3511,5728)
crop_number = 5

train_generator,val_generator = generate_data(X_train,Y_train,X_val,Y_val,img_size)
train_crops = paired_crop_generator_resize_pad_sel(train_generator, crop_size, crop_number)
val_crops = paired_crop_generator_resize_pad_sel(val_generator, crop_size, crop_number)

model = resunet_without_pool(64,512,512,bt_state=True)

model.summary()

current_dir = pathlib.Path().absolute()
dest_path = os.path.join(current_dir, exp_name)
os.makedirs(dest_path, exist_ok=True)

make_path = os.path.join(current_dir, exp_name,'models')
os.makedirs(make_path, exist_ok=True)

fit_model1(model,train_crops,val_crops,save_int_name,exp_name,save_name,steps_per_epoch,val_steps,epochs_no)

number_of_images = 0

train_images_path = os.path.join(X_train, '1')
test_images_path = os.path.join(X_val, '1')
gt_train_images_path = os.path.join(Y_train, '1')
gt_test_images_path = os.path.join(Y_val, '1')

# Final model
model_weights = os.path.join(current_dir, exp_name, 'models',save_name)
test_dest_path = os.path.join(current_dir, exp_name, 'test_output_final_1')
os.makedirs(test_dest_path, exist_ok=True)
train_dest_path = os.path.join(current_dir, exp_name, 'train_output_final')
os.makedirs(train_dest_path, exist_ok=True)

stitch_save(model,model_weights,test_images_path,gt_test_images_path,test_dest_path,crop_size,number_of_images,save_images=True)

