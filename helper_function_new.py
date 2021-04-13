from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from os import listdir
from os.path import join
from matplotlib import pyplot
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from scipy import signal
import csv
from tensorflow.python.keras import backend as K
import copy
from resunet import *

def combine_generator(gen1, gen2):
    while True:
        yield (gen1.next(), gen2.next())


def group_generators(gen1, data1, gen2, data2, img_size):
    seed = 1
    image_generator = gen1.flow_from_directory(
        data1, class_mode=None, color_mode='grayscale', target_size=img_size, seed=seed, batch_size=1)
    mask_generator = gen2.flow_from_directory(data2, class_mode=None, color_mode='grayscale', target_size=img_size, seed=seed, batch_size=1)
    grouped_generator = combine_generator(image_generator, mask_generator)

    return grouped_generator


def generate_data(X_train, Y_train, X_val, Y_val, img_size):
    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True, rescale=1.0 / 255.0)

    data_gen_args1 = dict(
        horizontal_flip=True,
        vertical_flip=True, rescale=1.0 / 255.0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args1)

    # image_datagen.fit(fit_data)

    train_generator = group_generators(image_datagen, X_train, mask_datagen, Y_train, img_size)
    val_generator = group_generators(image_datagen, X_val, mask_datagen, Y_val, img_size)

    return train_generator, val_generator


def downsize(array, downsize_ratio, y_val=0):
    a = ndimage.interpolation.zoom(array, downsize_ratio)
    if y_val == 1:
        a = np.where(a > 0.5, 1, 0)
    return a


def paired_crop_generator_resize(batches, crop_length, number_crops):
    while True:
        (batch_X, batch_Y) = next(batches)
        crop_length1 = 512
        batch_Xcrops = np.zeros((batch_X.shape[0] * number_crops, crop_length1, crop_length1, 1))
        batch_Ycrops = np.zeros((batch_X.shape[0] * number_crops, crop_length1, crop_length1, 1))
        height, width = batch_X.shape[1], batch_X.shape[2]
        dy, dx = crop_length, crop_length

        k = 0
        for i in range(batch_X.shape[0]):
            for j in range(number_crops):
                y = np.random.randint(0, height - dy + 1)
                x = np.random.randint(0, width - dx + 1)
                batch_Xcrops[k, :, :, 0] = downsize(batch_X[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=0)
                d2 = downsize(batch_Y[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=1)
                batch_Ycrops[k, :, :, 0] = d2
                k = k + 1
        yield (batch_Xcrops, batch_Ycrops)

def paired_crop_generator_resize_pad(batches, crop_length, number_crops):
    while True:
        (batch_X, batch_Y) = next(batches)
        crop_length1 = 512
        pad_x = crop_length - (batch_X.shape[1] % crop_length)
        pad_y = crop_length - (batch_X.shape[2] % crop_length)

        batch_X1 = np.zeros((batch_X.shape[0], batch_X.shape[1]+pad_x,batch_X.shape[2]+pad_y,batch_X.shape[3]))
        batch_Y1 = np.zeros((batch_X.shape[0], batch_X.shape[1]+pad_x,batch_X.shape[2]+pad_y,batch_X.shape[3]))
        for i in range(batch_X.shape[0]):
            batch_X1[i, :, :, 0] = np.pad(batch_X[i, :, :, 0], ((0, pad_x), (0, pad_y)), 'reflect')
            batch_Y1[i, :, :, 0] = np.pad(batch_Y[i, :, :, 0], ((0, pad_x), (0, pad_y)), 'reflect')
        batch_Xcrops = np.zeros((batch_X.shape[0] * number_crops, crop_length1, crop_length1, 1))
        batch_Ycrops = np.zeros((batch_X.shape[0] * number_crops, crop_length1, crop_length1, 1))
        height, width = batch_X1.shape[1], batch_X1.shape[2]
        dy, dx = crop_length, crop_length
        total_crops = number_crops*batch_X.shape[0]
        k = 0
        for i in range(batch_X.shape[0]):
            while k < total_crops:
                y = np.random.randint(0, height - dy + 1)
                x = np.random.randint(0, width - dx + 1)

                batch_Xcrops[k, :, :, 0] = downsize(batch_X1[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=0)
                d2 = downsize(batch_Y1[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=1)
                batch_Ycrops[k, :, :, 0] = d2
                k = k + 1
        yield (batch_Xcrops, batch_Ycrops)


def paired_crop_generator_resize_pad_sel(batches, crop_length, number_crops):
    while True:
        (batch_X, batch_Y) = next(batches)
        crop_length1 = 512
        pad_x = crop_length - (batch_X.shape[1] % crop_length)
        pad_y = crop_length - (batch_X.shape[2] % crop_length)

        batch_X1 = np.zeros((batch_X.shape[0], batch_X.shape[1]+pad_x,batch_X.shape[2]+pad_y,batch_X.shape[3]))
        batch_Y1 = np.zeros((batch_X.shape[0], batch_X.shape[1]+pad_x,batch_X.shape[2]+pad_y,batch_X.shape[3]))
        for i in range(batch_X.shape[0]):
            batch_X1[i, :, :, 0] = np.pad(batch_X[i, :, :, 0], ((0, pad_x), (0, pad_y)), 'reflect')
            batch_Y1[i, :, :, 0] = np.pad(batch_Y[i, :, :, 0], ((0, pad_x), (0, pad_y)), 'reflect')
        batch_Xcrops = np.zeros((batch_X.shape[0] * number_crops, crop_length1, crop_length1, 1))
        batch_Ycrops = np.zeros((batch_X.shape[0] * number_crops, crop_length1, crop_length1, 1))
        height, width = batch_X1.shape[1], batch_X1.shape[2]
        dy, dx = crop_length, crop_length
        total_crops = batch_X.shape[0] * number_crops
        k = 0
        for i in range(batch_X.shape[0]):
            while k < total_crops:
                y = np.random.randint(0, height - dy + 1)
                x = np.random.randint(0, width - dx + 1)

                if k < np.ceil(total_crops * 0.8):
                    c = batch_Y1[i, y:(y + dy), x:(x + dx), 0]
                    if np.count_nonzero(c)>100:
                        batch_Xcrops[k, :, :, 0] = downsize(batch_X1[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=0)
                        batch_Ycrops[k, :, :, 0] = downsize(batch_Y1[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=1)
                        k = k + 1
                else:
                    batch_Xcrops[k, :, :, 0] = downsize(batch_X1[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=0)
                    batch_Ycrops[k, :, :, 0] = downsize(batch_Y1[i, y:(y + dy), x:(x + dx), 0], 512 / crop_length, y_val=1)
                    k = k + 1
        yield (batch_Xcrops, batch_Ycrops)



def fit_model1(model,train_generator,val_generator,save_int_name,exp_name,save_final_name,spe,val_steps,epochs_no):

    model_checkpoint = ModelCheckpoint(join(exp_name,"models", save_int_name), monitor='loss',
                                       save_best_only=True, verbose=True)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=spe, validation_data=val_generator, validation_steps=val_steps,
        epochs=epochs_no, verbose=2,callbacks=[model_checkpoint])

    model.save(join(exp_name,"models", save_final_name))
    csvfile = open(os.path.join(exp_name, 'train_loss.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(history.history['loss'])
    csvfile.flush()
    csvfile = open(os.path.join(exp_name, 'val_loss.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(history.history['val_loss'])
    csvfile.flush()


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_thres(y_true, y_pred):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def get_dice_thres(predicted_op, gt_file_path):
    gt_img1 = gt_file_path
    p_copy = copy.deepcopy(predicted_op)
    dice = dice_coef_thres(gt_img1, p_copy)
    return dice


def get_dice(predicted_op, gt_file_path):
    gt_img1 = gt_file_path
    dice = dice_coef(predicted_op, gt_img1)
    return dice

def stitch_save(model, model_weights, input_images_path, gt_images_path, dest_path, sub_image_size, num_images,save_images = False):
    model.load_weights(model_weights)
    onlyfiles = [f for f in listdir(input_images_path)]
    onlyfiles_gt = [f for f in listdir(gt_images_path)]
    print('Number of test files:', len(onlyfiles))
    csvfile = open(os.path.join(dest_path, 'dice_metrics.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(['image', 'dice', 'dice_threshold', 'fiou', 'dice_loftis'])

    csvfile1 = open(os.path.join(dest_path, 'dice_metrics_mean.csv'), 'w', newline='')
    iteration_metrics_writer1 = csv.writer(csvfile1)
    iteration_metrics_writer1.writerow(['image', 'mean_dice', 'mean_dice_threshold', 'mean_fiou', 'mean_dice'])

    if num_images == 0:
        final = len(onlyfiles)
    else:
        final = num_images
    k = 0
    d_sim = []
    d_sim_t = []
    foreground_iou_list = []
    foreground_f_score_list = []
    for i in range(0, final):
        if onlyfiles[i][0] is not '.':
            file = join(input_images_path, onlyfiles[i])
            ground_truth = (join(gt_images_path, onlyfiles_gt[i]))

            stitched_result = corrected_predicted_image(model, model_weights,file, sub_image_size, 1)
            dice_score = get_dice(stitched_result, ground_truth)
            #dice_numpy = dice_score.eval(session=tf.compat.v1.Session())
            dice_numpy = dice_score.numpy()
            # print(dice_numpy)

            dice_score_t = get_dice_thres(stitched_result, ground_truth)
            #dice_numpy_t = dice_score_t.eval(session=tf.compat.v1.Session())
            dice_numpy_t = dice_score_t.numpy()
            # print(dice_numpy_t)

            foreground_iou = compute_iou_for_class(stitched_result, ground_truth, cls=1)
            if foreground_iou is not None:
                foreground_iou_list.append(foreground_iou)
            foreground_f_score = compute_F_score_for_class(stitched_result, ground_truth, cls=1)
            if foreground_f_score is not None:
                foreground_f_score_list.append(foreground_f_score)

            d_sim.append(dice_numpy)
            d_sim_t.append(dice_numpy_t)
            iteration_metrics_writer.writerow(
                [onlyfiles[i], d_sim[k], d_sim_t[k], foreground_iou_list[k], foreground_f_score_list[k]])
            if save_images == True:
                raw_name = join(dest_path, onlyfiles[i])
                raw_name = raw_name + '.png'
                stitched_result = stitched_result * 255
                cv2.imwrite(raw_name, stitched_result)
            k = k + 1
    iteration_metrics_writer.writerow(['mean', sum(d_sim) / len(d_sim), sum(d_sim_t) / len(d_sim_t),
                                       sum(foreground_iou_list) / len(foreground_iou_list),
                                       sum(foreground_f_score_list) / len(foreground_f_score_list)])
    iteration_metrics_writer1.writerow(
        ['final_mean', sum(d_sim) / len(d_sim), sum(d_sim_t) / len(d_sim_t),
         sum(foreground_iou_list) / len(foreground_iou_list),
         sum(foreground_f_score_list) / len(foreground_f_score_list)])
    csvfile.flush()
    csvfile1.flush()

def corrected_predicted_image(model,model_weights, name, image_size, mean):
    if image_size == 512:
        sub = 256
    elif image_size == 2048:
        sub = 1023

    stitched_result_1 = predict_and_stitch4(model,model_weights, name, image_size, mean, 0, 0)
    stitched_result_2 = predict_and_stitch4(model,model_weights, name, image_size, mean, 0, sub)
    stitched_result_3 = predict_and_stitch4(model,model_weights, name, image_size, mean, sub, 0)
    stitched_result_4 = predict_and_stitch4(model,model_weights, name, image_size, mean, sub, sub)
    r=image_size;c=image_size;p = 0.5
    wc = signal.tukey(r,p)
    wr = signal.tukey(c,p)
    [maskr,maskc] = np.meshgrid(wr,wc)
    w = np.multiply(maskr,maskc)
  
    [s1,s2]=stitched_result_1.shape
    new = np.tile(w,(np.ceil(s1/image_size). astype(int),np.ceil(s2/image_size). astype(int)))
    new1 = new[0:s1,0:s2]
     
    m1 = np.multiply(stitched_result_1,new1)
    l2 = 1-new1
    g2 = m1 + np.multiply(stitched_result_2,l2)
    g3 = m1 + np.multiply(stitched_result_3,l2)
    g_all = m1 + np.multiply(stitched_result_4,l2)
    
    mask1 = copy.deepcopy(l2)
    mask2 = copy.deepcopy(l2)
    
    ind = np.where(l2[:,-1000] == 0)[0]
    mask1[ind,:]=0
    mask1[mask1>0]=1
    ind2 = np.where(l2[1000,:] == 0)[0]
    mask2[:,ind2]=0
    mask2[ind,:]=0
    mask2[mask2>0]=1
    
    int1 = np.multiply(g3,mask1) + np.multiply(g2,1-mask1)
    int2 = np.multiply(int1,1-mask2) + np.multiply(g_all,mask2)
    
    int2[0:(image_size//2),:]=g2[0:(image_size//2),:]
    int2[:,0:(image_size//2)]=g3[:,0:(image_size//2)]
    
    mask3 = 1-mask2
    mask3[image_size//2:,image_size//2:]=0
    mask3[0:(image_size//2),0:(image_size//2)]=1
    
    final = np.multiply(int2,1-mask3) + np.multiply(stitched_result_1,mask3)
    return final 



def predict_and_stitch4(model, model_weights, name, image_size, mean, start1, start2):
    model = resunet_without_pool(64,512,512,bt_state=True)
    model.load_weights(model_weights)
    img = cv2.imread(name, cv2.COLOR_BGR2GRAY)
    pad_x = image_size - (img.shape[0] % image_size) + start1
    pad_y = image_size - (img.shape[1] % image_size) + start2
    new_img = np.pad(img, ((0, pad_x), (0, pad_y)), 'reflect')
    stitched_output = np.zeros([img.shape[0], img.shape[1]])
    for row in range(start1, new_img.shape[0], image_size):
        for col in range(start2, new_img.shape[1], image_size):
            if col < img.shape[1]:
                sub_image = new_img[row:row + image_size, col: col + image_size]
                sub_image1 = downsize(sub_image, 512 / image_size, y_val=0)

                if mean == 1:
                    I2 = sub_image1 / 255.
                else:
                    I2 = sub_image1
                test_image = np.zeros([1, 512, 512, 1])
                test_image[0, :, :, 0] = I2
                pred_out = model.predict(test_image)
                #a = pred_out[0, :, :, 0]
                a = ndimage.interpolation.zoom(pred_out[0, :, :, 0], image_size / 512)
                if (row > img.shape[0] - image_size and col > img.shape[1] - image_size):
                    stitched_output[row:img.shape[0], col: img.shape[1]] = a[0:img.shape[0] - row, 0:img.shape[1] - col]
                elif (row > img.shape[0] - image_size and col < img.shape[1] - image_size):
                    stitched_output[row:img.shape[0], col: col + image_size] = a[0:img.shape[0] - row, :]
                elif (row < img.shape[0] - image_size and col > img.shape[1] - image_size):
                    stitched_output[row:row + image_size, col: img.shape[1]] = a[:, 0:img.shape[1] - col]
                else:
                    stitched_output[row:row + image_size, col: col + image_size] = a
    K.clear_session()
    return stitched_output


def cal_dice(pred_images_path, gt_images_path):
    onlyfiles = [f for f in listdir(pred_images_path)]
    print('Number of test files:', len(onlyfiles))
    csvfile = open(os.path.join(pred_images_path, 'dice_metrics2.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(['image', 'dice'])

    k = 0
    d_sim = []
    for i in range(0, len(onlyfiles)):
        if onlyfiles[i][0] is not '.':
            file = join(pred_images_path, onlyfiles[i])
            filename = onlyfiles[i]
            ground_truth = (join(gt_images_path, filename))
            ground_truth = ground_truth[:-4]
            print(file)
            print(ground_truth)
            pred_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            gt_img = cv2.imread(ground_truth, cv2.COLOR_BGR2GRAY)
            gt_img1 = np.array(gt_img, dtype=float)
            pred_img1 = np.array(pred_img, dtype=float)
            dice_score = dice_coef(pred_img1, gt_img1)
            # dice_numpy = dice_score.eval(session=tf.compat.v1.Session())
            dice_numpy = dice_score.numpy()
            print(dice_numpy)
            d_sim.append(dice_numpy)
            iteration_metrics_writer.writerow([onlyfiles[i], d_sim[k]])
            k = k + 1
    iteration_metrics_writer.writerow(['mean', sum(d_sim) / len(d_sim)])
    csvfile.flush()



def from_txt_files(model, model_weights, input_images_path, gt_images_path, dest_path, img_txt, gt_img_txt,
                  sub_image_size,save_images=False):
    #model.load_weights(model_weights)
    #print(model.weights)
    print('yes')
    # onlyfiles = [f for f in listdir(input_images_path)]
    # onlyfiles_gt = [f for f in listdir(gt_images_path)]
    text_file = open(img_txt, "r")
    onlyfiles = text_file.read().splitlines()
    text_file.close()
    text_file1 = open(gt_img_txt, "r")
    onlyfiles_gt = text_file1.read().splitlines()
    text_file1.close()
    print('Number of test files:', len(onlyfiles))
    csvfile = open(os.path.join(dest_path, 'dice_metrics.csv'), 'w', newline='')
    iteration_metrics_writer = csv.writer(csvfile)
    iteration_metrics_writer.writerow(['image', 'dice', 'dice_threshold', 'fiou', 'dice_loftis'])

    csvfile1 = open(os.path.join(dest_path, 'dice_metrics_mean.csv'), 'w', newline='')
    iteration_metrics_writer1 = csv.writer(csvfile1)
    iteration_metrics_writer1.writerow(['image', 'mean_dice', 'mean_dice_threshold', 'mean_fiou', 'mean_dice'])

    k = 0
    d_sim = []
    d_sim_t = []
    foreground_iou_list = []
    foreground_f_score_list = []
    all_us=[]
    all_os=[]
    all_jac=[]
    all_olc=[]
    all_olc1=[]
    for i in range(0, len(onlyfiles)):
        if onlyfiles[i][0] is not '.':
            file1 = join(input_images_path, onlyfiles[i])
            ground_truth = (join(gt_images_path, onlyfiles_gt[i]))
            gt_img = cv2.imread(ground_truth, cv2.COLOR_BGR2GRAY)
            gt_img1 = np.array(gt_img, dtype=float)
            gt_img1[gt_img1 == 255] = 1
            ground_truth = gt_img1
            print(file1)
            print(ground_truth)

            stitched_result1 = corrected_predicted_image(model, model_weights,file1, sub_image_size, 1)
            stitched_result = np.array(stitched_result1, dtype=float)
            dice_score = get_dice(stitched_result, ground_truth)
            #dice_numpy = dice_score.eval(session=tf.compat.v1.Session())
            dice_numpy = dice_score.numpy()
            #print(dice_numpy)

            dice_score_t = get_dice_thres(stitched_result, ground_truth)
            #dice_numpy_t = dice_score_t.eval(session=tf.compat.v1.Session())
            dice_numpy_t = dice_score_t.numpy()
            #print(dice_numpy_t)

            foreground_iou = compute_iou_for_class(stitched_result, ground_truth, cls=1)
            if foreground_iou is not None:
                foreground_iou_list.append(foreground_iou)
            else:
                foreground_iou_list.append(0)
            foreground_f_score = compute_F_score_for_class(stitched_result, ground_truth, cls=1)
            if foreground_f_score is not None:
                foreground_f_score_list.append(foreground_f_score)
            else:
                foreground_f_score_list.append(0)

            pred_img1 = stitched_result
            pred_img1[pred_img1 < 0.5] = 0
            pred_img1[pred_img1 >= 0.5] = 1
            us,osg=seg(gt_img1,pred_img1)
            jac = jaccard(gt_img1,pred_img1)
            olc = overlap_coef(gt_img1, pred_img1)
            olc1 = overlap_coef1(gt_img1,pred_img1)
            d_sim.append(dice_numpy)
            d_sim_t.append(dice_numpy_t)
            all_us.append(us)
            all_os.append(osg)
            all_jac.append(jac)
            all_olc.append(olc)
            all_olc1.append(olc1)
            iteration_metrics_writer.writerow(
                [onlyfiles[i], d_sim[k], d_sim_t[k], foreground_iou_list[k], foreground_f_score_list[k],all_us[k],all_os[k],all_jac[k],all_olc[k],all_olc1[k]])
            if save_images == True:
                raw_name = join(dest_path, onlyfiles[i])
                raw_name = raw_name + '.png'
                stitched_result = stitched_result * 255
                cv2.imwrite(raw_name, stitched_result)
            k = k + 1
    iteration_metrics_writer.writerow(['mean', sum(d_sim) / len(d_sim), sum(d_sim_t) / len(d_sim_t),
                                       sum(foreground_iou_list) / len(foreground_iou_list),
                                       sum(foreground_f_score_list) / len(foreground_f_score_list),sum(all_us)/len(all_us),
                                       sum(all_os)/len(all_os),
                                       sum(all_jac)/len(all_jac),
                                       sum(all_olc)/len(all_olc),
                                       sum(all_olc1)/len(all_olc1)])
    iteration_metrics_writer1.writerow(
        ['final_mean', sum(d_sim) / len(d_sim), sum(d_sim_t) / len(d_sim_t),
         sum(foreground_iou_list) / len(foreground_iou_list),
         sum(foreground_f_score_list) / len(foreground_f_score_list),sum(all_us)/len(all_us),
                                       sum(all_os)/len(all_os),
                                       sum(all_jac)/len(all_jac),
                                       sum(all_olc)/len(all_olc),
                                       sum(all_olc1)/len(all_olc1)])
    csvfile.flush()
    csvfile1.flush()




def compute_iou_for_class(predicted, actual, cls=1, threshold=0.5):
    predicted, actual = predicted.flatten(), actual.flatten()
    if len(predicted) != len(actual):
        raise ValueError('The two vectors are not of equal size')

    predicted = predicted >= threshold
    actual = actual >= threshold

    true_positive = np.sum(np.logical_and(actual == cls, predicted == cls) * 1)
    false_positive = np.sum(np.logical_and(actual != cls, predicted == cls) * 1)
    false_negative = np.sum(np.logical_and(actual == cls, predicted != cls) * 1)
    intersection = int(true_positive)
    union = int(true_positive + false_positive + false_negative)
    try:
        iou = intersection / union
    except ZeroDivisionError:
        return None
    return iou


def compute_F_score_for_class(predicted, actual, cls=1, threshold=0.5):
    predicted, actual = predicted.flatten(), actual.flatten()
    if len(predicted) != len(actual):
        raise ValueError('The two vectors are not of equal size')

    predicted = predicted >= threshold
    actual = actual >= threshold

    true_positive = np.sum(np.logical_and(actual == cls, predicted == cls) * 1)
    false_positive = np.sum(np.logical_and(actual != cls, predicted == cls) * 1)
    false_negative = np.sum(np.logical_and(actual == cls, predicted != cls) * 1)

    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return None
    return f_score

def seg(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)

    intersection = np.logical_and(x, y)

    und = x.sum() - intersection.sum()
    ovr = y.sum() - intersection.sum()

    return (und.astype(float) / x.sum().astype(float), ovr.astype(float) / y.sum().astype(float))


def overlap_coef(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    intersection = np.logical_and(im1, im2)

    return intersection.sum().astype(float) / im2.sum().astype(float)


def overlap_coef1(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    intersection = np.logical_and(im1, im2)

    return intersection.sum().astype(float) / im1.sum().astype(float)

def jaccard(x,y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)

    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

