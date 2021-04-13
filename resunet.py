from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout,Conv2DTranspose, BatchNormalization, Add, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

smooth = 1.
#smooth = 10e-5

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def dual_dice_coef(y_true, y_pred):
    Foreground = dice_coef(y_true, y_pred)
    Background = dice_coef(1. - y_true, 1. - y_pred)
    return 0.5 * (Foreground + Background)

def dual_dice_coef_loss(y_true, y_pred):
    return 1.0 - dual_dice_coef(y_true, y_pred)


def pooling_step(x,kernel_size=(2,2)):
    pool1 = MaxPooling2D(kernel_size)(x)
    return pool1

# Resunet blocks
def first_step(x, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    conv1 = Conv2D(num_filters, kernel_size, padding=pad,strides=stride)(x)
    if bt_norm == True:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act)(conv1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad,strides=1)(conv1)

    shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding=pad, strides=stride)(x)
    shortcut = BatchNormalization()(shortcut)

    final = Add()([conv1,shortcut])
    return final

def res_conv_block(x, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    if bt_norm == True:
        x1 = BatchNormalization()(x)
    else:
        x1 = x
    conv1 = Activation(act)(x1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad, strides=stride)(conv1)
    if bt_norm == True:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act)(conv1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad, strides=1)(conv1)

    shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding=pad, strides=stride)(x)
    shortcut = BatchNormalization()(shortcut)

    final = Add()([conv1,shortcut])
    return final

def res_decoder_step(x, encoder_layer, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    deconv4 = Conv2DTranspose(num_filters, kernel_size, activation=act, padding=pad, strides=(2, 2))(x)
    if bt_norm == True:
        deconv4 = BatchNormalization()(deconv4)
    uconv4 = concatenate([deconv4, encoder_layer])
    uconv4 = res_conv_block(uconv4,num_filters,bt_norm, kernel_size, act, pad, stride)
    return uconv4

# Blocks for RESUNET - A
def conv_block(x, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1,dialation=1):
    if bt_norm == True:
        x1 = BatchNormalization()(x)
    else:
        x1 = x
    conv1 = Activation(act)(x1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad, strides=stride,dilation_rate=dialation)(conv1)
    if bt_norm == True:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act)(conv1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad, strides=1,dilation_rate=dialation)(conv1)
    return conv1


def resunet_a_block(x, num_filters, d=[1], bt_norm=True):
    if len(d) == 4:
        conv1 = conv_block(x, num_filters, bt_norm, dialation=d[0])
        conv2 = conv_block(x, num_filters, bt_norm, dialation=d[1])
        conv3 = conv_block(x, num_filters, bt_norm, dialation=d[2])
        conv4 = conv_block(x, num_filters, bt_norm, dialation=d[3])
        final = Add()([conv1,conv2,conv3,conv4])
        return final
    elif len(d) == 3:
        conv1 = conv_block(x, num_filters, bt_norm, dialation=d[0])
        conv2 = conv_block(x, num_filters, bt_norm, dialation=d[1])
        conv3 = conv_block(x, num_filters, bt_norm, dialation=d[2])
        final = Add()([conv1, conv2, conv3])
        return final
    elif len(d) == 2:
        conv1 = conv_block(x, num_filters, bt_norm, dialation=d[0])
        conv2 = conv_block(x, num_filters, bt_norm, dialation=d[1])
        final = Add()([conv1, conv2])
        return final
    elif len(d) == 1:
        conv1 = conv_block(x, num_filters, bt_norm, dialation=d[0])
        return conv1

def resunet_a_decoder_step(x, encoder_layer, num_filters,d=[1], bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    deconv4 = Conv2DTranspose(num_filters, kernel_size, activation=act, padding=pad, strides=(2, 2))(x)
    if bt_norm == True:
        deconv4 = BatchNormalization()(deconv4)
    uconv4 = concatenate([deconv4, encoder_layer])
    uconv4 = resunet_a_block(uconv4, num_filters, d, bt_norm=True)
    return uconv4


# Different models
def resunet_with_pool(start_neurons,img_rows,img_cols,bt_state=True):
    input_layer = Input((img_rows,img_cols, 1))
    if bt_state == True:
        input_layer1 = BatchNormalization()(input_layer)
    else:
        input_layer1 = input_layer

    conv1 = first_step(input_layer1,start_neurons * 1, bt_norm=bt_state)
    pool1 = pooling_step(conv1, kernel_size=(2, 2))
    conv2 = res_conv_block(pool1,start_neurons * 2, bt_norm=bt_state)
    pool2 = pooling_step(conv2, kernel_size=(2, 2))
    conv3 = res_conv_block(pool2, start_neurons * 4, bt_norm=bt_state)
    pool3 = pooling_step(conv3, kernel_size=(2, 2))
    conv4 = res_conv_block(pool3, start_neurons * 8, bt_norm=bt_state)
    pool4 = pooling_step(conv4, kernel_size=(2, 2))

    # Middle
    conv5 = res_conv_block(pool4, start_neurons * 16, bt_norm=bt_state)

    deconv4 = res_decoder_step(conv5, conv4, start_neurons * 8, bt_norm=bt_state)
    deconv3 = res_decoder_step(deconv4, conv3, start_neurons * 4, bt_norm=bt_state)
    deconv2 = res_decoder_step(deconv3, conv2, start_neurons * 2, bt_norm=bt_state)
    deconv1 = res_decoder_step(deconv2, conv1, start_neurons * 1, bt_norm=bt_state)

    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(deconv1)

    model = Model(input_layer,output_layer)

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def resunet_without_pool(start_neurons,img_rows,img_cols,bt_state=True):

    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        input_layer = Input((img_rows,img_cols, 1))
        if bt_state == True:
            input_layer1 = BatchNormalization()(input_layer)
        else:
            input_layer1 = input_layer

        conv1 = first_step(input_layer1,start_neurons * 1, bt_norm=bt_state)
        conv2 = res_conv_block(conv1,start_neurons * 2, stride=2, bt_norm=bt_state)
        conv3 = res_conv_block(conv2, start_neurons * 4, stride=2, bt_norm=bt_state)
        conv4 = res_conv_block(conv3, start_neurons * 8, stride=2,  bt_norm=bt_state)
        conv5 = res_conv_block(conv4, start_neurons * 8, stride=2, bt_norm=bt_state)

        # Middle
        conv6 = res_conv_block(conv5, start_neurons * 16, bt_norm=bt_state)

        deconv4 = res_decoder_step(conv6, conv4, start_neurons * 8, bt_norm=bt_state)
        deconv3 = res_decoder_step(deconv4, conv3, start_neurons * 4, bt_norm=bt_state)
        deconv2 = res_decoder_step(deconv3, conv2, start_neurons * 2, bt_norm=bt_state)
        deconv1 = res_decoder_step(deconv2, conv1, start_neurons * 1, bt_norm=bt_state)

        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(deconv1)

        model = Model(input_layer,output_layer)

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def resunet_a(start_neurons,img_rows,img_cols,bt_state=True):
    input_layer = Input((img_rows,img_cols, 1))
    if bt_state == True:
        input_layer1 = BatchNormalization()(input_layer)
    else:
        input_layer1 = input_layer

    conv1 = first_step(input_layer1,start_neurons * 1, bt_norm=bt_state)
    conv2 = resunet_a_block(conv1, start_neurons * 1, d= [1,3,15,31], bt_norm=bt_state)
    pool2 = Conv2D(start_neurons * 2, kernel_size=(1,1), padding="same", strides=(2,2))(conv2)
    conv3 = resunet_a_block(pool2, start_neurons * 2, d= [1,3,15,31], bt_norm=bt_state)
    pool3 = Conv2D(start_neurons * 4, kernel_size=(1, 1), padding="same", strides=(2, 2))(conv3)
    conv4 = resunet_a_block(pool3, start_neurons * 4, d= [1,3,15],  bt_norm=bt_state)
    pool4 = Conv2D(start_neurons * 8, kernel_size=(1, 1), padding="same", strides=(2, 2))(conv4)
    conv5 = resunet_a_block(pool4, start_neurons * 8, d= [1,3,15], bt_norm=bt_state)
    pool5 = Conv2D(start_neurons * 16, kernel_size=(1, 1), padding="same", strides=(2, 2))(conv5)
    conv6 = resunet_a_block(pool5, start_neurons * 16, d= [1], bt_norm=bt_state)
    pool6 = Conv2D(start_neurons * 32, kernel_size=(1, 1), padding="same", strides=(2, 2))(conv6)

    # Middle
    conv7 = resunet_a_block(pool6, start_neurons * 32, d= [1], bt_norm=bt_state)

    deconv5 = resunet_a_decoder_step(conv7, conv6, start_neurons * 16, d= [1],bt_norm=bt_state)
    deconv4 = resunet_a_decoder_step(deconv5, conv5, start_neurons * 8, d= [1,3,15],bt_norm=bt_state)
    deconv3 = resunet_a_decoder_step(deconv4, conv4, start_neurons * 4, d= [1,3,15],bt_norm=bt_state)
    deconv2 = resunet_a_decoder_step(deconv3, conv3, start_neurons * 2, d= [1,3,15,31],bt_norm=bt_state)
    deconv1 = resunet_a_decoder_step(deconv2, conv2, start_neurons * 1, d= [1,3,15,31],bt_norm=bt_state)

    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(deconv1)

    model = Model(input_layer,output_layer)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == "__main__":
    """
    testing
    """
    model = resunet_with_pool(32, 512, 512,bt_state=True)

    # model = ResUNet()
    model.summary()
