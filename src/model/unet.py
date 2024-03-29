from keras import Model, layers
from keras.applications import vgg16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPool2D, concatenate, UpSampling2D
# from tensorflow.python.keras.applications import vgg16
# from tensorflow.python.keras.layers import Activation
# from tensorflow.python.keras.layers import BatchNormalization
# from tensorflow.python.keras.layers import concatenate
# from tensorflow.python.keras.layers import Conv2D
# from tensorflow.python.keras.layers import MaxPool2D
# from tensorflow.python.keras.layers import Reshape
# from tensorflow.python.keras.layers import UpSampling2D


def UNet(nClasses, input_height, input_width):
    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 1))
    ft = Conv2D(filters=3,
                kernel_size=3,
                padding='same',
                kernel_initializer='glorot_normal',
                bias_initializer='zeros'
                )(img_input)

    md = Model(inputs=img_input, outputs=ft)
    vgg_streamlined = vgg16.VGG16(include_top=False, weights=None, input_tensor=md.output)
    assert isinstance(vgg_streamlined, Model)

    # 解码层
    o = UpSampling2D((2, 2))(vgg_streamlined.output)
    o = concatenate([vgg_streamlined.get_layer(name="block4_pool").output, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(name="block3_pool").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(name="block2_pool").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(name="block1_pool").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    # UNet网络处理输入时进行了镜面放大2倍，所以最终的输入输出缩小了2倍
    # 此处直接上采样置原始大小
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)

    # o = Reshape((-1, nClasses))(o)
    o = Activation("softmax")(o)
    # print('o shape: {}'.format(o.get_shape()))

    model = Model(inputs=img_input, outputs=o)
    return model


# if __name__ == '__main__':
#     m = UNet(5, 320, 320)
#     # print(m.get_weights()[2]) # 看看权重改变没，加载vgg权重测试用
#     # from keras.utils import plot_model
#     # plot_model(m, show_shapes=True, to_file='model_unet.png')
#     print(len(m.layers))
#     m.summary()
