from tensorflow import keras
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization, LayerNormalization, Flatten, GlobalAveragePooling1D
from keras.layers import Add, Concatenate, Lambda, Input, Permute
from keras.constraints import max_norm

from keras import backend as K
from attention_models import attention_block,eca_attention, improved_cbam_block

def DB_ATCNet(n_classes, in_chans=22, in_samples=1125, n_windows=3, attention=None,
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=8, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average',drop1=0.35,drop2=0.1,drop3=0.15,drop4=0.15,depth1=2,depth2=4):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)

    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # ADBC Block
    block1 = ADBC(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                            kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                            in_chans=in_chans, dropout=eegn_dropout,drop1=drop1,depth1=depth1,depth2=depth2)
    # Improved CBAM (was ECA2)
    block1 = improved_cbam_block(block1)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    # Sliding window
    sw_concat = []  # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # The ATFC block includes the following MHA block and TCFN block

        # MHA Block
        block2 = attention_block(block2,attention)
        # TCFN Block
        block3 = TCFN(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                               kernel_size=tcn_kernelSize, filters=tcn_filters,
                               dropout=tcn_dropout, activation=tcn_activation,drop2=drop2,drop3=drop3,drop4=drop4)

        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if (fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_constraint=max_norm(regRate))(block3))
        elif (fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if (fuse == 'average'):
        if len(sw_concat) > 1:  # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:  # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif (fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(sw_concat)

    softmax = Activation('softmax', name='softmax')(sw_concat)
    return Model(inputs=input_1, outputs=softmax)

def ADBC(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1,drop1=0.3,depth1=2,depth2=4):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    # Improved CBAM (was ECA1)
    block1 = eca_attention(block1)


    # Brunch 1
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=depth1,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    # SeparableConv2D
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)


    # Brunch 2
    block4 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=depth2,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block4 = BatchNormalization(axis=-1)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D((8, 1), data_format='channels_last')(block4)
    block4 = Dropout(dropout)(block4)
    # SeparableConv2D
    block5 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block4)
    block5 = BatchNormalization(axis=-1)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block5)
    block5 = Dropout(dropout)(block5)

    # Brunch 1 add Brunch 2
    out = Add()([block3, block5])

    out = Dropout(drop1)(out)

    return out

def TCFN(input_layer, input_dimension, depth, kernel_size, filters, dropout,drop2,drop3,drop4,activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        # Residual Connection 1
        input_layer = Dropout(drop2)(input_layer)
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)

        # Residual Connection 2
        input_layer = Dropout(drop3)(input_layer)
        block = Add()([block, input_layer])
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)

        # Residual Connection 3
        input_layer = Dropout(drop4)(input_layer)
        added = Add()([block, input_layer])
        out = Activation(activation)(added)

    return out

# %% The proposed ATCNet model, https://doi.org/10.1109/TII.2022.3197419
def ATCNet(n_classes, in_chans=22, in_samples=1125, n_windows=3, attention=None,
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=8, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):
    """ ATCNet model from Altaheri et al 2022.
        See details at https://ieeexplore.ieee.org/abstract/document/9852687

        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors

        References
        ----------
        .. H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics, 2022,
           doi: 10.1109/TII.2022.3197419.
    """
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)
    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    # Sliding window
    sw_concat = []  # to store concatenated or averaged sliding window outputs
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # Attention_model
        if attention is not None:
            block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCN_block(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                           kernel_size=tcn_kernelSize, filters=tcn_filters,
                           dropout=tcn_dropout, activation=tcn_activation)
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if (fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_constraint=max_norm(regRate))(block3))
        elif (fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if (fuse == 'average'):
        if len(sw_concat) > 1:  # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:  # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif (fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(sw_concat)

    softmax = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=softmax)


# %% Convolutional (CV) block used in the ATCNet model
def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    """ Conv_block

        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


# %% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):
    """ TCN_block from Bai et al 2018
        Temporal Convolutional Network (TCN)

        Notes
        -----
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622

        References
        ----------
        .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           An empirical evaluation of generic convolutional and recurrent networks
           for sequence modeling.
           arXiv preprint arXiv:1803.01271.
    """

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out


# %% Reproduced TCNet_Fusion model: https://doi.org/10.1016/j.bspc.2021.102826
def TCNet_Fusion(n_classes, Chans=22, Samples=1125, layers=2, kernel_s=4, filt=12,
                 dropout=0.3, activation='elu', F1=24, D=2, kernLength=32, dropout_eeg=0.3):
    """ TCNet_Fusion model from Musallam et al 2021.
    See details at https://doi.org/10.1016/j.bspc.2021.102826

        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors

        References
        ----------
        .. Musallam, Y.K., AlFassam, N.I., Muhammad, G., Amin, S.U., Alsulaiman,
           M., Abdul, W., Altaheri, H., Bencherif, M.A. and Algabri, M., 2021.
           Electroencephalography-based motor imagery classification
           using temporal convolutional network fusion.
           Biomedical Signal Processing and Control, 69, p.102826.
    """
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25

    numFilters = F1
    F2 = numFilters * D

    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    FC = Flatten()(block2)

    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                     dropout=dropout, activation=activation)

    Con1 = Concatenate()([block2, outs])
    out = Flatten()(Con1)
    Con2 = Concatenate()([out, FC])
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(Con2)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


# %% Reproduced EEGTCNet model: https://arxiv.org/abs/2006.00622
def EEGTCNet(n_classes, Chans=22, Samples=1125, layers=2, kernel_s=4, filt=12, dropout=0.3, activation='elu', F1=8, D=2,
             kernLength=32, dropout_eeg=0.2):
    """ EEGTCNet model from Ingolfsson et al 2020.
    See details at https://arxiv.org/abs/2006.00622

    The original code for this model is available at https://github.com/iis-eth-zurich/eeg-tcnet

        Notes
        -----
        The initial values in this model are based on the values identified by the authors

        References
        ----------
        .. Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N.,
           Cavigelli, L., & Benini, L. (2020, October).
           Eeg-tcnet: An accurate temporal convolutional network
           for embedded motor-imagery brain–machine interfaces.
           In 2020 IEEE International Conference on Systems,
           Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
    """
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25
    numFilters = F1
    F2 = numFilters * D

    EEGNet_sep = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[:, :, -1, :])(EEGNet_sep)
    outs = TCN_block(input_layer=block2, input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                     dropout=dropout, activation=activation)
    out = Lambda(lambda x: x[:, -1, :])(outs)
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(out)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


# %% Reproduced EEGNeX model: https://arxiv.org/abs/2207.12369
def EEGNeX_8_32(n_timesteps, n_features, n_outputs):
    """ EEGNeX model from Chen et al 2022.
    See details at https://arxiv.org/abs/2207.12369

    The original code for this model is available at https://github.com/chenxiachan/EEGNeX

        References
        ----------
        .. Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2022).
           Toward reliable signals decoding for electroencephalogram:
           A benchmark study to EEGNeX. arXiv preprint arXiv:2207.12369.
    """

    model = Sequential()
    model.add(Input(shape=(1, n_features, n_timesteps)))

    model.add(Conv2D(filters=8, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))

    model.add(DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias=False,
                              depthwise_constraint=max_norm(1.), data_format="channels_first"))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(AveragePooling2D(pool_size=(1, 4), padding='same', data_format="channels_first"))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 2),
                     data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))

    model.add(Conv2D(filters=8, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 4),
                     data_format='channels_first'))
    model.add(LayerNormalization())
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_outputs, kernel_constraint=max_norm(0.25)))
    model.add(Activation(activation='softmax'))

    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# %% Reproduced EEGNet model: https://arxiv.org/abs/1611.08024
def EEGNet_classifier(n_classes, Chans=22, Samples=1125, F1=8, D=2, kernLength=64, dropout_eeg=0.25):
    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25

    eegnet = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    eegnet = Flatten()(eegnet)
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(eegnet)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def EEGNet(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25):
    """ EEGNet model from Lawhern et al 2018
    See details at https://arxiv.org/abs/1611.08024

    The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels

        Notes
        -----
        The initial values in this model are based on the values identified by the authors

        References
        ----------
        .. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
           S. M., Hung, C. P., & Lance, B. J. (2018).
           EEGNet: A Compact Convolutional Network for EEG-based
           Brain-Computer Interfaces.
           arXiv preprint arXiv:1611.08024.
    """
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, Chans), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1),
                             data_format='channels_last',
                             use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


# %% Reproduced DeepConvNet model: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
def DeepConvNet(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

    The original code for this model is available at:
        https://github.com/braindecode/braindecode

    This implementation is taken from code by the Army Research Laboratory (ARL)
    at https://github.com/vlawhern/arl-eegmodels

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """

    # start the model
    # input_main   = Input((Chans, Samples, 1))
    input_main = Input((1, Chans, Samples))
    input_2 = Permute((2, 3, 1))(input_main)

    block1 = Conv2D(25, (1, 5),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_2)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# %% need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


# %% Reproduced ShallowConvNet model: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

    The original code for this model is available at:
        https://github.com/braindecode/braindecode

    This implementation is taken from code by the Army Research Laboratory (ARL)
    at https://github.com/vlawhern/arl-eegmodels

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    # input_main   = Input((Chans, Samples, 1))
    input_main = Input((1, Chans, Samples))
    input_2 = Permute((2, 3, 1))(input_main)

    block1 = Conv2D(40, (1, 13),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_2)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

# %% Multi-Scale ADBC Block with 3 temporal kernel sizes for capturing different oscillation frequencies
def ADBC_MultiScale(input_layer, F1=4, kernLengths=(16, 32, 64), poolSize=8, D=2, 
                    in_chans=22, dropout=0.1, drop1=0.4, depth1=1, depth2=2):
    """ ADBC with Multi-Scale Temporal Convolutions
    
        This block uses 3 parallel temporal convolutions with different kernel sizes
        to capture different EEG frequency bands (inspired by EEGNet-Fusion):
        - kernel=16:  captures beta band (~13-30Hz)
        - kernel=32:  captures alpha/mu band (~8-13Hz)  
        - kernel=64:  captures theta band (~4-8Hz)
        
        The multi-scale features are concatenated and processed by the dual-branch
        spatial processing. Using 3 kernels reduces parameter count while still
        capturing the most relevant motor imagery frequency bands.
    """
    F2 = F1 * D
    
    # Multi-scale temporal convolutions
    temporal_branches = []
    for kern_size in kernLengths:
        branch = Conv2D(F1, (kern_size, 1), padding='same', 
                        data_format='channels_last', use_bias=False)(input_layer)
        branch = BatchNormalization(axis=-1)(branch)
        temporal_branches.append(branch)
    
    # Concatenate all temporal branches (F1 * 5 = 80 filters if F1=16)
    num_kernels = len(kernLengths)
    multi_scale_channels = F1 * num_kernels  # 80
    
    block1 = Concatenate(axis=-1)(temporal_branches)
    block1 = BatchNormalization(axis=-1)(block1)
    
    # CBAM attention on FULL multi-scale features (80 channels)
    # This allows attention to learn which frequency bands are most important
    block1 = improved_cbam_block(block1)

    # ==================== BRANCH 1 ====================
    # With depth1=1: 80 × 1 = 80 (no expansion)
    # Gradual reduction: 80 → 40 → 32 (F2)
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=depth1,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    
    # Gradual reduction: 80 → 40 (2×)
    block3 = Conv2D(multi_scale_channels // 2, (8, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    
    # Final reduction: 40 → 32 (F2) (1.25×)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block3)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)

    # ==================== BRANCH 2 ====================
    # With depth2=2: 80 × 2 = 160 (modest expansion)
    # Gradual reduction: 160 → 80 → 32 (F2)
    block4 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=depth2,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block4 = BatchNormalization(axis=-1)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D((8, 1), data_format='channels_last')(block4)
    block4 = Dropout(dropout)(block4)
    
    # Gradual reduction: 160 → 80 (2×)
    block5 = Conv2D(multi_scale_channels, (8, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block4)
    block5 = BatchNormalization(axis=-1)(block5)
    block5 = Activation('elu')(block5)
    
    # Final reduction: 80 → 32 (F2) (2.5×)
    block5 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block5)
    block5 = BatchNormalization(axis=-1)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block5)
    block5 = Dropout(dropout)(block5)

    # Merge dual branches
    out = Add()([block3, block5])
    out = Dropout(drop1)(out)

    return out


# %% DB-ATCNet with Multi-Scale temporal convolutions
def DB_ATCNet_MultiScale(n_classes, in_chans=22, in_samples=1125, n_windows=3, attention=None,
                         eegn_F1=16, eegn_D=2, eegn_kernelSizes=(16, 32, 64), 
                         eegn_poolSize=8, eegn_dropout=0.4,
                         tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                         tcn_activation='elu', fuse='average',
                         drop1=0.45, drop2=0.1, drop3=0.15, drop4=0.15, depth1=1, depth2=2):
    """ DB-ATCNet with Multi-Scale Temporal Convolutions
    
        This variant uses 3 parallel temporal convolutions with different kernel sizes
        to capture motor imagery-relevant EEG oscillation frequencies (beta, alpha/mu, theta).
        
        Architecture:
        Input -> Multi-Scale Conv (3 kernels) -> Concat -> CBAM ->
        Dual-Branch Spatial Processing -> Sliding Window -> MHA -> TCFN -> Output
        
        Parameters
        ----------
        eegn_kernelSizes : tuple
            Tuple of 3 kernel sizes for multi-scale temporal convolutions.
            Default: (16, 32, 64) covering beta, alpha/mu, and theta bands.
    """
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)

    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # Multi-Scale ADBC Block
    block1 = ADBC_MultiScale(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                             kernLengths=eegn_kernelSizes, poolSize=eegn_poolSize,
                             in_chans=in_chans, dropout=eegn_dropout,
                             drop1=drop1, depth1=depth1, depth2=depth2)
    
    # Improved CBAM after ADBC (same as original DB-ATCNet which also has 2 CBAMs)
    block1 = improved_cbam_block(block1)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    # Sliding window
    sw_concat = []
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # MHA Block
        block2 = attention_block(block2, attention)
        
        # TCFN Block
        block3 = TCFN(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                      kernel_size=tcn_kernelSize, filters=tcn_filters,
                      dropout=tcn_dropout, activation=tcn_activation,
                      drop2=drop2, drop3=drop3, drop4=drop4)

        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window
        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_constraint=max_norm(regRate))(block3))
        elif fuse == 'concat':
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if fuse == 'average':
        if len(sw_concat) > 1:
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:
            sw_concat = sw_concat[0]
    elif fuse == 'concat':
        sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(sw_concat)

    softmax = Activation('softmax', name='softmax')(sw_concat)
    return Model(inputs=input_1, outputs=softmax)


# %% Efficient Multi-Scale ADBC Block using Dilated Convolutions + SE Attention
def ADBC_EfficientMultiScale(input_layer, F1=16, kernLength=32, poolSize=8, D=2, 
                              in_chans=22, dropout=0.1, drop1=0.35, depth1=2, depth2=4,
                              dilation_rates=(1, 2, 4), se_ratio=4):
    """ ADBC with Efficient Multi-Scale via Dilated Convolutions + Squeeze-Excitation
    
        This block captures multiple temporal scales using dilated convolutions,
        which is much more parameter-efficient than parallel convolution branches.
        
        Multi-scale philosophy:
        - dilation_rate=1: Captures fast oscillations (~beta band, 13-30Hz)
        - dilation_rate=2: Captures medium oscillations (~alpha/mu band, 8-13Hz)  
        - dilation_rate=4: Captures slow oscillations (~theta band, 4-8Hz)
        
        Key design choices from efficient CV models:
        1. Dilated convs instead of parallel branches (from WaveNet/EfficientNet)
           - Same parameters, 3x receptive field coverage
        2. Additive fusion instead of concatenation (from HRNet/Lite-HRNet)
           - No channel explosion, preserves original dimensionality
        3. Squeeze-Excitation for adaptive band weighting (from MobileNetV3)
           - Learns to weight frequency bands per sample
        4. Inverted bottleneck in spatial branch (from MobileNetV2)
           - Expand for processing, compress for efficiency
           
        Parameters
        ----------
        dilation_rates : tuple of int
            Dilation rates for multi-scale temporal convolutions.
            Default: (1, 2, 4) covering beta, alpha, theta bands.
        se_ratio : int
            Squeeze-Excitation reduction ratio. Default: 4.
    """
    from keras.layers import GlobalAveragePooling2D, Reshape, multiply
    
    F2 = F1 * D
    
    # ==================== MULTI-SCALE TEMPORAL FEATURE EXTRACTION ====================
    # Single learned kernel applied at multiple dilation rates
    # This is ~3x cheaper than 3 parallel conv branches
    
    # Base temporal features with standard kernel
    base_temporal = Conv2D(F1, (kernLength, 1), padding='same', 
                          data_format='channels_last', use_bias=False)(input_layer)
    base_temporal = BatchNormalization(axis=-1)(base_temporal)
    
    # Multi-scale via dilated convolutions (additive fusion - no channel explosion)
    multi_scale_features = []
    for dilation in dilation_rates:
        # Each dilated conv captures a different temporal scale
        # Effective receptive field = kernLength + (kernLength - 1) * (dilation - 1)
        dilated_branch = Conv2D(F1, (kernLength // 2, 1), 
                                padding='same', 
                                dilation_rate=(dilation, 1),
                                data_format='channels_last', 
                                use_bias=False)(base_temporal)
        multi_scale_features.append(dilated_branch)
    
    # Additive fusion - preserves channel count (F1), not concat which would give F1*3
    block1 = Add()(multi_scale_features)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    
    # SE attention on multi-scale features
    # SE = pure channel attention, ideal for weighting which frequency bands matter
    # CBAM's spatial attention isn't useful here since spatial processing hasn't happened yet
    # The second CBAM after dual-branch spatial processing will handle both channel + spatial
    from attention_models import se_block
    block1 = se_block(block1, ratio=se_ratio)
    
    # ==================== DUAL-BRANCH SPATIAL PROCESSING ====================
    # Keeping original dual-branch structure for spatial processing
    
    # Branch 1: Focused spatial features (lower depth multiplier)
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=depth1,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    
    # Pointwise conv to project to F2 (efficient 1x1 projection)
    block3 = Conv2D(F2, (16, 1), data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)

    # Branch 2: Diverse spatial features (higher depth multiplier)
    block4 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=depth2,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block4 = BatchNormalization(axis=-1)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D((8, 1), data_format='channels_last')(block4)
    block4 = Dropout(dropout)(block4)
    
    # Pointwise conv to project to F2
    block5 = Conv2D(F2, (16, 1), data_format='channels_last',
                    use_bias=False, padding='same')(block4)
    block5 = BatchNormalization(axis=-1)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block5)
    block5 = Dropout(dropout)(block5)

    # Merge dual branches
    out = Add()([block3, block5])
    out = Dropout(drop1)(out)

    return out


# %% DB-ATCNet with Efficient Multi-Scale (Dilated Convs + SE Attention)
def DB_ATCNet_EfficientMultiScale(n_classes, in_chans=22, in_samples=1125, n_windows=3, attention='mha',
                                   eegn_F1=16, eegn_D=2, eegn_kernelSize=32, 
                                   eegn_poolSize=8, eegn_dropout=0.3,
                                   tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                                   tcn_activation='elu', fuse='average',
                                   drop1=0.35, drop2=0.1, drop3=0.15, drop4=0.15, 
                                   depth1=2, depth2=4,
                                   dilation_rates=(1, 2, 4), se_ratio=4):
    """ DB-ATCNet with Efficient Multi-Scale Temporal Processing
    
        This variant uses parameter-efficient multi-scale techniques inspired by
        state-of-the-art lightweight computer vision models:
        
        Key innovations:
        1. Dilated Convolutions (WaveNet/EfficientNet-inspired)
           - One kernel, multiple receptive fields via dilation
           - Captures beta (fast), alpha (medium), theta (slow) oscillations
           - ~3x fewer parameters than parallel multi-scale branches
           
        2. Squeeze-Excitation Attention (MobileNetV3-inspired)
           - Adaptively weights frequency bands per sample
           - Handles subject/trial variability in band importance
           
        3. Additive Fusion (HRNet/Lite-HRNet-inspired)
           - Sum multi-scale features instead of concatenate
           - No channel explosion, maintains compact representation
           
        Compared to DB_ATCNet_MultiScale:
        - ~40% fewer parameters
        - No abrupt channel drops
        - Adaptive per-sample band weighting
        
        Parameters
        ----------
        dilation_rates : tuple of int
            Dilation rates for multi-scale temporal convolutions.
            Default: (1, 2, 4) for beta/alpha/theta frequency bands.
        se_ratio : int
            Squeeze-Excitation reduction ratio. Default: 4.
    """
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)

    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # Efficient Multi-Scale ADBC Block
    block1 = ADBC_EfficientMultiScale(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                                       kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                                       in_chans=in_chans, dropout=eegn_dropout,
                                       drop1=drop1, depth1=depth1, depth2=depth2,
                                       dilation_rates=dilation_rates, se_ratio=se_ratio)
    
    # Second CBAM after ADBC (same as original DB-ATCNet)
    block1 = improved_cbam_block(block1)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    # Sliding window (same as original DB-ATCNet)
    sw_concat = []
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        # MHA Block
        block2 = attention_block(block2, attention)
        
        # TCFN Block
        block3 = TCFN(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                      kernel_size=tcn_kernelSize, filters=tcn_filters,
                      dropout=tcn_dropout, activation=tcn_activation,
                      drop2=drop2, drop3=drop3, drop4=drop4)

        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window
        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_constraint=max_norm(regRate))(block3))
        elif fuse == 'concat':
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if fuse == 'average':
        if len(sw_concat) > 1:
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:
            sw_concat = sw_concat[0]
    elif fuse == 'concat':
        sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(sw_concat)

    softmax = Activation('softmax', name='softmax')(sw_concat)
    return Model(inputs=input_1, outputs=softmax)


# %% DB-ATCNet with Task-Conditioned Prompt Learning (TCPL)
# Based on: Wang et al. 2025, "TCPL: Task-Conditioned Prompt Learning for 
# Few-Shot Cross-Subject Motor Imagery EEG Decoding"
def DB_ATCNet_TCPL(n_classes, in_chans=22, in_samples=1125, n_windows=3, 
                   n_prompts=10, prompt_dim=32,
                   eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=8, eegn_dropout=0.3,
                   tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                   tcn_activation='elu', fuse='average',
                   drop1=0.35, drop2=0.1, drop3=0.15, drop4=0.15, depth1=2, depth2=4):
    """
    DB-ATCNet with Task-Conditioned Prompt Learning for few-shot cross-subject MI decoding.
    
    This model is designed for meta-learning training where subject-specific prompts
    are generated from a few-shot support set and injected into the MHA block.
    
    Architecture (minimal changes from DB_ATCNet):
    1. ADBC Block (unchanged)
    2. ECA Attention (unchanged) 
    3. Prompt-Augmented MHA (prompts prepended to input sequence)
    4. TCFN Block (unchanged)
    5. Classification head (unchanged)
    
    Parameters
    ----------
    n_prompts : int
        Number of prompt tokens to generate (default: 10)
    prompt_dim : int  
        Dimension of each prompt token. Should match F2 (eegn_F1 * eegn_D) for compatibility.
        
    Notes
    -----
    This model requires:
    1. TCPModule from tcpl_modules.py to generate prompts from support set
    2. Meta-learning training loop (see Physionet_main_TCPL.py)
    3. Episodic data sampling with support/query splits
    
    The model itself does NOT include the TCP module - prompts are passed in during
    forward pass. This allows flexible training strategies.
    
    References
    ----------
    Wang et al. 2025. TCPL: Task-Conditioned Prompt Learning for Few-Shot 
    Cross-Subject Motor Imagery EEG Decoding. Front. Neurosci. 19:1689286.
    doi: 10.3389/fnins.2025.1689286
    """

    from tcpl_modules import TCPL_Backbone
    
    # Two inputs: EEG data and prompt tokens
    input_eeg = Input(shape=(1, in_chans, in_samples), name='eeg_input')  
    input_prompts = Input(shape=(n_prompts, prompt_dim), name='prompt_input')
    
    input_2 = Permute((3, 2, 1))(input_eeg)

    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # ADBC Block (Feature Extractor)
    block1 = ADBC(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                  kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                  in_chans=in_chans, dropout=eegn_dropout,
                  drop1=drop1, depth1=depth1, depth2=depth2)
    
    # ECA Attention
    block1 = eca_attention(block1)
    # Remove dimension 2 (channels=1) -> (Batch, Time, Features)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
    
    # TCPL Backbone (4-layer Transformer with Deep Prompt Injection)
    # Replaces TCFN and Sliding Window
    # d_model = F2 = 32. We use 2 heads, key_dim=16.
    backbone = TCPL_Backbone(
        n_layers=4, 
        key_dim=16, 
        num_heads=2, 
        dropout=0.5, 
        n_prompts=n_prompts
    )
    
    # Pass features and prompts to backbone
    x = backbone([block1, input_prompts])
    
    # Classification Head
    # Global Average Pooling over time tokens
    x = GlobalAveragePooling1D()(x)
    
    softmax = Dense(n_classes, activation='softmax', kernel_constraint=max_norm(regRate))(x)
    
    # Function ends here, returning the model
    return Model(inputs=[input_eeg, input_prompts], outputs=softmax, name='DB_ATCNet')


def get_tcpl_backbone(n_classes, in_chans=22, in_samples=1125, n_windows=3,
                      n_prompts=10, prompt_dim=32, **kwargs):
    """
    Convenience function to create DB_ATCNet_TCPL with default parameters.
    
    Returns both the model and a helper function to create prompts.
    
    Usage
    -----
    model, create_prompts = get_tcpl_backbone(n_classes=4, in_chans=64, in_samples=640)
    
    # During training/inference:
    prompts = create_prompts(support_set)  # (n_prompts, prompt_dim)
    predictions = model([query_samples, prompts])
    """
    from tcpl_modules import TCPModule
    
    model = DB_ATCNet_TCPL(
        n_classes=n_classes,
        in_chans=in_chans, 
        in_samples=in_samples,
        n_windows=n_windows,
        n_prompts=n_prompts,
        prompt_dim=prompt_dim,
        **kwargs
    )
    
    tcp_module = TCPModule(n_prompts=n_prompts, prompt_dim=prompt_dim)
    
    return model, tcp_module
