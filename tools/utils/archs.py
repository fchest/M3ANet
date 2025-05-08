from keras.models import Sequential,Model
from keras.layers import Dense, Activation,LSTM,Lambda,MaxPooling1D,SeparableConv1D,TimeDistributed,Masking,Reshape,Flatten,Conv2D,Conv1D,Dropout,MaxPooling2D,UpSampling2D,BatchNormalization,Multiply,Input,Add,PReLU,UpSampling1D,Concatenate,LeakyReLU

def arch001(config):
    print('Building model with implementation arch001...')
    input = Input(shape=(None,1))

    #Encoder 9
    enc9 = Conv1D(64, config['filter_size'], padding='same', strides=1 )(input)
    enc9 = PReLU(shared_axes=[1],name='enc9_out')(enc9)

    #Encoder 8
    enc8 = Conv1D(64, config['filter_size'], padding='same', strides=2 )(enc9)
    enc8 = PReLU(shared_axes=[1],name='enc8_out')(enc8)       

    #Encoder 7
    enc7 = Conv1D(64, config['filter_size'], padding='same', strides=2 )(enc8)
    enc7 = PReLU(shared_axes=[1],name='enc7_out')(enc7) 
    enc7 = Dropout(0.2) (enc7)       

    #Encoder 6   
    enc6 = Conv1D(128, config['filter_size'], padding='same', strides=2 )(enc7)
    enc6 = PReLU(shared_axes=[1],name='enc6_out')(enc6) 

    #Encoder 5
    enc5 = Conv1D(128, config['filter_size'], padding='same', strides=2 )(enc6)
    enc5 = PReLU(shared_axes=[1],name='enc5_out')(enc5) 

    #Encoder 4
    enc4 = Conv1D(128, config['filter_size'], padding='same', strides=2 )(enc5)
    enc4 = PReLU(shared_axes=[1],name='enc4_out')(enc4) 
    enc4 = Dropout(0.2) (enc4)

    #Encoder 3   
    enc3 = Conv1D(256, config['filter_size'], padding='same', strides=2 )(enc4)
    enc3 = PReLU(shared_axes=[1],name='enc3_out')(enc3) 

    #Encoder 2
    enc2 = Conv1D(256, config['filter_size'], padding='same', strides=2 )(enc3)
    enc2 = PReLU(shared_axes=[1],name='enc2_out')(enc2) 

    #Encoder 1
    enc1 = Conv1D(256, config['filter_size'], padding='same', strides=2 )(enc2)
    enc1 = PReLU(shared_axes=[1],name='enc1_out')(enc1)         
    enc1 = Dropout(0.2)(enc1)

    #Decoder 1
    dec1 = UpSampling1D()(enc1)
    dec1 = Concatenate(axis=-1)([dec1, enc2])
    dec1 = Conv1D(256, config['filter_size'], padding='same', strides=1 )(dec1)
    dec1 = PReLU(shared_axes=[1],name='dec1_out')(dec1)         

    #Decoder 2
    dec2 = UpSampling1D(size=2)(dec1)
    dec2 = Concatenate(axis=-1)([dec2, enc3])
    dec2 = Conv1D(256, config['filter_size'], padding='same', strides=1 )(dec2)
    dec2 = PReLU(shared_axes=[1],name='dec2_out')(dec2)         

    #Decoder 3
    dec3 = UpSampling1D(size=2)(dec2)
    dec3 = Concatenate(axis=-1)([dec3, enc4]) 
    dec3 = Conv1D(128, config['filter_size'], padding='same', strides=1 )(dec3)
    dec3 = PReLU(shared_axes=[1],name='dec3_out')(dec3)         
    dec3 = Dropout(0.2)(dec3)

    #Decoder 4
    dec4 = UpSampling1D(size=2)(dec3)
    dec4 = Concatenate(axis=-1)([dec4, enc5]) 
    dec4 = Conv1D(128, config['filter_size'], padding='same', strides=1 )(dec4)
    dec4 = PReLU(shared_axes=[1],name='dec4_out')(dec4)     

    #Decoder 5 
    dec5 = UpSampling1D(size=2)(dec4)
    dec5 = Concatenate(axis=-1)([dec5, enc6])
    dec5 = Conv1D(128, config['filter_size'], padding='same', strides=1 )(dec5)
    dec5 = PReLU(shared_axes=[1],name='dec5_out')(dec5)     

    #Decoder 6
    dec6 = UpSampling1D(size=2)(dec5)
    dec6 = Concatenate(axis=-1)([dec6, enc7]) 
    dec6 = Conv1D(64, config['filter_size'], padding='same', strides=1 )(dec6)
    dec6 = PReLU(shared_axes=[1],name='dec6_out')(dec6)             
    dec6 = Dropout(0.2)(dec6) 

    #Decoder 7
    dec7 = UpSampling1D(size=2)(dec6)
    dec7 = Concatenate(axis=-1)([dec7, enc8])
    dec7 = Conv1D(64, config['filter_size'], padding='same', strides=1 )(dec7)
    dec7 = PReLU(shared_axes=[1],name='dec7_out')(dec7) 

    #Decoder 8
    dec8 = UpSampling1D(size=2)(dec7)
    dec8 = Concatenate(axis=-1)([dec8, enc9])
    dec8 = Conv1D(64, config['filter_size'], padding='same', strides=1 )(dec8)
    dec8 = PReLU(shared_axes=[1],name='dec8_out')(dec8) 

    #Decoder 9
    output = Conv1D(1, 1, padding='same', strides=1, activation='tanh',name='dec9_out' )(dec8)    

    return Model(inputs=input, outputs=output) 