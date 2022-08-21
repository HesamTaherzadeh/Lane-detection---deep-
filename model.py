import tensorflow as tf 

class Residual_block(tf.keras.Model):
    def __init__(self , filter_count, kernel_size, activation_function = 'relu'):
        super(Residual_block , self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_count, kernel_size, activation = activation_function, padding="same")
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(1, 1, activation =activation_function, padding="same")
        

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x_1 = self.batch_normalization(x)
        x = self.conv2(inputs)
        x = tf.keras.layers.Add()([x, x_1])
        x = tf.keras.layers.ReLU()(x)
        x = self.batch_normalization(x)
        return x

class Spatial_attention(tf.keras.Model):
    def __init__(self, pooling_size, conv_filtersize):
        super(Spatial_attention, self).__init__()
        self.avg_pooling = tf.keras.layers.AveragePooling2D(pooling_size)
        self.max_pooling = tf.keras.layers.AveragePooling2D(pooling_size)
        self.conv = tf.keras.layers.Conv2D(1, conv_filtersize, padding="same", activation='sigmoid')
        # self.conv2 = tf.keras.layers.Conv2D(4, 8, padding="same", activation='relu')

    def call(self, inputs, training):
        x_avg = self.avg_pooling(inputs)
        # x_avg = self.conv2(x_avg)
        x_max = self.max_pooling(inputs)
        # x_max = self.conv2(x_max)
        x = tf.keras.layers.concatenate([x_max, x_avg], axis=-1)
        x = self.conv(x)
        return x


class Channel_attention(tf.keras.Model):

    def __init__(self, first_dense_node_count, second_dense_node_count):
        super(Channel_attention, self).__init__()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.dense1 = tf.keras.layers.Dense(first_dense_node_count, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(second_dense_node_count, activation = 'relu')

    def call(self, inputs, training):
        x_gap = self.gap(inputs)
        x_gmp = self.gmp(inputs)
        dense_output_gap = self.dense1(x_gap)
        dense_output_gmp = self.dense1(x_gmp)
        second_dense_output_gap = self.dense2(dense_output_gap)
        second_dense_output_gmp = self.dense2(dense_output_gmp)
        x = tf.keras.layers.Add()([second_dense_output_gap, second_dense_output_gmp])
        x = tf.keras.layers.Activation("sigmoid")(x)
        x = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(x)
        return tf.keras.layers.Multiply()([inputs, x])


class Deconvolve(tf.keras.Model):
    def __init__(self, deconvolve_filter_count, kernel_size):
        super(Deconvolve, self).__init__()
        self.convtranspose1 = tf.keras.layers.Conv2DTranspose(deconvolve_filter_count, kernel_size, activation = 'relu', padding = 'same')
        self.convtranspose2 = tf.keras.layers.Conv2DTranspose(2 * deconvolve_filter_count, kernel_size, activation = 'relu', padding = 'same')
        self.batch_normaliztion = tf.keras.layers.BatchNormalization()
    def call(self, inputs, training):
        x = self.convtranspose1(inputs)
        x = self.batch_normaliztion(x)
        x = self.convtranspose1(x)
        x = self.batch_normaliztion(x)
        return x 

# class Attention(Channel_attention, Spatial_attention):

#     def __init__(self, first_dense_node_count, second_dense_node_count, pooling_size, conv_filtersize):
#         Channel_attention.__init__(self, first_dense_node_count, second_dense_node_count)
#         Spatial_attention.__init__(self, pooling_size, conv_filtersize)
    
#     def call(self, inputs):
#         mul_channel_input = tf.keras.Multiply()([inputs, Channel_attention.call(self, inputs)])
#         mul_mul_channel_input_spatial = tf.keras.Multiply()([mul_channel_input, Spatial_attention.call(self, inputs)])
#         return mul_mul_channel_input_spatial

class Attention(tf.keras.Model):
    def __init__(self, first_dense_node_count, second_dense_node_count, pooling_size, conv_filtersize):
        super(Attention, self).__init__()
        self.channel_attention = Channel_attention(first_dense_node_count, second_dense_node_count)
        self.spatial_attention = Spatial_attention(pooling_size, conv_filtersize)
    
    def call(self, inputs, training):
        x_1 = self.spatial_attention(inputs)
        x_2 = self.channel_attention(inputs)
        attention_result = tf.keras.layers.Multiply()([x_1, x_2])
        return attention_result

class Deep_LDW(tf.keras.Model):

    def __init__(self, shape) :
        super(Deep_LDW, self).__init__()
        self.count = shape
        self.res_block_1 = Residual_block(8, (3, 3))
        self.maxpool_2x2 = tf.keras.layers.MaxPooling2D(pool_size=2, padding="same")
        self.res_block_2 = Residual_block(16, (3, 3))
        self.res_block_3 = Residual_block(32, (3, 3))
        self.res_block_4 = Residual_block(64, (3, 3))
        self.attention = Attention(64//8, self.count* 2 , 1, (7,7)) # dont know
        self.upsample_2x2 = tf.keras.layers.UpSampling2D(size = (2,2))
        self.upsample_4x4 = tf.keras.layers.UpSampling2D(size = (4,4))
        self.upsample_16x16 = tf.keras.layers.UpSampling2D(size = (16,16))
        self.deconv_1 = Deconvolve(32, (3,3))
        self.deconv_2 = Deconvolve(32, (3,3))
        self.deconv_3 = Deconvolve(64, (3,3))

    def call(self, inputs, training=True):
        # Part-1 of training : residual bloxks
        x = self.res_block_1(inputs)
        pool_1 = self.maxpool_2x2(x)
        x = self.res_block_2(pool_1)
        pool_2 = self.maxpool_2x2(x)
        x = self.res_block_3(pool_2)
        pool_3 = self.maxpool_2x2(x)
        x = self.res_block_4(pool_3)
        pool_4 = self.maxpool_2x2(x)
        pools_ = [pool_1, pool_2, pool_3, pool_4]
        pools = [tf.keras.layers.Conv2D(16, (1,1), padding='same')(i) for i in pools_]

        # Part-2 : Attention models
        attentions = [self.attention(pools[i]) for i in range(4)]
        concatenated_attention_1_and_2 = tf.keras.layers.concatenate([self.upsample_2x2(attentions[1]), attentions[0]], axis=3)
        concatenated_attention_3_and_4 = tf.keras.layers.concatenate([self.upsample_2x2(attentions[3]), attentions[2]], axis=3)
        upsampled_concatenated_attention_1_and_2 = self.upsample_2x2(concatenated_attention_1_and_2)
        upsampled_concatenated_attention_3_and_4 = self.upsample_2x2(concatenated_attention_3_and_4)

        # Part-3 : Deconvoloutions
        upsampled_concatenated_attention_1_and_2 = tf.keras.layers.Conv2D(32, (1,1), padding='same')(upsampled_concatenated_attention_1_and_2)
        deconv_1 = self.deconv_1(upsampled_concatenated_attention_1_and_2)
        upsampled_concatenated_attention_3_and_4 = self.upsample_4x4(upsampled_concatenated_attention_3_and_4)
        upsampled_concatenated_attention_3_and_4 = tf.keras.layers.Conv2D(32, (1,1), padding='same')(upsampled_concatenated_attention_3_and_4)
        deconv_2 = self.deconv_2(upsampled_concatenated_attention_3_and_4)
        upsampled_pool4 = self.upsample_16x16(pool_4)
        # upsampled_pool4 = tf.keras.layers.Conv2D(32, (1,1), padding='same')(upsampled_pool4)
        deconv_3 = self.deconv_3(upsampled_pool4)
        concatenated_deconvs = tf.keras.layers.concatenate([deconv_1, deconv_2, deconv_3], axis=3)

        # Part4 : output 
        output = tf.keras.layers.Conv2DTranspose(1 , 1, activation = 'sigmoid')(concatenated_deconvs)
        return output
    
    def model(self):
        x = tf.keras.layers.Input(shape=(80, 160, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class Earlystopper(tf.keras.callbacks.Callback):
    def __init__(self):
        super(Earlystopper, self).__init__()
    
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.95):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(self.accuracy_threshold*100))
            self.model.stop_training = True

