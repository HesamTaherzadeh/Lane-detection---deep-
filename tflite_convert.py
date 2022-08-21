import tensorflow as tf 
import os



class Converter:
    def __init__(self, model_path):
        self.model_path = model_path
    
    def _get_file_size(self):
        self.size = os.path.getsize(self.model_path)
        return self.size
    
    def _details(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        return input_details, output_details
    
    def __str__(self):
        input_details, output_details = self._details()
        print("Input Shape:", input_details[0]['shape'])
        print("Input Type:", input_details[0]['dtype'])
        print("Output Shape:", output_details[0]['shape'])
        print("Output Type:", output_details[0]['dtype'])
    
    def shape(self):
        self._get_file_size()
        return print(round(self.size / 1024, 3))

    def convert(self, name, optimizer=[tf.lite.Optimize.DEFAULT], type=None):
        self.tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(self.model_path)
        self.tf_lite_converter.optimizations = optimizer
        self.tf_lite_converter.target_spec.supported_types = type
        self.tflite_model = self.tf_lite_converter.convert()
        open(name, "wb").write(self.tflite_model)
        self.interpreter = tf.lite.Interpreter(model_path = self.tflite_model)
    
        

    def reshape(self, shape):
        input_details, output_details = self._details()
        self.interpreter.resize_tensor_input(input_details[0]['index'], shape)
        self.interpreter.resize_tensor_input(output_details[0]['index'], shape)
        self.interpreter.allocate_tensors()
    