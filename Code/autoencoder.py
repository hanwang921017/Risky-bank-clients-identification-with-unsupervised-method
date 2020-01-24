import os
import pandas as pd
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model,model_from_json


"""
This is a class to implement Autoencoder
Inputs:





"""
class Auto_encoder:
    def __init__(self,csv_name,encoder_num):
        self.dataframe=pd.read_csv(csv_name)
        self.train_set=self.dataframe.values
        self.feature_num=self.train_set.shape[1]
        self.output_num=encoder_num

    def define_model_128(self):
        input_feature=Input(shape=(self.feature_num,))
        encoded=Dense(1024,activation='relu')(input_feature)
        encoded=Dense(512,activation='relu')(encoded)
        encoded=Dense(256,activation='relu')(encoded)
        encoder_output=Dense(self.output_num)(encoded)
        decoded=Dense(256,activation='relu')(encoder_output)
        decoded=Dense(512,activation='relu')(decoded)
        decoded=Dense(1024,activation='relu')(decoded)
        decoded=Dense(self.feature_num,activation='tanh')(decoded)
        autoencoder = Model(input=input_feature, output=decoded)
        encoder=Model(input=input_feature,output=encoder_output)
        return autoencoder,encoder

    def define_model_256(self):
        input_feature=Input(shape=(self.feature_num,))
        encoded=Dense(1024,activation='relu')(input_feature)
        encoded=Dense(512,activation='relu')(encoded)
        encoder_output=Dense(self.output_num)(encoded)
        decoded=Dense(512,activation='relu')(encoder_output)
        decoded=Dense(1024,activation='relu')(decoded)
        decoded=Dense(self.feature_num,activation='tanh')(decoded)
        autoencoder = Model(input=input_feature, output=decoded)
        encoder=Model(input=input_feature,output=encoder_output)
        return autoencoder,encoder

    def train_model(self):
        if self.output_num==128:
            autoencoder,encoder=self.define_model_128()
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(self.train_set, self.train_set, nb_epoch=20, batch_size=256, shuffle=True)
            encoder.save_weights('encoder_128.h5')
            encoder_json = encoder.to_json()
            with open('encoder_128.json', 'w') as json_file:
                json_file.write(encoder_json)
            return encoder

        elif self.output_num==256:
            autoencoder, encoder = self.define_model_256()
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(self.train_set, self.train_set, nb_epoch=20, batch_size=256, shuffle=True)
            encoder.save_weights('encoder_256.h5')
            encoder_json=encoder.to_json()
            with open('encoder_256.json','w') as json_file:
                json_file.write(encoder_json)
            return encoder
        else:
            raise Exception("feature num should be 128 or 256")

    def predict_result(self):
        if self.output_num==128:
            name_h5='encoder_128.h5'
            name_json='encoder_128.json'
            save_name='encoded_results_128.csv'
        elif self.output_num==256:
            name_h5 = 'encoder_256.h5'
            name_json = 'encoder_256.json'
            save_name = 'encoded_results_256.csv'
        else:
            raise Exception("feature num should be 128 or 256")

        if os.path.exists(name_h5) and os.path.exists(name_json):
            json_file = open(name_json, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            encoder = model_from_json(loaded_model_json)
            encoder.load_weights(name_h5)
            encoder.compile(optimizer='adam', loss='mse')
            encoded_results = encoder.predict(self.train_set)


        else:
            encoder=self.train_model()
            encoded_results=encoder.predict(self.train_set)

        df_results = pd.DataFrame(encoded_results)
        df_results.to_csv(save_name)
        return df_results

if __name__=="__main__":
    my_encoder=Auto_encoder('result.csv',256)
    results=my_encoder.predict_result()
    print(results.shape)

