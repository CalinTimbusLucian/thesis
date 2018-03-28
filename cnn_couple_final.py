import numpy as np
import datetime as dt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dense,BatchNormalization,GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.models import model_from_json,clone_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image



def createCNN():
    classifier = Sequential()
    classifier.add(Convolution2D(32,(3, 3),input_shape = (64,64,3), activation = "relu",padding = "same"))
    classifier.add(GaussianNoise(stddev = 0.01))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(64,(3, 3), activation = "relu",padding = "same"))
    classifier.add(GaussianNoise(stddev = 0.01))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(128,(3, 3), activation = "relu",padding = "same"))  
    classifier.add(GaussianNoise(stddev = 0.01))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(256,(3, 3), activation = "relu",padding = "same"))  
    classifier.add(GaussianNoise(stddev = 0.01))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(activation = 'linear', units = 256,
                         kernel_regularizer=regularizers.l2(0.01)))
    classifier.add(LeakyReLU(alpha=.001))
    classifier.add(GaussianNoise(stddev = 0.01))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation = 'linear', units = 256,
                         kernel_regularizer=regularizers.l2(0.01)))
    classifier.add(LeakyReLU(alpha=.001))
    classifier.add(GaussianNoise(stddev = 0.01))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation = 'softmax', units = 42))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
    return classifier

def applyImageAugmentationAndReturnGenerators():
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        samplewise_center=True,
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale =1./255)
    return train_datagen,validation_datagen,test_datagen


def importDataSets():
    train_datagen,validation_datagen,test_datagen = applyImageAugmentationAndReturnGenerators()
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

    validation_set = validation_datagen.flow_from_directory('dataset/validation_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
                    'dataset/test_set',
                    target_size=(64, 64),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=False)

    return training_set, validation_set, test_set




def createBestValidationLossCheckpointer():
    #Best Model GPU e cu LeakyRELU
    #Best Model GPU_RELU e cu RELU
    checkpointer = ModelCheckpoint(filepath='best_model_gpu_relu_batchnorm.hdf5', verbose=1, save_best_only=True)
    return checkpointer

#ASSIGN CLASS WEIGHTS
def retrieveClassWeights():
   dictionary = {}
   dictionary[0] = 2.98
   dictionary[1] = 8.50
   dictionary[2] = 6.32
   dictionary[3] = 11.33
   dictionary[4] = 1.88
   dictionary[5] = 10.07
   dictionary[6] = 4.18
   dictionary[7] = 5.03
   dictionary[8] = 1.04
   dictionary[9] = 4.53
   dictionary[10] = 1.07
   dictionary[11] = 7.15
   dictionary[12] = 5.91
   dictionary[13] = 1.51
   dictionary[14] = 8.50
   dictionary[15] = 1.72
   dictionary[16] = 2.04
   dictionary[17] = 1.12
   dictionary[18] = 3.57
   dictionary[19] = 5.55
   dictionary[20] = 1.53
   dictionary[21] = 10.07
   dictionary[22] = 10.07
   dictionary[23] = 5.55
   dictionary[24] = 1.56
   dictionary[25] = 1.60
   dictionary[26] = 11.33
   dictionary[27] = 1.01
   dictionary[28] = 1
   dictionary[29] = 1.60
   dictionary[30] = 1.13
   dictionary[31] = 1.21
   dictionary[32] = 3.35
   dictionary[33] = 1.08
   dictionary[34] = 1.88
   dictionary[35] = 5.91
   dictionary[36] = 11.33
   dictionary[37] = 5.55
   dictionary[38] = 7.77
   dictionary[39] = 6.32
   dictionary[40] = 2.95
   dictionary[41] = 3.88
   return dictionary
    


def trainCNN(classifier,training_set,validation_set):
        classifier.fit_generator(training_set,
                                 steps_per_epoch=1311,
                                 epochs=45,
                                 validation_data=validation_set,
                                 validation_steps=139,
                                 callbacks = [createBestValidationLossCheckpointer()],
                                 class_weight = retrieveClassWeights()
                                )


#SAVE THE MODEL
def saveModel(classifier):
     classifier_json = classifier.to_json()
     with open("best_model.json", "w") as json_file:
         json_file.write(classifier_json)
     print("Saved model to disk")
     return


#RETRIEVE THE MODEL
def retrieveModel():
    json_file = open('best_model.json', 'r')
    loaded_classifier_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_classifier_json)
    # load weights into new model
    loaded_model.load_weights("best_model.hdf5")
    print("Loaded model from disk")
    return loaded_model

def makeUniquePrediction(training_set,loaded_model,test_image):
        initialTime = dt.datetime.now()
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        finishTime = dt.datetime.now()
        forwardPassTime = finishTime - initialTime
        print(forwardPassTime)
        print('Result is',result)
        return result

#Part 3 - MAKING NEW SINGLE PREDICTIONS
def makeSingleNewPredicitons(training_set,loaded_model):
        import numpy as np
        import datetime as dt
        from keras.preprocessing import image
        initialTime = dt.datetime.now()
        test_image = image.load_img('Resized_Crop0.jpg',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        finishTime = dt.datetime.now()
        forwardPassTime = finishTime - initialTime
        print(forwardPassTime)
        
        test_image = image.load_img('dataset/single_prediction/stop-sign.jpg',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        training_set.class_indices
        
        initialTime = dt.datetime.now()
        test_image = image.load_img('dataset/single_prediction/limita-120.jpg',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        finishTime = dt.datetime.now()
        forwardPassTime = finishTime - initialTime
        print(forwardPassTime)
        
        test_image = image.load_img('dataset/single_prediction/limita-50.jpg',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        initialTime = dt.datetime.now()
        test_image = image.load_img('dataset/single_prediction/giratoriu.png',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        finishTime = dt.datetime.now()
        forwardPassTime = finishTime - initialTime
        print(forwardPassTime)
        
        
        initialTime = dt.datetime.now()
        test_image = image.load_img('dataset/single_prediction/interzis-depasire-masini.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        finishTime = dt.datetime.now()
        forwardPassTime = finishTime - initialTime
        print(forwardPassTime)
        
        test_image = image.load_img('dataset/single_prediction/drum_cu_prioritate.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        test_image = image.load_img('dataset/single_prediction/cedeaza_trecerea.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        test_image = image.load_img('dataset/single_prediction/stop_mic.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        """
        test_image = image.load_img('dataset/single_prediction/sens_giratoiru.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        """
        test_image = image.load_img('dataset/single_prediction/intersectie_cu_drum_secundar.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        
        test_image = image.load_img('dataset/single_prediction/drum_in_lucru.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
      
        test_image = image.load_img('dataset/single_prediction/00137.ppm',target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        print(result)
        training_set.class_indices
      
        return



#classifier = createCNN()
#training_set,validation_set,test_set = importDataSets()
#last_layer.trainable = False
#print(last_layer.trainable)
#classifier.summary()
#trainCNN(classifier,training_set,validation_set)
#saveModel(classifier)
#loaded_model = retrieveModel()
#makeSingleNewPredicitons(training_set,loaded_model)
#probabilities = loaded_model.predict_generator(test_set, 146)
#probabilities.std()




"""



################### TRANSFER LEARNING ####################

#Deep Copy of the existing model
loaded_model = retrieveModel()
classifier_2 = clone_model(loaded_model)
classifier_2.set_weights(loaded_model.get_weights())


#Making the first two layers(+MaxPooling) untrainable
classifier_2.summary()
classifier_2.layers
classifier_2.weights

for layer in classifier_2.layers[:-7]:
    print(layer.get_weights())
    layer.trainable = False
    
#Check CNN again
classifier_2.summary()

#Pop the last layer to introduce another one
classifier_2.summary()
last_layer = classifier_2.layers.pop()
classifier_2.summary()

#Transforming the CNN at output layer to 4 neurons instead of 42
classifier_2.add(Dense(activation = 'softmax', units = 4))
classifier_2.summary()
"""



