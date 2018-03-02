from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint



def createCNN():
    classifier = Sequential()
    classifier.add(Convolution2D(32,(3, 3),input_shape = (64,64,3), activation = "relu",padding = "same"))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(64,(3, 3), activation = "relu",padding = "same"))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(128,(3, 3), activation = "relu",padding = "same"))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(256,(3, 3), activation = "relu",padding = "same"))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(activation = 'relu', units = 128))
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
    checkpointer = ModelCheckpoint(filepath='best_model_2.hdf5', verbose=1, save_best_only=True)
    return checkpointer


def trainCNN(classifier,training_set,validation_set):
        classifier.fit_generator(training_set,
                                 steps_per_epoch=1312,
                                 epochs=25,
                                 validation_data=validation_set,
                                 validation_steps=139,
                                 workers = 4,
                                 callbacks = [createBestValidationLossCheckpointer()]
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

#Part 3 - MAKING NEW SINGLE PREDICTIONS
def makeSingleNewPredicitons(training_set,loaded_model):
        import numpy as np
        import datetime as dt
        from keras.preprocessing import image
        initialTime = dt.datetime.now()
        
       
        test_image = image.load_img('dataset/single_prediction/atentie_pericole.ppm',target_size = (64,64))
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
        training_set.class_indices
        
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



classifier = createCNN()
training_set,validation_set,test_set = importDataSets()
trainCNN(classifier,training_set,validation_set)
loaded_model = retrieveModel()
makeSingleNewPredicitons(training_set,loaded_model)
probabilities = loaded_model.predict_generator(test_set, 4660)




