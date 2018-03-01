from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

#Step 1 - Initialising the CNN
classifier = Sequential()
#Step 2 - Convolution and Pooling Layers
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

#Creating the callbacks
#Save the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath='best_model.hdf5', verbose=1, save_best_only=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

validation_set = validation_datagen.flow_from_directory('dataset/validation_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                    'dataset/test_set',
                    target_size=(64, 64),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=False)

    
classifier.fit_generator(training_set,
                        steps_per_epoch=1311,
                        epochs=25,
                        validation_data=validation_set,
                        validation_steps=139,
                        workers = 4,
                        callbacks = [checkpointer]
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

loaded_model = retrieveModel()
probabilities = loaded_model.predict_generator(test_generator, 4660)
"""
#Get training set class names and indices
def retrieveTrainingSetClassIDS(training_set):
    inv_map = {v: k for k, v in  training_set.class_indices.items()}
    return inv_map

def revertMap(input_map):
    inv_map = {v: k for k, v in input_map.items()}
    return inv_map
"""
#Part 3 - MAKING NEW SINGLE PREDICTIONS
def makeSingleNewPredicitons(loaded_model):
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

"""
def importTestSetGenerationModule():
    import os
    import sys
    sys.path.append(os.path.abspath("F:/PRS CNN/testsetgeneration.py"))
    return
No need to import the modules, files in the same place"""
"""
def getTestSetClassIdsAndApparitions():
    from testsetgeneration import getClassIdsAndApparitions
    return getClassIdsAndApparitions()

def mapOldTrainingSetClassIndicesToStringClasses():
    from testsetgeneration import mapOldTrainingSetClassIndicesToStringClasses
    return mapOldTrainingSetClassIndicesToStringClasses()

def retrieveTestSetImages():
    from testsetgeneration import readTestSetImages
    return readTestSetImages()

def getPredictionArray():
    from keras.preprocessing import image
    from testsetgeneration import testSetClassIndicesToStringClasses
    from testsetgeneration import getClassIds
    from testsetgeneration import trainingSetToTestSetMapping
    import numpy as np
    loaded_model = retrieveModel()
    prediction_array = []
    pathPreppender = {}
    pathPreppender[0] = '0'
    pathPreppender[1] = '00'
    pathPreppender[2] = '000'
    pathPreppender[3] = '0000'
    test_set_class_indices_and_strings = testSetClassIndicesToStringClasses()
    test_class_ids = getClassIds()
    training_set_class_indices = retrieveTrainingSetClassIDS(training_set)
    training_set_to_test_set_mapping = trainingSetToTestSetMapping()
    
    count = 0
    step = 0
    for i in range(0,12629):
       path = 'dataset/test_set/images/'
       if(i<10):
           path = path + pathPreppender[3] + str(i) + '.ppm'
       elif (i>=10 and i<100):
           path = path + pathPreppender[2] + str(i) + '.ppm'
       elif (i>=100 and i<=999):
           path = path + pathPreppender[1] + str(i) + '.ppm'
       elif (i>=1000): path = path + pathPreppender[0] + str(i) + '.ppm'
      
       test_image = image.load_img(path,target_size = (64,64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = loaded_model.predict(test_image)
       predicted_class = result.argmax()
       
  
       true_image_class_id = test_class_ids[i]
       print('True image class id',true_image_class_id)
       print('Predicted Training-Class is',predicted_class)
       print('Predicted Class id is',training_set_to_test_set_mapping[predicted_class])
       step = step + 1
       if(int(true_image_class_id) == int(training_set_to_test_set_mapping.get(predicted_class))):
           count = count + 1
       print('Step is', step)
       print('Correct out of',step,'are',count)
       prediction_array.append(predicted_class)
  
    return prediction_array


        
        

makeSingleNewPredicitons(retrieveModel())
training_set_class_indices = retrieveTrainingSetClassIDS(training_set)
prediction_array = getPredictionArray()
"""
#saveModel(classifier)

