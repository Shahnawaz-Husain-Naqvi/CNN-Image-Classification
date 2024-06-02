from preprocessing import test_datagen,train_datagen
from Layers import classifier


training_data_path = "C:/deep learning dataset/archive (11)/Fruits Classification/train"
test_data_path = "C:/deep learning dataset/archive (11)/Fruits Classification/test"

training_set = train_datagen.flow_from_directory(training_data_path,
                                                target_size = (64,64),
                                                batch_size = 12,
                                                class_mode = 'categorical')
test_set = train_datagen.flow_from_directory(test_data_path,
                                                target_size = (64,64),
                                                batch_size = 12,
                                                class_mode = 'categorical')
history = classifier.fit(training_set,
                            epochs = 1,
                            verbose = 1,
                            validation_data = test_set,
                            )
classifier.save('model/model_last.h5')