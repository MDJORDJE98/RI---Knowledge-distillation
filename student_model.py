
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import regularizers

def build_student_model(input_shape, num_classes = 10):

  model = Sequential()
  model.add(Conv2D(16,(3,3), activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l2(0.0001), input_shape = input_shape))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.2))


  model.add(Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.3))
  
  model.add(Conv2D(64, (3,3), activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001)))

  model.add(Flatten())
  
  model.add(Dense(num_classes))
  model.summary()
  return model


