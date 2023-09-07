
# if run == main();
if __name__ == "__main__":


    from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import legacy


traindata=ImageDataGenerator(rescale=1./255)
testdata=ImageDataGenerator(rescale=1./255)

traingen=traindata.flow_from_directory('data/train',
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')
testgen=testdata.flow_from_directory('data/test',
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')


emotionalmodel=Sequential()

emotionalmodel.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotionalmodel.add(Conv2D(32, kernel_size=(3,3), activation='relu',))
emotionalmodel.add(MaxPooling2D(pool_size=(2,2)))
emotionalmodel.add(Dropout(0.25))

emotionalmodel.add(Conv2D(128,kernel_size=(3,3),activation='relu',))
emotionalmodel.add(MaxPooling2D(pool_size=(2,2)))
emotionalmodel.add(Conv2D(128,kernel_size=(3,3),activation='relu',))
emotionalmodel.add(MaxPooling2D(pool_size=(2,2)))
emotionalmodel.add(Dropout(0.25))

emotionalmodel.add(Flatten())
emotionalmodel.add(Dense(1024,activation='relu'))
emotionalmodel.add(Dropout(0.5))
emotionalmodel.add(Dense(7,activation='softmax'))

emotionalmodel.compile(loss='categorical_crossentropy', optimizer = legacy.Adam(learning_rate=0.001),metrics=['accuracy'] )


#train the neural network

emotionalmodelinfo=emotionalmodel.fit(
    traingen,
    steps_per_epoch=28709 // 64,

    epochs=50,
    validation_data=testgen,
    validation_steps=7178 // 64
)

#save model on jsonfile

modeljson=emotionalmodel.to_json()

with open("emotionalmodeljson","w") as json_file:
    json_file.write(modeljson)

