import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

#load the images
test_images = sorted(os.listdir('img_test'), key=lambda x: int(os.path.splitext(x)[0]))

# train the model if it doesn't exist or if the -t flag is passed
if not os.path.exists('model.h5') or len(sys.argv) > 1 and sys.argv[1] == '-t':
    print('Training the model...')
    #load the dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #normalize the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    #create the model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    #compile the model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    #train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=32)
    model.evaluate(test_images, test_labels)
    #test the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_accuracy}')
    #save the model
    model.save('model.h5')
else:
    pass

#predict all th images in the test folder
for image_name in test_images:
    image_path = os.path.join('img_test', image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (28, 28))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    #load the model
    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    
    recognized_character = int(predicted_class)
    print(f'image: {image_name}, recognized character: {recognized_character}')
    with open(f'output/{image_name}.txt', 'w') as f:
        f.write(str(recognized_character))
    plt.imshow(image, cmap='gray')
    plt.show()


