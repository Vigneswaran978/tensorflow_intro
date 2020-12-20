
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>.998):
      print("\nAccuracy is greater than 99%  so cancelling training!")
      self.model.stop_training = True


mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images/255.0
test_images = test_images.reshape(10000, 28, 28, 1 )
test_images = test_images / 255.0


# YOUR CODE STARTS HERE

# YOUR CODE ENDS HERE
callbacks = myCallback()
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), input_shape=(28, 28, 1), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs = 10, callbacks=[callbacks])


# YOUR CODE ENDS HERE