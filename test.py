import tensorflow as tf
from tensorflow import keras
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Returns a short sequential model


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


class Class:
    def __init__(self):
        x = 0

    def new(self):
        self.model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])

        self.model.fit(train_images, train_labels, epochs=5)

        # Save entire model to a HDF5 file
        self.model.save('mymodel.h5')

    def load(self):
        # Recreate the exact same model, including weights and optimizer.
        self.model = keras.models.load_model('mymodel.h5')
        self.model.summary()

        loss, acc = self.model.evaluate(test_images, test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))
        self.model.fit(train_images, train_labels, epochs=5)

        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)

        # create on Colab directory
        self.model.save('model.h5')
        model_file = drive.CreateFile({'title': 'model.h5'})
        model_file.SetContentFile('model.h5')
        model_file.Upload()

c = Class()
c.new()
c.load()
