import os

# کاهش پیام‌های اضافی TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
CHANNELS = 3

LATENT_DIM = 100
# ============================================================
# ABSOLUTE PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

DISCRIMINATOR_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fake_discriminator.keras"
)

GENERATOR_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fake_generator.keras"
)

print(
    "[DEBUG] Model path:",
    DISCRIMINATOR_MODEL_PATH
)

print(
    "[DEBUG] Model exists:",
    os.path.exists(
        DISCRIMINATOR_MODEL_PATH
    )
)

GENERATOR_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fake_generator.keras"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# GAN GENERATOR
# ============================================================

def build_generator():

    """
    Generator:
    یک بردار تصادفی 100 بعدی دریافت می‌کند
    و یک تصویر مصنوعی چهره 128x128 تولید می‌کند.
    """

    model = Sequential(
        name="GAN_Generator"
    )

    # --------------------------------------------------------
    # Latent Vector
    # --------------------------------------------------------

    model.add(
        Input(
            shape=(LATENT_DIM,)
        )
    )

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    model.add(
        Dense(
            8 * 8 * 256,
            use_bias=False
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    model.add(
        Reshape(
            (8, 8, 256)
        )
    )

    # --------------------------------------------------------
    # 8x8 -> 16x16
    # --------------------------------------------------------

    model.add(
        Conv2DTranspose(
            256,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    # --------------------------------------------------------
    # 16x16 -> 32x32
    # --------------------------------------------------------

    model.add(
        Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    # --------------------------------------------------------
    # 32x32 -> 64x64
    # --------------------------------------------------------

    model.add(
        Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    # --------------------------------------------------------
    # 64x64 -> 128x128
    # --------------------------------------------------------

    model.add(
        Conv2DTranspose(
            CHANNELS,
            kernel_size=4,
            strides=2,
            padding="same",
            activation="tanh"
        )
    )

    return model


# ============================================================
# GAN DISCRIMINATOR
# ============================================================

def build_discriminator():

    """
    Discriminator:
    تصویر 128x128 دریافت می‌کند
    و احتمال Real بودن را بین 0 تا 1 برمی‌گرداند.

    1 = Real
    0 = Fake
    """

    model = Sequential(
        name="GAN_Discriminator"
    )

    model.add(
        Input(
            shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                CHANNELS
            )
        )
    )

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    model.add(
        Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    model.add(
        Dropout(0.30)
    )

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    model.add(
        Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    model.add(
        Dropout(0.30)
    )

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    model.add(
        Conv2D(
            256,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    model.add(
        Dropout(0.35)
    )

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------

    model.add(
        Conv2D(
            512,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        LeakyReLU(
            negative_slope=0.2
        )
    )

    model.add(
        Dropout(0.40)
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    model.add(
        Flatten()
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    return model


# ============================================================
# BUILD GAN
# ============================================================

def build_gan(
    generator,
    discriminator
):

    """
    اتصال Generator به Discriminator
    برای مرحله آموزش GAN
    """

    discriminator.trainable = False

    noise_input = Input(
        shape=(LATENT_DIM,)
    )

    fake_image = generator(
        noise_input
    )

    validity = discriminator(
        fake_image
    )

    gan = Model(
        noise_input,
        validity,
        name="Facial_Fake_Detection_GAN"
    )

    gan.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            beta_1=0.5
        ),
        loss="binary_crossentropy"
    )

    return gan


# ============================================================
# PREPROCESS FACE
# ============================================================

def preprocess_face(
    face_image
):

    """
    تصویر چهره را برای Discriminator آماده می‌کند.
    """

    if face_image is None:

        raise ValueError(
            "Face image is None."
        )

    if face_image.size == 0:

        raise ValueError(
            "Face image is empty."
        )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Normalize -1 to +1
    # --------------------------------------------------------

    image = (
        image - 127.5
    ) / 127.5

    # --------------------------------------------------------
    # Add Batch Dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# FAKE DETECTOR CLASS
# ============================================================

class FakeDetector:

    def __init__(
        self,
        model_path=DISCRIMINATOR_MODEL_PATH,
        real_threshold=0.65
    ):

        self.model_path = model_path

        self.real_threshold = real_threshold

        self.model = None

        self.model_loaded = False

        self.load_model()


    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    def load_model(self):

        if not os.path.exists(
            self.model_path
        ):

            print(
                "[FAKE DETECTOR] "
                "Trained discriminator not found."
            )

            print(
                "[FAKE DETECTOR] "
                "Fake detection is disabled "
                "until training is completed."
            )

            self.model = None

            self.model_loaded = False

            return


        try:

            self.model = tf.keras.models.load_model(
    self.model_path,
    compile=False
)

            self.model_loaded = True

            print(
                "[FAKE DETECTOR] "
                "Trained model loaded successfully."
            )

        except Exception as error:

            print(
                "[FAKE DETECTOR ERROR]",
                error
            )

            self.model = None

            self.model_loaded = False


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        face_image
    ):

        """
        خروجی:

        {
            status: REAL / FAKE / MODEL_NOT_TRAINED
            real_probability: ...
            fake_probability: ...
        }
        """

        if not self.model_loaded:

            return {

                "status":
                    "MODEL_NOT_TRAINED",

                "real_probability":
                    None,

                "fake_probability":
                    None
            }


        processed_face = preprocess_face(
            face_image
        )


        prediction = self.model.predict(
            processed_face,
            verbose=0
        )


        real_probability = float(
            prediction[0][0]
        )

        fake_probability = (
            1.0
            - real_probability
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if (
            real_probability
            >= self.real_threshold
        ):

            status = "REAL"

        else:

            status = "FAKE"


        return {

            "status":
                status,

            "real_probability":
                real_probability,

            "fake_probability":
                fake_probability
        }


# ============================================================
# CREATE MODELS FOR TEST
# ============================================================

def show_model_architecture():

    print("")
    print("=" * 70)

    print(
        "BUILDING GAN ARCHITECTURE"
    )

    print("=" * 70)


    generator = build_generator()

    discriminator = build_discriminator()


    # --------------------------------------------------------
    # Compile Discriminator
    # --------------------------------------------------------

    discriminator.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            beta_1=0.5
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    print("")
    print("=" * 70)

    print(
        "GENERATOR"
    )

    print("=" * 70)

    generator.summary()


    print("")
    print("=" * 70)

    print(
        "DISCRIMINATOR"
    )

    print("=" * 70)

    discriminator.summary()


    print("")
    print(
        "[SUCCESS] GAN architecture created successfully."
    )

    print(
        "[IMPORTANT] Model is NOT trained yet."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    show_model_architecture()