import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model = load_model("Custom_CNN_LungCancerDetection.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Define custom class names
custom_class_names = {
    "lung_n": "Normal Lung Image",
    "lung_aca": "Lung Adenocarcinomas condition",
    "lung_scc": "Lung Squamous Cell Carcinomas condition"
}

def main():
    # Set app title
    st.title("Lung Cancer Type Classification")

    # Create a file uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the uploaded image using PIL
        image = Image.open(uploaded_file).convert("RGB")

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize the image to be at least 224x224 and then crop from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # Turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Expand dimensions to match the input shape of the model
        data = np.expand_dims(normalized_image_array, axis=0)

        # Predict using the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip().split()[1]

        # Get the custom class name
        custom_class_name = custom_class_names.get(class_name, class_name)

        # Print prediction and confidence score
        print("Class:", custom_class_name)
        print("Confidence Score:", prediction[0][index])

        # Display the predicted lung cancer type
        st.success(f"Predicted Lung Cancer Type: {custom_class_name}")
        #st.write(f"Probability / Confidence Score: {prediction[0][index] * 100:.2f}%")

if __name__ == "__main__":
    main()
