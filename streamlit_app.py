from PIL import Image,ImageOps
import keras_preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
# import tensorflow_hub as hub

def main():
    # We can add a picture to the interface by providing the location below
    image = Image.open('./Malaria-mosquito.jpg')
    st.image(image, caption='malaria', use_column_width=True)
    st.title("Malaria Parasite Detector Application")

    # Provide database link for testing with multiple images
    st.write("If you want to test the model with multiple images, you can download the dataset from [this link](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).")

    # File uploader for user-uploaded images
    file_uploaded = st.file_uploader("Upload a blood smear image", type=['jpg', 'png', 'jpeg'])

    # Option to select from preloaded images
    st.write("Or select one of the preloaded images below:")
    preloaded_images = {
        "Infected Cell": './images/infected.png',
        "Uninfected Cell": './images/safe.png',
        "Sample Cell": './images/C100P61ThinF_IMG_20150918_144348_cell_144.png'
    }
    selected_image_label = st.selectbox("Choose a preloaded image", options=["None"] + list(preloaded_images.keys()))

    # Determine which image to use
    current_image = None
    if file_uploaded is not None:
        current_image = Image.open(file_uploaded)
        st.write("**Using uploaded image:**")
    elif selected_image_label != "None":
        current_image = Image.open(preloaded_images[selected_image_label])
        st.write(f"**Using preloaded image: {selected_image_label}**")

    # Display and predict for the current image
    if current_image is not None:
        # Display the image
        figure = plt.figure()
        plt.imshow(current_image)
        plt.axis('off')
        st.pyplot(figure)
        
        # Single predict button for whichever image is selected
        if st.button('Predict'):
            with st.spinner('Making prediction...'):
                result = predict_class(current_image)
                st.write("### Prediction Result:")
                st.write(result)

def predict_class(image):
    try:
        # Load the model
        classifier_model = tf.keras.models.load_model(r'./model_vgg19.h5')
        
        # Preprocess the image
        test_image = image.resize((224, 224))
        test_image = keras_preprocessing.image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make prediction
        class_names = ['infected', 'uninfected']
        predictions = classifier_model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        
        # Get the predicted class (REMOVED the hardcoded line)
        image_class = class_names[np.argmax(scores)]
        confidence = np.max(scores) * 100
        
        # Generate result message based on actual prediction
        if image_class == 'infected':
            result = f'⚠️ **INFECTED**: You are infected with malaria. Please consult your doctor as soon as possible.\n\nConfidence: {confidence:.2f}%'
        else:
            result = f'✅ **UNINFECTED**: Congrats!! You are not infected with malaria.\n\nConfidence: {confidence:.2f}%'
        
        return result
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

if __name__ == '__main__':
    main()
