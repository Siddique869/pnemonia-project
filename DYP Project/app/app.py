import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import mysql.connector

datapath = 'DYP Project/app/sample'
@st.cache_data 
def load_sample_data1():
    img1 = Image.open(datapath + '/data1n.jpeg')
    # img1 = Image.open(datapath + '\\data1n.jpeg')
    return img1

@st.cache_data 
def load_sample_data2():
    img2 = Image.open(datapath + '/data1p.jpeg')
    # img2 = Image.open(datapath + '\\data1p.jpeg')
    return img2

path = "DYP Project/app/model/resnet_model.h5"  
model = load_model(path)

# Load and preprocess an image
def load_and_preprocess_image(img):
    img = img.resize((256, 256))  # Resize to the target size expected by your model
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Ensure the image is in RGB mode
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    img_arr = img_arr / 255.0  # Normalize the image to the range [0, 1]
    return img_arr

# STREAMLIT CODE

top_image = Image.open("DYP Project/app/sidebar1.jpg")

st.sidebar.image(top_image)

st.sidebar.title(" PneumoScan")
st.sidebar.markdown("> An inference API for pre-screening upper-respiratory infectious diseases based on Chest X-ray (CXR) images.")
st.sidebar.header("Choose a page to proceed!")
page = st.sidebar.selectbox("", ["Upload Your Image", "Sample Data"])

home_tab, working_tab, developer_tab,Records_tab = st.tabs(["Home", "Working", "Developer","Records"])

with home_tab:
    st.title('Pnemonia disease symptoms and precautions:')
    st.image('DYP Project/app/pnemonia1.jpg')
    st.write('Pneumonia is an infection that inflames the air sacs in one or both lungs, which can cause symptoms ranging from mild to severe. The severity of pneumonia can depend on factors such as the persons age, overall health, and the type of germ causing the infection. Common symptoms of pneumonia include:')
    st.title('Common Symptoms of Pneumonia:')   
    st.write('1.Cough:')  
    st.write('Often productive, meaning the person coughs up mucus or phlegm, which can be green, yellow, or even blood-tinged.')
    st.write('2.Chest Pain:')
    st.write('Sharp or stabbing pain, often worsens with breathing or coughing.')
    st.write('3.Fever:')
    st.write('A high fever is common, often accompanied by chills and sweating.')
    st.write('4.Shortness of Breath:')
    st.write('Difficulty breathing or feeling like you cant get enough air, especially when walking or doing physical activity.')
    st.write('5.Rapid Breathing:')
    st.write('Increased breathing rate, or feeling like you are "breathing harder" than usual.')
    st.write('6.Sweating:')
    st.write('Heavy sweating and chills may occur, especially with a high fever.')
    st.write('7.Confusion or Mental Changes:')
    st.write('Particularly in older adults, confusion or changes in mental status can be a sign of pneumonia, especially if the infection is severe.')
    st.write('8.Bluish or Gray Skin Tone (Cyanosis):')
    st.write('This can happen if the pneumonia causes a lack of oxygen in the blood, leading to a bluish tint on the lips, face, or extremities.')
    st.header('Preventive Measures:')
    st.write('1.Vaccination:')
    st.write(' Vaccines like the pneumococcal vaccine and flu vaccine can help prevent some types of pneumonia.')
    st.write('2.Hand hygiene:')
    st.write('Regular handwashing helps prevent the spread of respiratory infections.')
    st.write('3.Good hygiene: ')
    st.write('Covering coughs and sneezes can reduce the transmission of pneumonia-causing germs.')
   

with working_tab :
    st.title(' PneumoScan Working!')
    st.image('DYP Project/app/ae-cnn-final.png')
    st.header('Encoder, Decoder, and Autoencoder')
    st.write("Overview of AE-CNN: Our proposed framework consists of three main blocks namely encoder, decoder, and classifier. The figure shows the autoencoder based convolutional neural network (AE-CNN) model for disease classification. Here, autoencoder reduces the spatial dimension of the imput image of size 1024 × 1024. The encoder produces a latent code tensor of size 256 × 256 and decoder reconstructs back the image. This latent code tensor is passed through a CNN classifier for classifying the chest x-rays. The final loss is the weighted sum of the resconstruction loss by decoder and classification loss by the CNN classifier.")
    st.write("Encoder:")
    st.markdown("""> Function: The encoder compresses the input data into a latent-space representation. In the context of an image, it extracts important features and reduces the dimensionality.
                    Architecture: Typically consists of convolutional layers followed by pooling layers. For example, in ResNet-50, the initial layers can be considered part of the encoder as they extract features from the input image.""")
    st.write("")
    st.write("Decoder:")
    st.markdown("""> Function: The decoder reconstructs the input data from the latent-space representation. This is used in tasks where output images are required (e.g., image generation or segmentation).
                    Architecture: Typically consists of upsampling layers (like transposed convolutions) that increase the dimensionality of the latent representation back to the original input size.""")
    st.write("")
    st.write("Autoencoder:")
    st.markdown("""> Function: An autoencoder is a type of neural network used to learn efficient codings of input data. It consists of two parts: the encoder and the decoder. The goal is to compress the input into a latent-space representation and then reconstruct the output as closely as possible to the original input.
                    Architecture: Combines both the encoder and decoder. The encoder reduces the input to a latent space, and the decoder reconstructs the input from this latent space.""")
    
    st.subheader("ResNet-50 Model for Pnemonia Detection Using Chest X-Ray")
    st.write("ResNet-50 (Residual Network):")
    st.markdown("ResNet-50 is a deep convolutional neural network that is 50 layers deep. It is well-known for its ability to handle the vanishing gradient problem, which is common in very deep networks. This is achieved through the introduction of residual blocks.")
    

    # Feature Extraction with ResNet-50 Encoder
    st.subheader("Feature Extraction with ResNet-50 Encoder")
    st.markdown("""
    The **ResNet-50** model's convolutional layers act as an encoder, extracting high-level features from chest X-ray images.
    """)

    # Classification Head
    st.subheader("Classification Head")
    st.markdown("""
    After feature extraction, a global average pooling layer and a fully connected layer are added to classify the image as **Pnemonia** or **negative**.
    """)

    # Training
    st.subheader("Training")
    st.markdown("""
    The model is trained on a labeled dataset of chest X-ray images, with labels indicating the presence or absence of Pnemonia.
    """)

    # Evaluation
    st.subheader("Evaluation")
    st.markdown("""
    The model is evaluated on a separate test set to measure its accuracy, sensitivity, specificity, and other relevant metrics.
    """)

with developer_tab:
  st.title(" PneumoScan: Chest X-ray AI")
        
  st.markdown("> Disclaimer: I do not claim this application as a highly accurate COVID Diagnosis Tool. This application has not been professionally or academically vetted. This is purely for educational purposes to demonstrate the potential of AI's help in medicine.")
  st.markdown("Developed by: [Siddique, Sufiyan, Shubham, Siddshesh, Sidhant]")
  st.markdown("**Note:** You should upload at most one Chest X-ray image of either class (COVID-19 infected or normal). Since this application is a classification task, not a segmentation task.")

  if page == 'Sample Data':
        st.header("Sample Data for Detecting Pnemonia")
        st.markdown("Here you can choose Sample Data")
        sample_option = st.selectbox('Please Select Sample Data', ('Sample Data I', 'Sample Data II'))

        st.write('You selected:', sample_option)
        sample_image = None
        if sample_option == 'Sample Data I':
            if st.checkbox('Show Sample Data I'):
                st.info("Loading Sample data I.......")
                sample_image = load_sample_data1()
                st.image(sample_image, caption='Sample Data I', use_column_width=True)
        else:
            if st.checkbox('Show Sample Data II'):
                st.info("Loading Sample data II..........")
                sample_image = load_sample_data2()
                st.image(sample_image, caption='Sample Data II', use_column_width=True)
        
        st.write("")
        if sample_image is not None:
            sample_img_arr = load_and_preprocess_image(sample_image)
            sample_prediction = model.predict(sample_img_arr)
            
            if st.button("Detect", type="primary"):
                st.write("Classifying...")
                if sample_prediction[0][0] > 0.5:
                    st.success("The Patient has Positive X-Ray:Pnemonia Positive")
                else:
                    st.warning("The Patient has Normal X-Ray: Pnemonia Negative")

  else:
        uploaded_image_file = st.file_uploader("Choose a chest X-ray image...", type="jpg")

        if uploaded_image_file is not None:
            uploaded_image = Image.open(uploaded_image_file)
            st.image(uploaded_image, caption='Uploaded Chest X-ray', use_column_width=True)
            st.write("")

            
            uploaded_img_array = load_and_preprocess_image(uploaded_image)
            uploaded_prediction = model.predict(uploaded_img_array)

            if st.button("Detect", type="primary"):
                st.write("Classifying...")
                if uploaded_prediction[0][0] > 0.5:
                    prediction = "COVID-19 Positive"
                    st.success("The Patient has Positive X-Ray: Pnemonia Positive")
                else:
                    prediction = "Normal"
                    st.warning("The Patient has Normal X-Ray: Pnemonia Negative")
                    
with Records_tab:
 
 # Database configuration
  config = {
    'user': 'if0_38718473',
    'password': 'siddiquesanadi',
    'host': 'sql300.infinityfree.com',
    'database': 'if0_38718473_userdb',
      'port':3306
       }

  def create_connection():
    """Create a connection to the MySQL database."""
    try:
        db = mysql.connector.connect(**config)
        return db
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return None

  def create_patients_table(db):
    """Create the patients table in the database."""
    cursor = db.cursor()
    create_patients_table_query = """
    CREATE TABLE IF NOT EXISTS patients (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        age INT,
        contact_number VARCHAR(20),
        Result VARCHAR(255),
        date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        email VARCHAR(255)
    )
    """
    cursor.execute(create_patients_table_query)
    db.commit()
    cursor.close()
    st.success("Patients table created successfully.")

  def insert_patient_record(db, name, age, contact_number, email, Result):
    """Insert a new patient record into the 'patients' table."""
    cursor = db.cursor()
    insert_patient_query = """
    INSERT INTO patients (name, age, contact_number, email, Result)
    VALUES (%s, %s, %s, %s, %s)
    """
    patient_data = (name, age, contact_number, email, Result)
    cursor.execute(insert_patient_query, patient_data)
    db.commit()
    cursor.close()
    st.success("Patient record inserted successfully.")

  def fetch_all_patients(db):
    """Fetch all records from the 'patients' table."""
    cursor = db.cursor()
    select_patients_query = "SELECT * FROM patients"
    cursor.execute(select_patients_query)
    patients = cursor.fetchall()
    cursor.close()
    return patients

def main():
    db = create_connection()
    if db is None:
        return  # Exit if the connection failed

    create_patients_table(db)  # Ensure the table exists

    st.title("Patient Management System :hospital:")
    menu = ["Add Patient Record", "Show Patient Records"]
    options = st.sidebar.radio("Select an Option :dart:", menu)

    if options == "Add Patient Record":
        st.subheader("Enter patient details :woman_in_motorized_wheelchair:")
        name = st.text_input("Enter name of patient", key="name")
        age = st.number_input("Enter age of patient", key="age", value=1)
        contact = st.text_input("Enter contact of patient", key="contact")
        email = st.text_input("Enter email of patient", key="email")
        Result = st.text_input("Enter Result of patient", key="Result")

        if st.button("Add Patient Record"):
            insert_patient_record(db, name, age, contact, email, Result)

    elif options == "Show Patient Records":
        patients = fetch_all_patients(db)
        if patients:
            st.subheader("All patient records :magic_wand:")
            df = pd.DataFrame(patients, columns=['ID', 'Name', 'Age', 'Contact Number', 'Email', 'Result', 'Date Added'])
            st.dataframe(df)
        else:
            st.write("No patient records found.")

    db.close()  # Close the database connection

if __name__ == "__main__":
    main()
