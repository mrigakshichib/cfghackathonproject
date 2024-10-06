from flask import Flask, request, render_template,redirect, url_for
import cv2
import os
import numpy as np
import torch
from PIL import Image
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv() 



# Configure MySQL connection
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Deandofe#2023',
    database='employment_management'
)

cursor = db.cursor()
app = Flask(__name__)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route("/home")
def hello_world():
    return render_template("index.html")
    # return render_template("supervisor.html")
    # return "<p>Hello, World!</p>"

def detect_and_display_green_areas(image_path):
    # Load the image
    image = cv2.imread(image_path)
    file_name = os.path.basename(image_path)
    
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])  # Lower bound for green
    upper_green = np.array([80, 255, 255])  # Upper bound for green

    # Create a mask for green areas
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Calculate the percentage of green area
    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = cv2.countNonZero(green_mask)
    green_area_percentage = (green_pixels / total_pixels) * 100

    # Create an output image to display the detected areas
    output_image = image_rgb.copy()
    output_image[green_mask > 0] = [0, 255, 0]  # Highlight green areas in green
    output_image_path = os.path.join("static/LatanusResults", f'output_{file_name}.jpg')
    cv2.imwrite(output_image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    return green_area_percentage, output_image_path

@app.route("/loading")
def loading():
    return render_template("loading.html")

@app.route("/latanus", methods=['GET', 'POST'])
def latanus_detection():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get list of files
        print("files")
        result_image_file = ""
        percentage_total = 0
        for file in files:
            if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(file.stream)
                file_path = f'static/LatanusImages/{file.filename}'
                image.save(file_path)
                percentage, result_image = detect_and_display_green_areas(file_path)
                percentage_total = percentage
                # Save the file to a static directory to display later
                # result_file_path = f'static/LatanusResults/{file.filename}'
                # result_image.save(result_file_path)
                result_image_file = result_image
                
        return render_template('latanus.html', percentage = percentage_total, result = result_image_file)
    
    return render_template("latanus.html")


#  Load the YOLOv5 model

def count_animals(image):
    # Define animal classes
    animal_classes = {'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}
    
    # Perform object detection
    results = model(image)
    
    # Parse the results
    detected_objects = results.pandas().xyxy[0]
    animal_count = 0
    
    for _, row in detected_objects.iterrows():
        if row['name'] in animal_classes:
            animal_count += 1
    
    return animal_count

@app.route("/animal_detection", methods=['GET', 'POST'])
def animal_detection():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get list of files
        results = []
        
        for file in files:
            if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(file.stream)  # Open image directly from file stream
                animal_count = count_animals(image)
                # Save the file to a static directory to display later
                file_path = f'static/animalImages/{file.filename}'
                image.save(file_path)
                # filename = file.filename
                final_file_path = os.path.join("static/animalImages", f'{file.filename}')
                results.append({'file_path': final_file_path, 'animal_count': animal_count})
                
        return render_template('animal.html', results=results)

    return render_template('animal.html', results=None)




@app.route('/employee_details')
def employee_details():
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    return render_template('employee.html', employees=employees)

@app.route('/add', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['text']
        position = request.form['position']
        salary = request.form['salary']

        cursor.execute("INSERT INTO employees (first_name, last_name, email, position, salary) VALUES (%s, %s, %s, %s, %s)", 
                       (first_name, last_name, email, position, salary))
        db.commit()
        return redirect(url_for('employee_details'))

    return render_template('add_employee.html')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_employee(id):
    cursor.execute("SELECT * FROM employees WHERE id=%s", (id,))
    employee = cursor.fetchone()

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['text']
        position = request.form['position']
        salary = request.form['salary']

        cursor.execute("UPDATE employees SET first_name=%s, last_name=%s, email=%s, position=%s, salary=%s WHERE id=%s",
                       (first_name, last_name, email, position, salary, id))
        db.commit()
        return redirect(url_for('employee_details'))

    return render_template('edit_employee.html', employee=employee)

@app.route('/delete/<int:id>', methods=['GET'])
def delete_employee(id):
    cursor.execute("DELETE FROM employees WHERE id=%s", (id,))
    db.commit()
    return redirect(url_for('employee_details'))

# if __name__ == "__main__":
#     app.run(debug=True) 
