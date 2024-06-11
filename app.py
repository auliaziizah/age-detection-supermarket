import numpy as np
from PIL import Image
import image_processing
import os
import io
import math
import argparse
from flask import Flask, render_template, request, make_response, Response, request
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import cv2
import math
import argparse
from flask import jsonify

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)

@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html")

@app.route("/process")
@nocache
def process():
    return render_template('upload.html', file_path="img/image_here.jpg")

@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"])
    else:
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"])
@app.route("/dilation", methods=["POST"])
@nocache
def dilation():
    image_processing.dilation()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/closing", methods=["POST"])
@nocache
def closing():
    image_processing.closing()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/opening", methods=["POST"])
@nocache
def opening():
    image_processing.opening()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/erosion", methods=["POST"])
@nocache
def erosion():
    image_processing.erosion()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/count_objects", methods=["POST"])
@nocache
def count_objects():
    image_processing.count_objects()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    return render_template("uploaded.html", file_path="img/img_now.jpg")
    
@app.route("/binary", methods=["POST"])
def binary():
    image_processing.binary_image()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/find_emoji", methods=["POST"])
def find_emoji():
    # Path to the uploaded image
    img = 'static/img/img_now.jpg'  # Assuming the uploaded image is saved as img_now.jpg in the static/img directory

    # Call the find_emoji function with the path of the uploaded image
    detected_emoji = image_processing.find_emoji(img)

    # Process the result as needed
    return render_template("uploaded.html", file_path="img/img_now.jpg", detected_emoji=detected_emoji)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Grab the frame dimensions and convert it to a blob.
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    # Pass the blob through the network and obtain the detections and predictions.
    net.setInput(blob)
    # net.forward() method detects the faces and stores the data in detections
    detections = net.forward()

    faceBoxes = []

    # This for loop is for drawing rectangle on detected face.
    for i in range(detections.shape[2]):    # Looping over the detections.
        # Extract the confidence (i.e., probability) associated with the prediction.
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:   # Compare it to the confidence threshold.
            # Compute the (x, y)-coordinates of the bounding box for the face.
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            # Drawing the bounding box of the face.
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


# Gives input img to the prg for detection.
# Using argparse library which was imported.
parser = argparse.ArgumentParser()
# If the input argument is not given it will skip this and open webcam for detection
parser.add_argument('--image')

args = parser.parse_args()

'''
Each model comes with two files: weight file and model file
weight file stores the data of the deployment of the model
model file stores actual predication done by the model
We are using pre trained models 

The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
The .caffemodel file which contains the weights for the actual layers
Both files are required when using models trained using Caffe for deep learning.
'''

def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Defining age range.
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # LOAD NETWORK
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open a video file or an image file or a camera stream
    video = cv2.VideoCapture(0)
    padding = 20
    while cv2.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

    # It will detect the no. of faces in the frame
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:   # If no faces are detected
            print("No face detected")   # Then it will print this message

        for faceBox in faceBoxes:
            # print facebox
            face = frame[max(0, faceBox[1]-padding):   # Face info is stored in this variable
                         min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        # The dnn.blobFromImage takes care of pre-processing
        # which includes setting the blob  dimensions and normalization.
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
        # genderNet.forward method will detect the gender of each face detected
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')  # print the gender in the console

            ageNet.setInput(blob)
        # ageNet.forward method will detect the age of the face detected
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')    # print the age in the console

        # Show the output frame
            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow("Detecting age and gender", resultImg)

            if resultImg is None:
                continue

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            #resultImg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


def gen_frames_photo(img_file):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Defining age range.
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # LOAD NETWORK
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open a video file or an image file or a camera stream

    frame = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    #frame = img_file
    #hasFrame, frame = img_file.read()
    #ret, frame = cv2.imencode('.jpg', img_file)
    #video = cv2.VideoCapture(img_file)
    padding = 20
    while cv2.waitKey(1) < 0:
        # Read frame
        #hasFrame, frame = video.read()
        # if not hasFrame:
        # cv2.waitKey()
        # break

        # It will detect the no. of faces in the frame
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:   # If no faces are detected
            print("No face detected")   # Then it will print this message

        for faceBox in faceBoxes:
            # print facebox
            face = frame[max(0, faceBox[1]-padding):   # Face info is stored in this variable
                         min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        # The dnn.blobFromImage takes care of pre-processing
        # which includes setting the blob  dimensions and normalization.
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
        # genderNet.forward method will detect the gender of each face detected
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')  # print the gender in the console

            ageNet.setInput(blob)
        # ageNet.forward method will detect the age of the face detected
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')    # print the age in the console

        # Show the output frame
            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow("Detecting age and gender", resultImg)

            if resultImg is None:
                continue

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            #resultImg = buffer.tobytes()
            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

def perform_age_detection(frame):
    # Di sini Anda akan menggunakan model atau algoritma deteksi usia
    # Untuk keperluan demonstrasi, kita akan mengembalikan usia secara acak
    import random
    return {'age': random.randint(10, 100)}  # Usia secara acak antara 10 dan 100 tahun

# Kemudian, Anda dapat memasukkan fungsi ini ke dalam endpoint '/get_age'
@app.route('/get_age', methods=['POST'])
def get_age():
    # Mendapatkan data gambar dari permintaan POST
    image_data = request.files['image'].read()
    # Konversi data gambar ke format yang dapat diproses oleh OpenCV
    image_np = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Lakukan deteksi usia menggunakan fungsi perform_age_detection
    age_result = perform_age_detection(frame)
    
    return jsonify(age_result)



@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/supermarket')
def supermarket():
    return render_template('supermarket.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_ip = np.asarray(img, dtype="uint8")
        print(img_ip)
        return Response(gen_frames_photo(img_ip), mimetype='multipart/x-mixed-replace; boundary=frame')
        # return 'file uploaded successfully'

if __name__ == '__main__':
    app.debug = True
    app.run()
