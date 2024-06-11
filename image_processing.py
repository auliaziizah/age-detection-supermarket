import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import savefig
import cv2

def grayscale():
    img = cv2.imread("static/img/img_now.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/img/img_now.jpg", gray_img)

def is_grey_scale(img_path):
    im = Image.open(img_path)
    if im.mode == "L":  # Mode "L" menunjukkan gambar skala abu-abu
        return True
    elif im.mode == "RGB":
        rgb = im.convert("RGB")
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                r, g, b = rgb.getpixel((i, j))
                if r != g != b:
                    return False
        return True
    else:
        return False

def zoomin():
    img = Image.open("static/img/img_now.jpg")
    width, height = img.size
    new_img = img.resize((width * 2, height * 2), Image.BICUBIC)
    new_img.save("static/img/img_now.jpg")

def zoomout():
    img = Image.open("static/img/img_now.jpg")
    width, height = img.size
    new_img = img.resize((width // 2, height // 2), Image.BICUBIC)
    new_img.save("static/img/img_now.jpg")

def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    shifted_img = img_arr[:, 50:]
    new_img = Image.fromarray(shifted_img)
    new_img.save("static/img/img_now.jpg")

def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    shifted_img = img_arr[:, :-50]
    new_img = Image.fromarray(shifted_img)
    new_img.save("static/img/img_now.jpg")

def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    shifted_img = img_arr[50:, :]
    new_img = Image.fromarray(shifted_img)
    new_img.save("static/img/img_now.jpg")

def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    shifted_img = img_arr[:-50, :]
    new_img = Image.fromarray(shifted_img)
    new_img.save("static/img/img_now.jpg")

def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    img_arr = np.clip(img_arr.astype(int) + 100, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")

def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    img_arr = np.clip(img_arr.astype(int) - 100, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")

def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    img_arr = np.clip(img_arr.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")

def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    img_arr = np.clip(img_arr.astype(np.float32) / 1.25, 0, 255).astype(np.uint8)
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    new_img = cv2.filter2D(img, -1, kernel)
    return new_img

def edge_detection():
    img = cv2.imread("static/img/img_now.jpg", 0) 
    edges = cv2.Canny(img, 100, 200)  
    cv2.imwrite("static/img/img_now.jpg", edges)

def blur():
    img = cv2.imread("static/img/img_now.jpg")
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  
    cv2.imwrite("static/img/img_now.jpg", blurred_img)

def sharpening():
    img = cv2.imread("static/img/img_now.jpg")
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)  
    cv2.imwrite("static/img/img_now.jpg", sharpened_img)

def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = cv2.imread(img_path)
    
    if len(img.shape) == 3:  # Gambar berwarna
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.savefig("static/img/rgb_histogram.jpg", dpi=300)
        plt.close()
    else:  # Gambar grayscale
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        plt.savefig("static/img/grey_histogram.jpg", dpi=300)
        plt.close()

def df(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    return hist

def cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return cdf_normalized

def histogram_equalizer():
    img = cv2.imread('static/img/img_now.jpg', 0)  # Baca gambar dalam skala abu-abu
    equalized_img = cv2.equalizeHist(img)  # Terapkan ekualisasi histogram
    cv2.imwrite('static/img/img_now.jpg', equalized_img)

def threshold(lower_thres, upper_thres):
    img = cv2.imread("static/img/img_now.jpg", 0)  # Baca gambar dalam skala abu-abu
    _, threshed_img = cv2.threshold(img, lower_thres, upper_thres, cv2.THRESH_BINARY)
    cv2.imwrite("static/img/img_now.jpg", threshed_img)

def dilation():
    # Baca gambar sebagai array NumPy
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Lakukan thresholding
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Lakukan dilasi
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # Kernel untuk dilasi
    dilated_img = cv2.dilate(img_thresh, kernel, iterations=1)
    
    # Simpan hasil dilasi
    cv2.imwrite("static/img/img_now.jpg", dilated_img)

def closing():
    # Baca gambar sebagai array NumPy
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Lakukan thresholding
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Lakukan closing
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # Kernel untuk dilasi
    closed_img = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    
    # Simpan hasil closing
    cv2.imwrite("static/img/img_now.jpg", closed_img)

def opening():
    # Baca gambar sebagai array NumPy
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Lakukan thresholding
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Lakukan opening
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # Kernel untuk dilasi
    opened_img = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    
    # Simpan hasil opening
    cv2.imwrite("static/img/img_now.jpg", opened_img)

def erosion():
    # Baca gambar sebagai array NumPy
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Lakukan thresholding
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Lakukan erosi
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(img_thresh, kernel, iterations=1)
    
    # Simpan hasil erosi
    cv2.imwrite("static/img/img_now.jpg", eroded_img)

def binary_image():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).copy()  # Salin larik agar dapat ditulis
    # Periksa jumlah channel gambar
    if len(img_arr.shape) == 3:  # Jika gambar berwarna (RGB)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)  # Konversi ke citra grayscale
    # Jika gambar sudah dalam citra grayscale, biarkan saja
    
    # Menentukan threshold secara otomatis menggunakan metode Otsu
    _, thresh_img = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_img = Image.fromarray(thresh_img)
    new_img.save("static/img/img_now.jpg")

def count_objects():
    image = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Terapkan thresholding adaptif
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    
    # Terapkan operasi closing untuk mengisi lubang-lubang kecil dan celah di depan
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    # Temukan kontur
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Hitung jumlah objek hitam (area yang terisi)
    object_count = len(contours)
    
    # Gambar kontur pada gambar asli
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    
    # Tambahkan teks jumlah objek ke citra
    text = f'{object_count} Object'
    
    # Tentukan font, ukuran, dan warna teks
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # Warna hitam dalam format BGR
    
    # Hitung ukuran teks
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    
    # Tentukan posisi x dan y untuk teks
    text_x = (image_with_contours.shape[1] - text_size[0]) // 2
    text_y = image_with_contours.shape[0] - 10  # 10 piksel dari bawah
    
    # Tambahkan teks ke citra
    cv2.putText(image_with_contours, text, (text_x, text_y), font, font_scale, font_color, 1)
    
    # Simpan citra dengan teks tambahan
    cv2.imwrite("static/img/img_now.jpg", image_with_contours)

def freeman_chain_code(contour):
    chain_code = []
    
    directions = [0, 1, 2, 3, 4, 5, 6, 7]

    starting_point = contour[0][0]
    current_point = starting_point

    for point in contour[1:]:
        x, y = point[0]
        dx = x - current_point[0]
        dy = y - current_point[1]
        direction = None

        if dx == 1 and dy == 0:
            direction = 0
        elif dx == 1 and dy == -1:
            direction = 1
        elif dx == 0 and dy == -1:
            direction = 2
        elif dx == -1 and dy == -1:
            direction = 3
        elif dx == -1 and dy == 0:
            direction = 4
        elif dx == -1 and dy == 1:
            direction = 5
        elif dx == 0 and dy == 1:
            direction = 6
        elif dx == 1 and dy == 1:
            direction = 7
        
        if direction is not None:
            chain_code.append(direction)
            current_point = (x, y)

    return chain_code

def thinning(image):
    # Convert the image to grayscale if it's not already in grayscale
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Apply Otsu's thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the binary image
    binary_image = cv2.bitwise_not(binary_image)

    size = np.size(binary_image)
    skeleton = np.zeros(binary_image.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False

    while not done:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()
        zeros = size - cv2.countNonZero(binary_image)
        if zeros == size:
            done = True

    return skeleton

def save_chain_codes_to_file(chain_codes, img_paths, file_name):
    with open(file_name, 'a') as f:
        for i, code in enumerate(chain_codes):
            file_name = img_paths[i].split('/')[-1].split('.')[0] 
            f.write(f"{file_name}={','.join(map(str, code))}\n")

def read_knowledge_base(file_name):
    known_emojis = {}
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('=')
                if len(parts) == 2:
                    emoji_name = parts[0]
                    chain_code = list(map(int, parts[1].split(',')))
                    known_emojis[emoji_name] = chain_code
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    return known_emojis

def match_digit(test_digit, known_emojis):
    for emoji, code in known_emojis.items():
        if code == test_digit:
            return emoji
    return None

def process_image_thinning(img):
    # Convert image to grayscale if it's not already
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Apply thresholding to obtain a binary image
    _, binary_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the binary image
    inverted_img = cv2.bitwise_not(binary_img)

    # Apply morphological operations to clean up noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    cleaned_img = cv2.morphologyEx(inverted_img, cv2.MORPH_CLOSE, kernel)

    # Apply thinning process
    thin_img = thinning(cleaned_img)

    # Identify contours in the thinned image
    contours, _ = cv2.findContours(thin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Process the largest contour
        largest_contour = contours[0]
        if len(largest_contour) >= 3:
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the bounding box region from the thinned image
            cropped_thin_img = thin_img[y:y+h, x:x+w]

            # Compute the skeleton image and chain code for the cropped region
            skeleton_img = thinning(cropped_thin_img)
            chain_code = freeman_chain_code(largest_contour)

            # Return skeleton image and chain code
            return skeleton_img, chain_code
        else:
            print("No clear contour found after thinning.")
            return None, None
    else:
        print("No contour found after thinning.")
        return None, None

def process_images_and_save_chain_codes():
    # List of image paths
    img_paths = ['static/emoji/blush.png', 
                'static/emoji/disappointed_relieved.png', 
                'static/emoji/expressionless.png',
                'static/emoji/face_with_raised_eyebrow.png',
                'static/emoji/face_with_rolling_eyes.png',
                'static/emoji/grin.png',
                'static/emoji/grinning.png',
                'static/emoji/heart_eyes.png',
                'static/emoji/hugging_face.png',
                'static/emoji/hushed.png',
                'static/emoji/joy.png',
                'static/emoji/kissing.png',
                'static/emoji/kissing_closed_eyes.png',
                'static/emoji/kissing_heart.png',
                'static/emoji/kissing_smiling_eyes.png',
                'static/emoji/laughing.png',
                'static/emoji/neutral_face.png',
                'static/emoji/no_mouth.png',
                'static/emoji/open_mouth.png',
                'static/emoji/persevere.png',
                'static/emoji/relaxed.png',
                'static/emoji/rolling_on_the_floor_laughing.png',
                'static/emoji/sleeping.png',
                'static/emoji/sleepy.png',
                'static/emoji/slightly_smiling_face.png',
                'static/emoji/smile.png',
                'static/emoji/smiley.png',
                'static/emoji/smirk.png',
                'static/emoji/star-struck.png',
                'static/emoji/sunglasses.png',
                'static/emoji/sweat_smile.png',
                'static/emoji/thinking_face.png',
                'static/emoji/tired_face.png',
                'static/emoji/wink.png',
                'static/emoji/yum.png',
                'static/emoji/zipper_mouth_face.png',
                ]

    # Inisialisasi list untuk menyimpan Freeman Chain Codes
    chain_codes = []

    # Loop untuk menghitung Freeman Chain Codes untuk setiap gambar
    for img_path in img_paths:
        # Read image and convert it to grayscale directly
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Process image and get skeleton image and chain code
        skeleton_img, current_chain_code = process_image_thinning(img)
        chain_codes.append(current_chain_code)

    # Save the chain codes to file
    save_chain_codes_to_file(chain_codes, img_paths,'knowledge-based.env')

def find_emoji(img):
    # Read knowledge base
    process_images_and_save_chain_codes()

    knowledge_base_file = "knowledge-based.env"
    known_emojis = read_knowledge_base(knowledge_base_file)

    if not known_emojis:
        print("Basis pengetahuan kosong atau tidak dapat dibaca. Keluar.")
        return None
    
    # Read image from file path
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Check if the image is read successfully
    if img is None:
        print("Tidak dapat membaca citra input. Keluar.")
        return None
    
    # Process thinning and get Freeman Chain Code
    img, chain_code = process_image_thinning(img)

    if chain_code is None:
        print("Tidak dapat menghitung Freeman Chain Code untuk citra input. Keluar.")
        return None

    # Match with knowledge base
    matched_emoji = match_digit(chain_code, known_emojis)
    if matched_emoji is not None:
        print("Emoji yang cocok:", matched_emoji)
        return matched_emoji
    else:
        print("Tidak ditemukan emoji yang cocok dalam .env.")
        return None