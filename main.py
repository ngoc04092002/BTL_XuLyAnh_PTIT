import cv2
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from flask_cors import CORS
import io
from collections import defaultdict
import heapq
import numpy as np
from scipy.ndimage.filters import convolve

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r'/*': {'origins': '*'}})
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/', methods=['GET'])
def home():
    return render_template('btl_xu_ly_anh.html')

@app.route('/negative-transform', methods=['POST'])
def negative():
    if 'images' not in request.files:
        return 'Không có ảnh tải lên'

    image = request.files['images']
    img = Image.open(image)
    # img = img.convert('L') # L: đại diện màu đen trắn

    # xử lý ảnh thành ảnh âm bản
    img_array = np.array(img)
    #lấy giá trị ngược của mỗi pixel
    negative_img_array = 255 - img_array # s=1-r

    #mảng negative_img_array được chuyển đổi thành đối tượng ảnh (negative_img)
    negative_img = Image.fromarray(negative_img_array)

    # Chuyển ảnh âm bản thành dữ liệu byte để trả về
    img_byte_arr = io.BytesIO()
    negative_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@app.route('/thresholding-image', methods=['POST'])
def thresholding():
    if 'images' not in request.files:
        return 'Không có ảnh tải lên'
    # Đọc ảnh
    image = request.files['images']
    img = Image.open(image)

    # Chuyển đổi ảnh thành ảnh đen trắng (grayscale)
    # img = img.convert('L')

    # Đọc ngưỡng
    # threshold = float(request.form.get('threshold'))

    # Áp dụng biến đổi phân ngưỡng
    thresholded_image = img.point(lambda p: p > 100 and 255)

    # Lưu ảnh đã xử lý
    img_byte_arr = io.BytesIO()
    thresholded_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@app.route('/logarit-transform', methods=['POST'])
def logarit():
    if 'images' not in request.files:
        return 'Không có ảnh tải lên'

    # Đọc ảnh
    image = request.files['images']
    img = Image.open(image)

    # Chuyển đổi ảnh thành ảnh đen trắng (grayscale)
    # img = img.convert('L')

    # Chuyển đổi ảnh thành mảng numpy
    image_array = np.array(img)

    # Đọc hệ số biến đổi
    # c = int(request.form.get('logarithmic'))
    c=1

    # Thực hiện biến đổi logarithmic
    transformed_array = c * np.log10(1 + image_array)
    transformed_array = (transformed_array - np.min(transformed_array)) / (np.max(transformed_array) - np.min(transformed_array)) * 255

    # Chuyển đổi mảng thành đối tượng ảnh
    transformed_image = Image.fromarray(transformed_array.astype(np.uint8))

    # Lưu ảnh đã xử lý
    img_byte_arr = io.BytesIO()
    transformed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@app.route('/power-law-transform', methods=['POST'])
def power_law():
    if 'images' not in request.files:
        return 'Không có ảnh tải lên'

    # Đọc ảnh
    image = request.files['images']
    img = Image.open(image)

    # Chuyển đổi ảnh thành ảnh đen trắng (grayscale)
    # img = img.convert('L')

    # Chuyển đổi ảnh thành mảng numpy
    image_array = np.array(img)

    # Đọc hệ số gamma
    # gamma = float(request.form.get('gamma'))
    gamma=0.6
    # Thực hiện biến đổi exponential
    transformed_array = 255 * (image_array / 255) ** gamma

    # Chuyển đổi mảng thành đối tượng ảnh
    transformed_image = Image.fromarray(transformed_array.astype(np.uint8))

    # Lưu ảnh đã xử lý
    img_byte_arr = io.BytesIO()
    transformed_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@app.route('/histogram-equalizing', methods=['POST'])
def histogram_equalizing():
    image_file = request.files['images']
    img_array = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_to_yuv = cv2.cvtColor(img_array,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

    pil_image = Image.fromarray(hist_equalization_result)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/weighted-averaging', methods=['POST'])
def weight():
    image_file = request.files['images']
    img_array = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    kernel = (
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    )  
    processedImage = cv2.filter2D(img_array, -1, kernel)
    pil_image = Image.fromarray(processedImage)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/median-filter', methods=['POST'])
def median():
    image_file = request.files['images']
    # Đọc ảnh
    img = Image.open(image_file).convert("L")

    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(img)

    filter_size=3
    # Kích thước của bộ lọc
    filter_half_size = filter_size // 2

    # Sao chép mảng ảnh để lưu kết quả
    filtered_array = np.copy(img_array)

    # Xử lý từng pixel trong ảnh
    for i in range(filter_half_size, img_array.shape[0] - filter_half_size):
        for j in range(filter_half_size, img_array.shape[1] - filter_half_size):
            # Lấy ma trận con (bộ lọc) quanh pixel hiện tại
            filter_matrix = img_array[i - filter_half_size : i + filter_half_size + 1, j - filter_half_size : j + filter_half_size + 1]

            # Tính giá trị trung vị của ma trận con
            median_value = np.median(filter_matrix)

            # Gán giá trị trung vị cho pixel hiện tại
            filtered_array[i, j] = median_value

    # Chuyển đổi mảng thành đối tượng ảnh
    filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
    img_buffer = io.BytesIO()
    filtered_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/roberts-operator', methods=['POST'])
def roberts():
    image_file = request.files['images']
    # Đọc ảnh
    img = Image.open(image_file)

    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(img)

    # Tạo ma trận kết quả với cùng kích thước như ảnh gốc
    result_array = np.zeros(img_array.shape, dtype=np.uint8)

    # Xử lý từng pixel trong ảnh (trừ viền)
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            # Tính giá trị gradient theo phép toán Roberts
            gx = img_array[i, j] - img_array[i+1, j+1]
            gy = img_array[i+1, j] - img_array[i, j+1]
            gradient_magnitude = np.sqrt(gx**2 + gy**2)

            # Gán giá trị gradient vào ma trận kết quả
            result_array[i, j] = gradient_magnitude

    # Chuyển đổi mảng thành đối tượng ảnh
    result_image = Image.fromarray(result_array.astype(np.uint8))
    img_buffer = io.BytesIO()
    result_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/sobel-operator', methods=['POST'])
def sobels():
    image_file = request.files['images']
    # Đọc ảnh
    img = Image.open(image_file)

    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(img)

    # Tạo ma trận kết quả với cùng kích thước như ảnh gốc
    result_array = np.zeros(img_array.shape, dtype=np.uint8)

    # Sobel Operator kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Xử lý từng pixel trong ảnh (trừ viền)
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            # Tính giá trị gradient theo phép toán Sobel
            gx = np.sum(img_array[i-1:i+2, j-1:j+2] * sobel_x)
            gy = np.sum(img_array[i-1:i+2, j-1:j+2] * sobel_y)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)

            # Gán giá trị gradient vào ma trận kết quả
            result_array[i, j] = gradient_magnitude

    # Chuyển đổi mảng thành đối tượng ảnh
    result_image = Image.fromarray(result_array.astype(np.uint8))
    img_buffer = io.BytesIO()
    result_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

    
@app.route('/prewitt-operator', methods=['POST'])
def prewitt():
    image_file = request.files['images']
    # Đọc ảnh
    img = Image.open(image_file).convert("L")
    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(img, dtype=np.float32)
    print(img_array.shape)
    height, width = img_array.shape
    result = np.zeros((height - 2, width - 2))

    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])

    for i in range(height - 2):
        for j in range(width - 2):
            region = img_array[i:i + 3, j:j + 3]
            prewitt_x = np.sum(region * kernel_x)
            prewitt_y = np.sum(region * kernel_y)
            result[i, j] = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

    # Chuyển đổi mảng thành đối tượng ảnh
    result_image = Image.fromarray(result.astype(np.uint8))
    img_buffer = io.BytesIO()
    result_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/laplacian-operator', methods=['POST'])
def laplace():
    image_file = request.files['images']
    # Đọc ảnh
    img = Image.open(image_file).convert("L")

    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(img)

    # Tạo ma trận kết quả với cùng kích thước như ảnh gốc
    result_array = np.zeros(img_array.shape, dtype=np.uint8)

    # Laplacian Operator kernel
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Xử lý từng pixel trong ảnh (trừ viền)
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            # Tính giá trị gradient theo phép toán Laplacian
            laplacian_value = np.sum(img_array[i-1:i+2, j-1:j+2] * laplacian_kernel)

            # Gán giá trị gradient vào ma trận kết quả
            result_array[i, j] = abs(laplacian_value)

    # Chuyển đổi mảng thành đối tượng ảnh
    result_image = Image.fromarray(result_array.astype(np.uint8))
    img_buffer = io.BytesIO()
    result_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/canny-operator', methods=['POST'])
def canny():
    image_file = request.files['images']
    # Đọc ảnh và chuyển đổi thành ảnh xám
    img = Image.open(image_file).convert("L")
    img_array = np.array(img)

    # Áp dụng bộ lọc Gaussian để làm mờ ảnh
    blurred_img = convolve(img_array, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16)

    # Tính đạo hàm theo chiều ngang và chiều dọc
    sobel_x = convolve(blurred_img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    sobel_y = convolve(blurred_img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    # Tính độ lớn gradient và hướng gradient
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
    low_threshold = 0.1
    high_threshold = 0.3

    # Áp dụng non-maximum suppression để tìm biên cạnh chính
    suppressed = np.zeros(img_array.shape, dtype=np.uint8)
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            angle = gradient_direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
                if gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1] and gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1]:
                    suppressed[i, j] = gradient_magnitude[i, j]
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                if gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j + 1] and gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j - 1]:
                    suppressed[i, j] = gradient_magnitude[i, j]
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                if gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j] and gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]:
                    suppressed[i, j] = gradient_magnitude[i, j]
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                if gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j - 1] and gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j + 1]:
                    suppressed[i, j] = gradient_magnitude[i, j]

    # Áp dụng ngưỡng hysteresis để tìm biên cạnh cuối cùng
    edges = np.zeros(img_array.shape, dtype=np.uint8)
    high_threshold_value = gradient_magnitude.max() * high_threshold
    low_threshold_value = high_threshold_value * low_threshold
    strong_i, strong_j = np.where(suppressed >= high_threshold_value)
    weak_i, weak_j = np.where((suppressed >= low_threshold_value) & (suppressed < high_threshold_value))
    edges[strong_i, strong_j] = 255

    # Kiểm tra các pixel lân cận của weak edges và xem chúng có nối với strong edges không
    for i, j in zip(weak_i, weak_j):
        if np.any(edges[max(0, i - 1):min(img_array.shape[0], i + 2), max(0, j - 1):min(img_array.shape[1], j + 2)] == 255):
            edges[i, j] = 255

    # Chuyển đổi mảng thành đối tượng ảnh
    result_image = Image.fromarray(edges)
    img_buffer = io.BytesIO()
    result_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/otsu-algorithm', methods=['POST'])
def otsu():
    image_file = request.files['images']
    # Đọc ảnh và chuyển đổi thành ảnh xám
    img = Image.open(image_file).convert("L")

    # Chuyển đổi ảnh thành mảng numpy
    img_array = np.array(img)

    # Tính histogram của ảnh
    histogram, _ = np.histogram(img_array, bins=256, range=(0, 256))

    # Tổng số điểm ảnh trong ảnh
    total_pixels = img_array.shape[0] * img_array.shape[1]

    # Tính tổng giá trị pixel từ histogram
    sum_pixels = np.sum(np.arange(256) * histogram)

    # Tạo mảng để lưu giá trị giữa trong quá trình tính Otsu
    between_variances = np.zeros(256)

    # Tạo mảng để lưu giá trị ngưỡng
    thresholds = np.arange(256)

    # Tìm ngưỡng tối ưu bằng cách tính giá trị phân tách giữa các lớp
    for t in thresholds:
        # Tính số điểm ảnh và histogram của lớp dưới (background)
        w0 = np.sum(histogram[:t])
        hist0 = histogram[:t]

        # Tính số điểm ảnh và histogram của lớp trên (foreground)
        w1 = total_pixels - w0
        hist1 = histogram[t:]

        # Tính trung bình và phương sai của lớp dưới
        if w0 > 0:
            mean0 = np.sum(np.arange(t) * hist0) / w0
            variance0 = np.sum(((np.arange(t) - mean0) ** 2) * hist0) / w0
        else:
            mean0 = 0
            variance0 = 0

        # Tính trung bình và phương sai của lớp trên
        if w1 > 0:
            mean1 = (sum_pixels - np.sum(np.arange(t) * hist0)) / w1
            variance1 = (np.sum(((np.arange(t) - mean1) ** 2) * hist1)) / w1
        else:
            mean1 = 0
            variance1 = 0

        # Tính tổng giữa lớp dưới và lớp trên
        between_variances[t] = w0 * w1 * (mean0 - mean1) ** 2

    # Tìm giá trị ngưỡng tối ưu dựa trên giá trị phân tách giữa các lớp
    optimal_threshold = np.argmax(between_variances)

    # Áp dụng ngưỡng tối ưu để nhị phân hóa ảnh
    binary_image = img_array > optimal_threshold
    binary_image = binary_image.astype(np.uint8) * 255

    # Chuyển đổi mảng thành đối tượng ảnh
    result_image = Image.fromarray(binary_image)
    img_buffer = io.BytesIO()
    result_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/run-length-coding', methods=['POST'])
def rlc():
    image_file = request.files['images']

     # Đọc ảnh và chuyển đổi thành ảnh xám
    img = Image.open(image_file).convert("L")

    pixels = list(img.getdata())
    width, height = img.size
    encoded_data = []

    for y in range(height):
        current_run = 1
        current_pixel = pixels[y * width]

        for x in range(1, width):
            pixel = pixels[y * width + x]
            if pixel == current_pixel:
                current_run += 1
            else:
                encoded_data.append((current_pixel, current_run))
                current_pixel = pixel
                current_run = 1

        encoded_data.append((current_pixel, current_run))

    compressed_image = Image.new("L", (width, height))
    flat_encoded_data = [item for sublist in encoded_data for item in sublist]
    compressed_image.putdata(flat_encoded_data)
    img_buffer = io.BytesIO()
    compressed_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/lzw-coding', methods=['POST'])
def lzw():
    image_file = request.files['images']

    originalImage = Image.open(image_file).convert("L")
    pixels = list(originalImage.getdata())
    dictionary = {i: chr(i) for i in range(256)}
    compressed_data = []
    current_code = 256
    sequence = pixels[0]

    for pixel in pixels[1:]:
        combined_sequence = sequence + pixel
        if combined_sequence in dictionary:
            sequence = combined_sequence
        else:
            compressed_data.append(dictionary[sequence])
            dictionary[combined_sequence] = current_code
            current_code += 1
            sequence = pixel

    compressed_data.append(dictionary[sequence])

    compressed_image = Image.new("L", originalImage.size)
    flat_compressed_data = []
    for code in compressed_data:
        flat_compressed_data.extend([ord(char) for char in code])

    # Đặt giá trị pixel của ảnh đã nén
    compressed_image.putdata(flat_compressed_data)
    img_buffer = io.BytesIO()
    compressed_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer


@app.route('/huffman-encoding', methods=['POST'])
def huffman(): 
    class HuffmanNode:
        def __init__(self, symbol, freq):
            self.symbol = symbol
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq


    def build_frequency_table(data):
        frequency = defaultdict(int)
        for symbol in data:
            frequency[symbol] += 1
        return frequency


    def build_huffman_tree(freq_table):
        heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merge = HuffmanNode(None, left.freq + right.freq)
            merge.left = left
            merge.right = right
            heapq.heappush(heap, merge)
        return heap[0]


    def build_codewords_table(root):
        codewords = {}

        def generate_codes(current_node, code):
            if current_node.symbol is not None:
                codewords[current_node.symbol] = code
                return
            generate_codes(current_node.left, code + '0')
            generate_codes(current_node.right, code + '1')

        generate_codes(root, '')
        return codewords


    def huffman_encoding(data):
        freq_table = build_frequency_table(data)
        huffman_tree = build_huffman_tree(freq_table)
        codewords_table = build_codewords_table(huffman_tree)
        encoded_data = ''.join(codewords_table[symbol] for symbol in data)
        return encoded_data, huffman_tree


    def huffman_decoding(encoded_data, tree):
        decoded_data = []
        current_node = tree
        for bit in encoded_data:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            if current_node.symbol is not None:
                decoded_data.append(current_node.symbol)
                current_node = tree
        return bytes(decoded_data)
    
    image_file = request.files['images']

    originalImage = Image.open(image_file)
    image_data = originalImage.tobytes()
    encoded_data, tree = huffman_encoding(image_data)
    decoded_data = huffman_decoding(encoded_data, tree)
    decoded_image = Image.frombytes('RGB', originalImage.size, decoded_data)
    img_buffer = io.BytesIO()
    decoded_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer
    

@app.route('/erosion', methods=['POST'])
def erosion():
    image_file = request.files['images']

    originalImage = np.array(Image.open(image_file))
    kernel = np.ones((3, 3), np.uint8)
    processedImage = cv2.erode(originalImage, kernel, iterations=1)

    pil_image = Image.fromarray(processedImage)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/dilation', methods=['POST'])
def dilate():
    image_file = request.files['images']
    originalImage = np.array(Image.open(image_file))
    processedImage = cv2.convertScaleAbs(originalImage, 1.1, 5)

    pil_image = Image.fromarray(processedImage)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/opening', methods=['POST'])
def open():
    image_file = request.files['images']
    originalImage = np.array(Image.open(image_file))
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(originalImage, kernel, iterations=1)
    processedImage = cv2.erode(img, kernel, iterations=1)

    pil_image = Image.fromarray(processedImage)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="PNG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

@app.route('/closing', methods=['POST'])
def close():
    image_file = request.files['images']
    originalImage = np.array(Image.open(image_file))
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(originalImage, kernel, iterations=1)
    processedImage = cv2.convertScaleAbs(img, 1.1, 5)

    pil_image = Image.fromarray(processedImage)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="PNG")
    img_buffer = img_buffer.getvalue()
    return img_buffer

if __name__ == '__main__':
    socketio.run(app)
