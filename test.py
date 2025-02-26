import fitz
import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.models import load_model
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2 #opencv
import os # folder directory navigation'
import re


doc = fitz.open('/Users/nguyen/Desktop/ssrn-4501707.pdf')
for i in range(20):
  page = doc.load_page(i)
  pix = page.get_pixmap()
  pix.save(f"/Users/nguyen/Downloads/test_NCKH/correct/{page.number}.png")


# Load mô hình đã được huấn luyện
model_path = "/Users/nguyen/Downloads/imageclassifier.h5"
model = load_model(model_path)

# Định nghĩa đường dẫn thư mục
correct_dir = "/Users/nguyen/Downloads/test_NCKH/correct/"
uncorrect_dir = "/Users/nguyen/Downloads/test_NCKH/uncorrect/"

# Đảm bảo thư mục đích tồn tại
os.makedirs(uncorrect_dir, exist_ok=True)

# Lấy danh sách tất cả ảnh trong thư mục correct
image_files = [f for f in os.listdir(correct_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
max=-1
min=100
# Duyệt qua từng ảnh
for img_name in image_files:
    img_path = os.path.join(correct_dir, img_name)

    # Đọc ảnh bằng OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc {img_name}")
        continue

    # Chuyển đổi ảnh sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Chuyển ảnh thành tensor và resize
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, (256, 256))

    # Chuẩn hóa ảnh về khoảng [0,1] và thêm batch dimension
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    # Dự đoán với model
    yhat = model.predict(img_input)

    # Kiểm tra kết quả dự đoán và di chuyển file nếu cần
    if yhat[0][0] > 0.5:
        new_path = os.path.join(uncorrect_dir, img_name)
        shutil.move(img_path, new_path)
        print(f"Đã di chuyển {img_name} đến thư mục 'uncorrect'")
    else:
        num = int(img_name.split('.')[0])
        if max < num:
          max = num
        if min > num:
          min = num
        print(f"Image {img_name}: Predicted class is Correct (Giữ nguyên trong thư mục 'correct')")
print(max)
print(min)

# Khởi tạo model OCR
ocr_model = PaddleOCR(lang='en',use_angle_cls=True)
# Lưu kết quả OCR vào danh sách
ocr_results = []

output_file = "/Users/nguyen/Downloads/test_NCKH/output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(min, max + 1):
        img_path = os.path.join('.', f'/Users/nguyen/Downloads/test_NCKH/correct/{i}.png')
        
        # Kiểm tra xem ảnh có tồn tại không
        if os.path.exists(img_path):
            result = ocr_model.ocr(img_path)
            
            # Trích xuất và định dạng văn bản OCR
            extracted_text = '\n'.join([line[1][0] for line in result[0]])
            # Ghi vào file
            f.write(extracted_text + "\n")


def merge_numeric_lines(text):
    lines = text.split('\n')
    merged_lines = []
    pattern_number = re.compile(r'\d+$')  # Chỉ chứa số
    pattern_section = re.compile(r'\d+\.\d+')  # Định dạng số như 1.1, 2.1
    pattern_subsection = re.compile(r'\d+\.\d+\.\d+')  # Định dạng số như 1.2.1, 2.3.1
    pattern_chapter = re.compile(r'Chapter \d+$')  # Định dạng "Chapter X"
    pattern_CHAPTER = re.compile(r'CHAPTER \d+$')  # Định dạng "Chapter X"
    remove_keywords = {"CONTENTS", "Contents", "Table of Contents", "Preface", "preface", "PREFACE", "page ", "Page",
                        "Online Resources", "ONLINE RESOURCES", "About the Author", "ABOUT THE AUTHOR"}

    # Xóa các dòng có từ khóa đặc biệt trong 8 dòng đầu tiên
    filtered_lines = [line for line in lines if not any(keyword in line for keyword in remove_keywords)]

    for i, line in enumerate(filtered_lines):
        if pattern_number.fullmatch(line.strip()):
            if merged_lines:
                merged_lines[-1] += ' ' + line.strip()
        elif pattern_section.fullmatch(line.strip()) or pattern_subsection.fullmatch(line.strip()):
            merged_lines.append(line)
        elif i > 0 and (pattern_section.fullmatch(filtered_lines[i - 1].strip()) or pattern_subsection.fullmatch(filtered_lines[i - 1].strip())):
            merged_lines[-1] += ' ' + line.strip()
        else:
            merged_lines.append(line)

        # Xóa số thứ hai nếu có hai số cuối dòng
    refined_lines = [re.sub(r'(\d+) (\d+)$', r'\1', line) for line in merged_lines]

    # Nếu dòng có dạng "Chapter X", thụt dòng tiếp theo
    final_lines = []
    i = 0
    while i < len(refined_lines):
        if (pattern_chapter.fullmatch(refined_lines[i].strip()) and i + 1 < len(refined_lines)) or (pattern_CHAPTER.fullmatch(refined_lines[i].strip()) and i + 1 < len(refined_lines)):
            final_lines.append(refined_lines[i] + ' ' + refined_lines[i + 1])
            i += 2  # Bỏ qua dòng tiếp theo vì đã được gộp
        else:
            final_lines.append(refined_lines[i])
            i += 1
    # Xóa dấu chấm ở cuối dòng
    final_lines = [re.sub(r'\.$', '', line) for line in final_lines]

    # Trích xuất số cuối mỗi dòng và tính toán giá trị mới
    extracted_numbers = []
    last_number = None
    check = None
    for line in final_lines:
        match = re.search(r' (\d+)$', line)
        if match:
            current_number = int(match.group(1))
            if check is None:
                check = current_number  # Lưu số đầu tiên phát hiện được
            if (last_number is not None) and (current_number - last_number) >=0 :
                extracted_numbers.append((current_number - last_number))
            last_number = current_number

    return '\n'.join(final_lines),extracted_numbers , check

with open(output_file, "r", encoding="utf-8") as f:
   text = f.read()

result, extracted_numbers,check = merge_numeric_lines(text)

with open("/Users/nguyen/Downloads/test_NCKH/mucluc.txt", "w", encoding="utf-8") as f:
   f.write(result)

text1 = """- Số buổi học: 48 buổi

- Thời lượng mỗi buổi: 50 phút

- Nội dung của sách bao gồm các phần sau (mục lục):
[Mục lục ở đây]

- Chuẩn đầu ra học phần như sau:
**CLO 1:** Hiểu được kiến trúc máy tính cơ bản bao gồm các thành phần chính như CPU, bộ nhớ, và hệ thống ngoại vi.

* Trình độ nhận biết: Khách hàng có thể mô tả các thành phần chính của kiến trúc máy tính.
* Trình độ hiểu biết: Khách hàng có thể giải thích vai trò của từng thành phần trong kiến trúc máy tính.
* Trình độ vận dụng: Khách hàng có thể thiết kế một hệ thống máy tính đơn giản dựa trên kiến thức về kiến trúc máy tính.

**CLO 2:** Hiểu được cơ chế thực hiện lệnh và các thuật toán quản lý bộ nhớ.

* Trình độ nhận biết: Khách hàng có thể mô tả các cơ chế thực hiện lệnh như pipelining, branch prediction.
* Trình độ hiểu biết: Khách hàng có thể giải thích cách thức hoạt động của các cơ chế thực hiện lệnh và các thuật toán quản lý bộ nhớ.
* Trình độ vận dụng: Khách hàng có thể tối ưu hóa một chương trình máy tính bằng cách sử dụng kiến thức về cơ chế thực hiện lệnh và các thuật toán quản lý bộ nhớ.

**CLO 3:** Hiểu được kiến trúc bộ xử lý (CPU) và các thành phần chính của nó.

* Trình độ nhận biết: Khách hàng có thể mô tả các thành phần chính của CPU như ALU, CU, Register.
* Trình độ hiểu biết: Khách hàng có thể giải thích vai trò của từng thành phần trong CPU.
* Trình độ vận dụng: Khách hàng có thể thiết kế một đơn vị xử lý cơ bản dựa trên kiến thức về kiến trúc bộ xử lý.

**CLO 4:** Hiểu được các nguyên tắc thiết kế và tối ưu hóa hệ thống máy tính.

* Trình độ nhận biết: Khách hàng có thể mô tả các nguyên tắc thiết kế và tối ưu hóa hệ thống máy tính.
* Trình độ hiểu biết: Khách hàng có thể giải thích cách thức áp dụng các nguyên tắc thiết kế và tối ưu hóa để cải thiện hiệu suất của hệ thống máy tính.
* Trình độ vận dụng: Khách hàng có thể thiết kế một hệ thống máy tính tối ưu dựa trên kiến thức về các nguyên tắc thiết kế và tối ưu hóa.

**Chuẩn đầu ra khóa học:**

Sau khi hoàn thành khóa học, sinh viên sẽ đạt được trình độ tương đương với CLOs trên và có khả năng:

* Hiểu và áp dụng kiến thức về kiến trúc máy tính, cơ chế thực hiện lệnh, các thuật toán quản lý bộ nhớ, kiến trúc bộ xử lý (CPU), và các nguyên tắc thiết kế và tối ưu hóa hệ thống máy tính.
* Thiết kế và tối ưu hóa một hệ thống máy tính đơn giản dựa trên kiến thức được học.
* Hiểu và giải thích các khái niệm liên quan đến tổ chức máy tính và kiến trúc máy tính.

Dựa vào thông tin được cung cấp (số lượng buổi học toàn khoá, thời lượng mỗi buổi, nội dung của sách và chuẩn đầu ra khóa học), hãy biên soạn kế hoạch giảng dạy dựa theo nội dung của sách [Thông tin sách]."""

thongtinsach="""Title: Financial Machine Learning 
Author: Bryan Kelly, Dacheng Xiu"""
# Thay thế đoạn text trong dấu [ ] bằng biến mới
prompt = text1.replace("[Mục lục ở đây]", result)
prompt = text1.replace("[Thông tin sách]", thongtinsach)
