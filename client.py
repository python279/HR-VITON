import requests
from PIL import Image
from io import BytesIO
import base64
import json


# 准备输入数据
inputs = [
    {'name': 'cloth', 'type': 'image', 'path': './datasets/images/cloth/06396_00.jpg', 'id': '06396_00', 'filename': '06396_00.jpg'},
    {'name': 'cloth-mask', 'type': 'image', 'path': './datasets/images/cloth-mask/06396_00.jpg', 'id': '06396_00', 'filename': '06396_00.jpg'},
    {'name': 'cloth', 'type': 'image', 'path': './datasets/images/cloth/00006_00.jpg', 'id': '00006_00', 'filename': '00006_00.jpg'},
    {'name': 'cloth-mask', 'type': 'image', 'path': './datasets/images/cloth-mask/00006_00.jpg', 'id': '00006_00', 'filename': '00006_00.jpg'},
    {'name': 'image', 'type': 'image', 'path': './datasets/images/image/00006_00.jpg', 'id': '00006_00', 'filename': '00006_00.jpg'},
    {'name': 'image-densepose', 'type': 'image', 'path': './datasets/images/image-densepose/00006_00.jpg', 'id': '00006_00', 'filename': '00006_00.jpg'},
    {'name': 'image-parse-agnostic-v3.2', 'type': 'image', 'path': './datasets/images/image-parse-agnostic-v3.2/00006_00.png', 'id': '00006_00', 'filename': '00006_00.png'},
    {'name': 'image-parse-v3', 'type': 'image', 'path': './datasets/images/image-parse-v3/00006_00.png', 'id': '00006_00', 'filename': '00006_00.png'},
    {'name': 'openpose_img', 'type': 'image', 'path': './datasets/images/openpose_img/00006_00_rendered.png', 'id': '00006_00', 'filename': '00006_00_rendered.png'},
    {'name': 'openpose_json', 'type': 'json', 'path': './datasets/images/openpose_json/00006_00_keypoints.json', 'id': '00006_00', 'filename': '00006_00_keypoints.json'},
]

for input in inputs:
    if input['type'] == 'image':
        with open(input['path'], 'rb') as f:
            image = f.read()
        # 将图像数据进行 base64 编码
        encoded_image = base64.b64encode(image).decode('utf-8')
        input['data'] = encoded_image
    elif input['type'] == 'json':
        with open(input['path'], 'rb') as f:
            json_data = json.load(f)
        input['data'] = json_data

# 构造请求数据
data = inputs
headers = {'Content-Type': 'application/json'}

# 发送 POST 请求到服务器
url = 'http://127.0.0.1:5000/infer'  # 根据实际情况修改 URL
response = requests.post(url, data=json.dumps(data), headers=headers)

# 解析服务器返回的 JSON 数据
result = response.json()

# 推理结果图像数据
output_image_data = result['result']['output_image']

# 解码返回的图像数据
decoded_image = base64.b64decode(output_image_data)

# 将解码后的图像数据加载为 PIL 图像对象
output_image = Image.open(BytesIO(decoded_image))

# 展示图像
output_image.show()
