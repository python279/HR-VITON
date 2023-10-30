from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from base64 import b64decode, b64encode
from hr_viton import HrViton

app = Flask(__name__)


hr_viton = HrViton()


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()  # 获取 POST 请求中的 JSON 数据
    for input in data:
        if input['type'] == 'image':
            input['data'] = Image.open(BytesIO(b64decode(input['data'])))
        elif input['type'] == 'json':
            input['data'] = input['data']
    output_image = hr_viton(data)
    if output_image:
        buffered = BytesIO()
        output_image.save(buffered, format='JPEG')
        output_image = b64encode(buffered.getvalue()).decode('utf-8')
    result = {'result': {'output_image': output_image}}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
