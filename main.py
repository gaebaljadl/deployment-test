from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

app = Flask(__name__)

# 모델 생성
resnet18 = models.resnet18(weights=None)

# 로컬 가중치 파일 로드
weights_path = '/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
resnet18.load_state_dict(torch.load(weights_path))
resnet18.eval()

# 이미지 전처리를 위한 변환 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 입력 이미지를 전처리하고 모델에 전달하여 예측을 생성하는 함수
def predict_image(image):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet18(image_tensor)
    return output

@app.route('/predict', methods=['POST'])
def predict():
    # 입력 이미지 받기
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))

    # 예측 생성
    output = predict_image(image)

    # 결과를 리스트로 변환
    output_list = output.tolist()

    # 가장 큰 값의 인덱스 찾기
    max_index = output_list[0].index(max(output_list[0]))

    # 결과 반환
    return jsonify({'predicted_class': max_index}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
