# 베이스 이미지 지정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY resnet18-f37072fd.pth /root/.cache/torch/hub/checkpoints/
COPY . .

# 환경변수 설정
ENV FLASK_APP=main.py

# 애플리케이션 실행
CMD ["flask", "run", "--host=0.0.0.0"]
