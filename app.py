from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mediapipe as mp
import cv2
import numpy as np
import base64
import io
import os
import json

app = Flask(__name__, template_folder='./www', static_folder='./www', static_url_path='/')
CORS(app)

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 얼굴 데이터 저장소 (실제로는 데이터베이스 사용 권장)
face_database = {}

# 데이터 디렉토리 생성
os.makedirs('face_data', exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('www', 'index.html')

def decode_image(image_data):
    # Base64 이미지 디코딩
    image_data = image_data.split(',')[1]  # 'data:image/jpeg;base64,' 부분 제거
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def extract_face_features(image):
    # MediaPipe로 얼굴 특징점 추출
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # RGB로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출
        results = face_detection.process(rgb_image)
        
        if results.detections:
            faces = []
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # 얼굴 영역 크롭
                face_crop = image[y:y+h, x:x+w]
                
                # 얼굴 메쉬로 특징점 추출
                with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_results = face_mesh.process(face_rgb)
                    
                    features = []
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            # 주요 특징점 추출 (예: 눈, 코, 입 등)
                            for landmark in face_landmarks.landmark:
                                features.extend([landmark.x, landmark.y, landmark.z])
                    
                    faces.append({
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'features': features
                    })
            
            return faces
    return None

def calculate_similarity(features1, features2):
    # 유클리드 거리 기반 유사도 계산
    if len(features1) != len(features2):
        return 0.0
    
    distance = np.linalg.norm(np.array(features1) - np.array(features2))
    # 거리를 유사도로 변환 (0~1)
    similarity = max(0, 1 - distance / 10)
    return similarity

@app.route('/register', methods=['POST'])
def register_face():
    try:
        data = request.json
        image = decode_image(data['image'])
        name = data['name']
        
        # 얼굴 특징 추출
        faces = extract_face_features(image)
        
        if not faces:
            return jsonify({
                'success': False,
                'message': '얼굴을 찾을 수 없습니다.'
            })
        
        # 이미 등록된 얼굴 체크
        for db_name, db_features in face_database.items():
            for face in faces:
                similarity = calculate_similarity(face['features'], db_features)
                if similarity > 0.95:
                    return jsonify({
                        'success': False,
                        'message': f'이미 등록된 얼굴입니다 (유사도: {similarity*100:.1f}%)'
                    })
        
        # 얼굴 등록
        face_database[name] = faces[0]['features']
        
        # 데이터 저장
        with open(f'face_data/{name}.json', 'w') as f:
            json.dump(faces[0]['features'], f)
        
        return jsonify({
            'success': True,
            'message': f'"{name}" 등록 완료'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        image = decode_image(data['image'])
        
        # 얼굴 특징 추출
        faces = extract_face_features(image)
        
        if not faces:
            return jsonify({
                'success': False,
                'message': '얼굴을 찾을 수 없습니다.'
            })
        
        result = {'faces': []}
        
        for face in faces:
            best_match = None
            best_similarity = 0
            
            for name, db_features in face_database.items():
                similarity = calculate_similarity(face['features'], db_features)
                if similarity > best_similarity and similarity > 0.7:
                    best_similarity = similarity
                    best_match = name
            
            face_result = {
                'x': face['bbox']['x'],
                'y': face['bbox']['y'],
                'width': face['bbox']['width'],
                'height': face['bbox']['height'],
                'name': best_match,
                'confidence': best_similarity
            }
            result['faces'].append(face_result)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

# 저장된 얼굴 데이터 로드
def load_face_data():
    global face_database
    face_data_dir = 'face_data'
    
    if os.path.exists(face_data_dir):
        for filename in os.listdir(face_data_dir):
            if filename.endswith('.json'):
                name = filename[:-5]  # .json 제거
                with open(os.path.join(face_data_dir, filename), 'r') as f:
                    face_database[name] = json.load(f)

# 앱 시작 시 데이터 로드
load_face_data()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
