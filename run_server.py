import subprocess
import time
from pyngrok import ngrok

# Flask 서버 시작
server_process = subprocess.Popen(["python", "app.py"])
print("Flask 서버가 시작되었습니다.")

# ngrok 터널 생성
http_tunnel = ngrok.connect(3000)
print(f"ngrok 터널이 생성되었습니다: {http_tunnel.public_url}")
print(f"모바일에서 접속하세요: {http_tunnel.public_url}")

try:
    # 앱이 계속 실행되도록 대기
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # 종료 시 프로세스 정리
    server_process.terminate()
    ngrok.kill()
