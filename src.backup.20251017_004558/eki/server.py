import socket
import time

# 서버 설정
host = '127.0.0.1'  # 로컬호스트
port = 65432        # 사용할 포트 번호

# 소켓 생성 및 바인딩
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"서버가 {host}:{port}에서 대기 중입니다...")

# 클라이언트 연결 대기
conn, addr = server_socket.accept()
print(f"{addr}에서 연결되었습니다.")

# 데이터 수신 및 송신
with conn:
    while True:
        # 데이터 수신
        data = conn.recv(1024)  # 데이터 수신
        if not data:
            break
        number = int(data.decode())
        print(f"받은 숫자: {number}")

        # 0.3초 대기 후 숫자 증가
        time.sleep(0.3)
        number += 1
        print(f"{number}를 클라이언트로 전송합니다.")
        conn.sendall(str(number).encode())  # 응답 데이터 전송

# 소켓 종료
server_socket.close()
