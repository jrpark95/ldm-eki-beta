"""
TCP Test Server (Legacy)

This is a simple TCP echo server used for testing socket-based communication.
This module is deprecated and kept only for backward compatibility testing.
Production systems use POSIX shared memory IPC instead.

Author:
    Siho Jang, 2025

Notes:
    - DO NOT use for production IPC
    - Replaced by POSIX shared memory (see eki_ipc_reader.py, eki_ipc_writer.py)
    - Kept only for testing legacy TCP socket interface
"""

import socket
import time

# Server configuration
host = '127.0.0.1'  # Localhost
port = 65432        # Port number

# Create socket and bind
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"Server waiting at {host}:{port}...")

# Wait for client connection
conn, addr = server_socket.accept()
print(f"Connected from {addr}.")

# Receive and send data
with conn:
    while True:
        # Receive data
        data = conn.recv(1024)  # Receive data
        if not data:
            break
        number = int(data.decode())
        print(f"Received number: {number}")

        # Wait 0.3 seconds and increment number
        time.sleep(0.3)
        number += 1
        print(f"Sending {number} to client.")
        conn.sendall(str(number).encode())  # Send response data

# Close socket
server_socket.close()
