"""
Kalman Filter 직접 구현 기반 객체 추적 (Windows 노트북 웹캠 버전)

원본: object_tracking.py (Jetson + CSI 카메라 / GStreamer nvarguscamerasrc)
변경: Windows 내장/USB 웹캠 (DirectShow 백엔드)

파란 원  = 실제 검출된 객체 (measurement)
노란 원  = Kalman Filter 가 추정한 위치 (belief)
초록 물체를 손으로 가려보면 노란 원이 예측만으로 따라오는 것을 볼 수 있습니다.

실행:
    python w_object_tracking.py
종료:
    영상 창을 클릭한 뒤 'q' 키
"""

import numpy as np
from numpy.linalg import inv
import cv2
import time


# --------------------------- 카메라 설정 ---------------------------
# Windows 웹캠 인덱스. 노트북 내장 캠은 보통 0,
# USB 캠을 추가로 꽂았다면 1 또는 2 로 바꿔가며 시도하세요.
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 25

# 노트북 웹캠은 Jetson CSI 카메라와 달리 상하 반전이 없습니다.
# 대신 거울처럼 좌우 반전(1)을 하면 실습 시 조작이 직관적입니다.
# 반전을 원하지 않으면 None 으로 두세요. (0=상하, 1=좌우, -1=상하좌우)
FLIP_CODE = 1
# -------------------------------------------------------------------


def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 객체 최초 검출 여부 확인용 boolean
found = False

# 객체 좌표 및 반지름 변수 정의
x_bel, y_bel = 0, 0
radius = 0

# Windows 에서는 GStreamer 파이프라인 대신 DirectShow 백엔드로 카메라를 엽니다.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        f"카메라({CAMERA_INDEX})를 열 수 없습니다.\n"
        "  1) 다른 앱(Zoom, Teams, 카메라 앱)이 웹캠을 쓰고 있는지 확인\n"
        "  2) 설정 > 개인 정보 및 보안 > 카메라 에서 데스크톱 앱 접근 허용\n"
        "  3) CAMERA_INDEX 값을 1, 2 로 바꿔서 재시도"
    )

# MJPG 로 받아야 대부분의 웹캠에서 720p 30fps 가 나옵니다. (기본 YUY2 는 매우 느림)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
        break

    if FLIP_CODE is not None:
        frame = cv2.flip(frame, FLIP_CODE)

    # 가우시안 블러
    blr = cv2.GaussianBlur(frame, (11, 11), 0)

    # LAB 색공간 변환
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # LAB color segmentation
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Opening 2회, Dilation 2회
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Contour detection
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
        _, radius = cv2.minEnclosingCircle(contour)
        # 무게중심
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # 객체 최초 검출
        if not found:
            found = True

    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 예측한 객체 위치에 노란 원 overlay
    cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 2)
    cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0, 255, 255), -1)

    cv2.imshow("Object Tracking", frame)
    # cv2.imshow("LAB Mask", mask)

    # imshow 로 창이 만들어진 뒤에 키 입력을 받아야 'q' 가 확실히 동작합니다.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # while loop rate (FPS) 설정
    time.sleep(max(1. / FPS - (time.time() - start), 0))

cap.release()
cv2.destroyAllWindows()
