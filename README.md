import cv2
import mediapipe as mp
import time

# Khởi tạo MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,               # Đổi thành 2 nếu muốn nhận diện cả 2 tay
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mở webcam (thử đổi 0 thành 1 nếu dùng webcam rời)
cap = cv2.VideoCapture(0)

prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không mở được camera.")
        break

    # Lật ảnh soi gương và chuyển BGR -> RGB
    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Xử lý nhận diện bàn tay
    results = hands.process(rgb_image)

    # Vẽ các khớp tay và xác định Trái / Phải
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Vẽ 21 khớp xương
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            # Lấy nhãn Tay Trái / Tay Phải (Left / Right)
            if results.multi_handedness:
                label = results.multi_handedness[idx].classification[0].label
                # Lấy tọa độ cổ tay (điểm landmark số 0) để ghi chữ
                h, w, _ = image.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)
                cv2.putText(image, label, (cx - 20, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tính toán FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Hiển thị FPS lên màn hình
    cv2.putText(image, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('NCKH - Nhan dien ban tay (MediaPipe)', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
hands.close()
cap.release()
cv2.destroyAllWindows()
