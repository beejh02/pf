from ultralytics import YOLO
import cv2
import numpy as np
import os

# -----------------------------
# 1. 유틸리티 함수: 라운드 박스 그리기
# -----------------------------
def draw_round_rect(img, x1, y1, x2, y2, color, thickness=2, radius=10):
    points = [(x1+radius, y1), (x2-radius, y1), (x2, y1+radius), (x2, y2-radius),
              (x2-radius, y2), (x1+radius, y2), (x1, y2-radius), (x1, y1+radius)]
    
    cv2.line(img, points[0], points[1], color, thickness)
    cv2.line(img, points[2], points[3], color, thickness)
    cv2.line(img, points[4], points[5], color, thickness)
    cv2.line(img, points[6], points[7], color, thickness)
    
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)

# -----------------------------
# 2. 지도 및 모델 설정
# -----------------------------
custom_map_data = [
    [0,0,0,0,0,4],
    [1,1,1,1,1,1],
    [1,0,0,0,0,1],
    [1,1,1,1,1,1],
    [1,0,1,0,0,1],
    [1,0,1,0,0,1],
    [1,0,1,0,0,1],
    [1,1,1,1,1,1],
    [0,0,0,0,0,3]
]

grid_map = np.array(custom_map_data)
grid_map_height, grid_map_width = grid_map.shape 

print(f"지도 로드 완료! 크기: {grid_map_width} x {grid_map_height}")

model = YOLO("./best.pt")
img_path = "image2.png"

# -----------------------------
# 3. 객체 감지 및 로직 수행
# -----------------------------
results = model.predict(source=img_path, conf=0.8, save=False, show=False)
orig_img = cv2.imread(img_path)
h, w, _ = orig_img.shape

for r in results:
    boxes = r.boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 시각화
            draw_round_rect(orig_img, x1, y1, x2, y2, (0, 255, 0), thickness=2, radius=12)
            label = f"{cls_id} {conf:.2f}"
            cv2.putText(orig_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 지도 업데이트 로직
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            map_x = int((center_x / w) * grid_map_width)
            map_y = int((center_y / h) * grid_map_height)
            map_x = max(0, min(map_x, grid_map_width - 1))
            map_y = max(0, min(map_y, grid_map_height - 1))

            print(f"   >>> 객체 감지됨! 지도 업데이트: ({map_x}, {map_y}) -> 2")
            grid_map[map_y][map_x] = 2

# 이미지 저장
cv2.imwrite("result_round_box.jpg", orig_img)

# -----------------------------
# 4. 저장 및 검증 (수정된 부분)
# -----------------------------

# (1) .npy 파일로 저장
file_name = 'updated_map.npy'
np.save(file_name, grid_map)
print(f"\n[저장 완료] 현재 지도 상태를 '{file_name}'에 저장했습니다.")

# (2) .npy 파일을 다시 불러와서 확인 (검증)
print("\n--- 저장된 .npy 파일 검증(Load & Print) ---")

# 파일이 실제로 존재하는지 확인
if os.path.exists(file_name):
    loaded_data = np.load(file_name) # 여기서 파일을 읽어옵니다.
    print(loaded_data)
    
    # 데이터 무결성 체크 (메모리에 있는 것과 파일에서 읽은 것이 같은지)
    if np.array_equal(grid_map, loaded_data):
        print("\n>> 검증 성공: 메모리 데이터와 파일 데이터가 정확히 일치합니다.")
    else:
        print("\n>> 검증 실패: 데이터가 다릅니다.")
else:
    print(f"오류: {file_name} 파일을 찾을 수 없습니다.")