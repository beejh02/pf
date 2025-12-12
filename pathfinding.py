import heapq
import numpy as np
import time
import os
from ultralytics import YOLO
import cv2

# -----------------------------
# 1. 초기 설정 (지도 & 모델)
# -----------------------------
# 0:벽, 1:길, 2:유실(장애물), 3:시작, 4:도착
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
HEIGHT, WIDTH = grid_map.shape

# YOLO 모델 로드 (루프 밖에서 한 번만 로드)
print("🤖 시스템 부팅 중... 모델을 로드합니다.")
model = YOLO("./best.pt")
img_path = "image2.png" # 실시간 카메라 대신 이미지를 사용한다고 가정

# -----------------------------
# 2. 다익스트라 (동적 경로 탐색용)
# -----------------------------
def get_path(grid, start, end):
    pq = [(0, start[0], start[1])]
    distances = np.full((HEIGHT, WIDTH), float('inf'))
    distances[start] = 0
    came_from = {start: None}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 상하좌우

    while pq:
        d, cy, cx = heapq.heappop(pq)

        if (cy, cx) == end:
            break

        if d > distances[cy, cx]:
            continue

        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < HEIGHT and 0 <= nx < WIDTH:
                # 벽(0)이거나 유실도로(2)면 못 감
                if grid[ny][nx] == 0 or grid[ny][nx] == 2:
                    continue
                
                new_dist = d + 1
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, ny, nx))
                    came_from[(ny, nx)] = (cy, cx)

    # 경로 역추적
    if distances[end] == float('inf'):
        return None
    
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return path

# -----------------------------
# 3. YOLO 감지 함수
# -----------------------------
def detect_and_update_map(current_grid, image_file):
    # 이미지 읽기
    frame = cv2.imread(image_file)
    h, w, _ = frame.shape
    
    # 예측
    results = model.predict(source=frame, conf=0.6, save=False, verbose=False)
    
    detected = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 중심점 계산 및 지도 좌표 변환
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            map_x = int((center_x / w) * WIDTH)
            map_y = int((center_y / h) * HEIGHT)
            
            # 범위 체크
            map_x = max(0, min(map_x, WIDTH - 1))
            map_y = max(0, min(map_y, HEIGHT - 1))

            # [중요] 지도가 '도로(1)'였던 곳에 객체가 있으면 '유실(2)'로 변경
            # 이미 2거나 벽(0)이면 굳이 업데이트 안 함
            if current_grid[map_y][map_x] == 1:
                current_grid[map_y][map_x] = 2
                print(f"\n⚠️ [경고] 전방 객체 탐지! 지도 업데이트: ({map_y}, {map_x}) -> 유실도로(2)")
                detected = True
                
    return detected

# -----------------------------
# 4. 메인 주행 루프
# -----------------------------
start_pos = tuple(np.argwhere(grid_map == 3)[0])
end_pos = tuple(np.argwhere(grid_map == 4)[0])

# 초기 경로 계산
current_path = get_path(grid_map, start_pos, end_pos)
current_pos = start_pos

if not current_path:
    print("❌ 시작부터 갈 수 있는 경로가 없습니다.")
    exit()

print(f"\n🚀 자율주행 시작! 목적지: {end_pos}")
time.sleep(1)

while current_pos != end_pos:
    os.system('cls' if os.name == 'nt' else 'clear') # 화면 클리어
    
    # 1. YOLO로 전방 주시 (지도 업데이트)
    # (실제 환경에선 카메라 영상을 넣겠지만, 여기선 image2.png를 계속 체크한다고 가정)
    is_map_changed = detect_and_update_map(grid_map, img_path)
    
    # 2. 지도가 바뀌었으면 경로 재계산 필요?
    if is_map_changed:
        # 현재 위치에서 다시 경로 계산
        print("🔄 지형 변경 감지! 경로를 재탐색합니다...")
        new_path = get_path(grid_map, current_pos, end_pos)
        
        if new_path:
            current_path = new_path
            # current_path[0]은 현재 위치이므로, 다음 이동은 [1]부터 해야 함
        else:
            print("⛔ 비상 정지! 목적지로 갈 수 있는 길이 모두 막혔습니다.")
            break

    # 3. 이동 로직 (경로의 다음 칸으로)
    # current_path[0]은 현재 내 위치, current_path[1]이 다음 갈 곳
    if len(current_path) > 1:
        next_pos = current_path[1] 
        current_pos = next_pos # 이동!
        
        # 경로 리스트 갱신 (이미 온 길 제거)
        current_path.pop(0) 
    
    # 4. 시각화 (현재 상태 출력)
    display_map = grid_map.copy()
    display_map[current_pos] = 9 # 내 차 위치 표시
    
    print(f"\n📍 현재 위치: {current_pos}")
    print("범례: [9:내차🚗] [0:벽⬛] [1:길⬜] [2:장애물❌]")
    print("-" * 30)
    print(display_map)
    print("-" * 30)
    
    if current_pos == end_pos:
        print("🎉 목적지에 도착했습니다!")
        break

    print("⏱️ 3초 대기 중... (다음 이동 준비)")
    # time.sleep(1)