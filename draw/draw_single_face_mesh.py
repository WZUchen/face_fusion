import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

def visualize_face_landmarks(image_path):
    # 绘图工具、绘图风格、面部特征
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # 设置绘图效果工具
    drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)

    # 创建面部检测对象
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(image_path)
        # 检测人脸面部点位
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated_image = image.copy()

        # 根据检测到的面部标记点绘制图形
        for face_landmarks in results.multi_face_landmarks:

            # 人脸关键点
            face_keypoints = []
            for landmark in face_landmarks.landmark:
                face_keypoints.append((landmark.x, landmark.y, landmark.z))

            # 绘制网格
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        plt.figure(figsize=(10.24, 10.24))  # 设置图像大小为 1024x1024 像素
        plt.imshow(annotated_image[:, :, ::-1])
        plt.show()

