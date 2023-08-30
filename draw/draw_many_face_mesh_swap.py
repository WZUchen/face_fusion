import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from utils.Triangle_patch_index import vertices


class FaceSwapWithVirtualFace:
    def __init__(self, virtural_img_path):
        self.virtural_img_path = virtural_img_path
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    def extract_number(self, file_name):
        match = re.search(r'\d+', file_name)
        if match:
            return int(match.group())
        return 0

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def blend_face(self, img, face_coordinates, non_drawn_mask, converted_image):
        face_center = tuple(((np.max(face_coordinates, axis=0) + np.min(face_coordinates, axis=0)) / 2).astype(int))
        seamless_clone = cv2.seamlessClone(converted_image, img, non_drawn_mask, face_center, cv2.MIXED_CLONE)
        return seamless_clone

    def face_mesh_detection(self, image_path):
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                        min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(image_path)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            annotated_image = image.copy()
            white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=white_background,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
            return white_background

    def detect_face_mesh(self, image_path):
        real_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        input_image.flags.writeable = False
        return self.face_mesh.process(input_image).multi_face_landmarks

    def detect_virtual_key_points(self, image_path):
        virtural_image = cv2.imread(image_path)
        ex_im_height, ex_im_width, _ = virtural_image.shape
        landmarks = self.face_mesh.process(cv2.cvtColor(virtural_image, cv2.COLOR_BGR2RGB)).multi_face_landmarks
        virtural_face_coordinates = np.array(
            [[int(landmark.x * ex_im_width), int(landmark.y * ex_im_height)] for landmark in landmarks[0].landmark])
        return virtural_face_coordinates

    def draw_virtual_face(self, real_image_path, virtural_image_path, real_face_landmarks, virtural_face_coordinates):
        real_image = cv2.imread(real_image_path)
        virtural_image = cv2.imread(virtural_image_path)
        converted_real_image = np.zeros(real_image.shape, dtype=np.uint8)
        for face_landmarks in real_face_landmarks:
            real_face_coordinates = np.array([[int(min([landmark.x * real_image.shape[1], real_image.shape[1] - 1])),
                                               int(min([landmark.y * real_image.shape[0], real_image.shape[0] - 1]))]
                                              for landmark in face_landmarks.landmark]).astype(int)
            real_face_coordinates[real_face_coordinates < 0] = 0

            for triangle_id in range(0, len(vertices), 3):
                corner1_id = vertices[triangle_id][0]
                corner2_id = vertices[triangle_id + 1][0]
                corner3_id = vertices[triangle_id + 2][0]

                virtural_pix_coords = virtural_face_coordinates[[corner1_id, corner2_id, corner3_id], :]
                face_pix_coords = real_face_coordinates[[corner1_id, corner2_id, corner3_id], :]

                ex_x, ex_y, ex_w, ex_h = cv2.boundingRect(virtural_pix_coords)
                face_x, face_y, face_w, face_h = cv2.boundingRect(face_pix_coords)
                cropped_virtural = virtural_image[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w]
                cropped_face = real_image[face_y:face_y + face_h, face_x:face_x + face_w]

                virtural_pix_crop_coords = virtural_pix_coords.copy()
                face_pix_crop_coords = face_pix_coords.copy()
                virtural_pix_crop_coords[:, 0] -= ex_x
                virtural_pix_crop_coords[:, 1] -= ex_y
                face_pix_crop_coords[:, 0] -= face_x
                face_pix_crop_coords[:, 1] -= face_y

                cropped_face_mask = np.zeros((face_h, face_w), np.uint8)
                triangle = (np.round(np.array([face_pix_crop_coords]))).astype(int)
                cv2.fillConvexPoly(cropped_face_mask, triangle, 255)

                warp_mat = cv2.getAffineTransform(virtural_pix_crop_coords.astype(np.float32),
                                                  face_pix_crop_coords.astype(np.float32))
                warped_triangle = cv2.warpAffine(cropped_virtural, warp_mat, (face_w, face_h),
                                                 borderMode=cv2.BORDER_REPLICATE)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_face_mask)

                cropped_new_face = converted_real_image[face_y:face_y + face_h, face_x:face_x + face_w]
                cropped_new_face_gray = cv2.cvtColor(cropped_new_face, cv2.COLOR_BGR2GRAY)
                _, non_filled_mask = cv2.threshold(cropped_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=non_filled_mask)
                cropped_new_face = cv2.add(cropped_new_face, warped_triangle)
                converted_real_image[face_y:face_y + face_h, face_x:face_x + face_w] = cropped_new_face

        converted_real_image_gray = cv2.cvtColor(converted_real_image, cv2.COLOR_BGR2GRAY)
        _, non_drawn_mask = cv2.threshold(converted_real_image_gray, 1, 255, cv2.THRESH_BINARY_INV)
        img = cv2.bitwise_and(real_image, real_image, mask=non_drawn_mask)
        non_drawn_mask = cv2.bitwise_not(non_drawn_mask)
        converted_real_image1 = cv2.add(converted_real_image, img)
        return converted_real_image1,real_face_coordinates,non_drawn_mask,converted_real_image

    def process_image(self, image_path):
        real_image = cv2.imread(image_path)
        face_coordinates = self.detect_face_mesh(image_path)
        face_keypoints = self.face_mesh_detection(image_path)
        virtural_face_coordinates = self.detect_virtual_key_points(self.virtural_img_path)
        converted_real_image, real_face_coordinates, non_drawn_mask, mask = self.draw_virtual_face(image_path, self.virtural_img_path,
                                                                            face_coordinates, virtural_face_coordinates)
        seamlessclone = self.blend_face(real_image, real_face_coordinates, non_drawn_mask, converted_real_image)
        return real_image, face_keypoints, converted_real_image, seamlessclone,mask

    def show_result(self,files_url,save_path):
        frame_files = os.listdir(files_url)
        frame_files = [file for file in frame_files if file.endswith(".jpg") or file.endswith(".png")]
        frame_files.sort(key=self.extract_number)
        fig, axes = plt.subplots(5, 5,figsize=(50, 50))
        # plt.subplots_adjust(right=0.3, top=0.29, wspace=0.01, hspace=0.01)

        for i, file in enumerate(frame_files[:5]):
            img_path = os.path.join(files_url, file)
            real_image, face_keypoints, converted_real_image, seamlessclone,mask = self.process_image(img_path)

            # 在第1行子图上展示原始图像
            axes[i // 5, i % 5].imshow(real_image[:, :, ::-1])
            axes[i // 5, i % 5].axis("off")

            # 在2子图上展示原始图像
            axes[(i // 5) + 1, i % 5].imshow(face_keypoints[:, :, ::-1])
            axes[(i // 5) + 1, i % 5].axis("off")

            # 在2子图上展示原始图像
            axes[(i // 5) + 2, i % 5].imshow(mask[:, :, ::-1])
            axes[(i // 5) + 2, i % 5].axis("off")

            # 在第3行子图上展示处理后的结果
            axes[(i // 5) + 3, i % 5].imshow(converted_real_image[:, :, ::-1])
            axes[(i // 5) + 3, i % 5].axis("off")

            # 在第4行子图上展示处理后的结果
            axes[(i // 5) + 4, i % 5].imshow(seamlessclone[:, :, ::-1])
            axes[(i // 5) + 4, i % 5].axis("off")
        plt.savefig(save_path)
        plt.show()


# 创建类实例并调用 show_result 方法展示结果
virtural_img_path = '../data/img/1-7.png'
files_url = './test_data/test_group_1'
save_path = './test_data/'
face_swap_instance = FaceSwapWithVirtualFace(virtural_img_path)
face_swap_instance.show_result(files_url=files_url,save_path=save_path)
