import cv2
import mediapipe as mp
import numpy as np
from utils.Triangle_patch_index import vertices

# 实时fake人脸融合驱动类
class FakeFace_Driving():

    def __init__(self, fake_face_url, show_webcam=True, max_people=1, detection_confidence=0.3):

        self.show_webcam = show_webcam

        self.initialize_model(max_people, detection_confidence)

        self.read_fakeface_image(fake_face_url)

        self.detect_fakeface_mesh()

    def __call__(self, image):

        return self.detect_and_draw_fake_face(image)

    def detect_and_draw_fake_face(self, image):
        landmarks = self.detect_realface_mesh(image)

        if not landmarks:
            return False, None

        return True, self.draw_fake_face(image, landmarks)

    def initialize_model(self, max_people, detection_confidence):

        # 初始化人脸特征点检测器 face_mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                                         max_num_faces=max_people,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=detection_confidence)

    def read_fakeface_image(self, fake_face_url):
        self.fake_face = cv2.imread(fake_face_url)
        self.fk_im_height, self.fk_im_width, _ = self.fake_face.shape

    def detect_fakeface_mesh(self):
        # 检测虚假人脸 返回特征点的位置(x，y)
        landmarks = self.face_mesh.process(cv2.cvtColor(self.fake_face, cv2.COLOR_BGR2RGB)).multi_face_landmarks
        self.fake_face_coordinates = np.array(
            [[int(landmark.x * self.fk_im_width), int(landmark.y * self.fk_im_height)] for landmark in
             landmarks[0].landmark])

    def detect_realface_mesh(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image.flags.writeable = False
        return self.face_mesh.process(input_image).multi_face_landmarks

    def draw_fake_face(self, img, landmarks):
        #   创建了一个名为 converted_image 的空白图像，与输入图像 img 具有相同的形状和数据类型。 converted_image 将用于绘制fake脸部特征。
        converted_image = np.zeros(img.shape, dtype=np.uint8)
        # 通过遍历 landmarks 列表中的每个元素 face_landmarks，我们将获取每个人脸的面部关键点坐标并进行处理。
        for face_landmarks in landmarks:
            face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])),
                                          int(min([landmark.y * img.shape[0], img.shape[0] - 1]))] for landmark in
                                         face_landmarks.landmark]).astype(int)
            face_coordinates[face_coordinates < 0] = 0
            # 遍历 vertices 数组，每次迭代处理一个三角形
            for triangle_id in range(0, len(vertices), 3):
                # 获取三角形的顶点坐标索引
                corner1_id = vertices[triangle_id][0]
                corner2_id = vertices[triangle_id + 1][0]
                corner3_id = vertices[triangle_id + 2][0]

                # 从fake_face_coordinates中选择对应顶点的坐标
                exorcist_pix_coords = self.fake_face_coordinates[[corner1_id, corner2_id, corner3_id], :]

                # 从face_coordinates中选择对应顶点的坐标
                face_pix_coords = face_coordinates[[corner1_id, corner2_id, corner3_id], :]

                # 将图像裁剪到带有三角形的剖面 （裁剪的结果是一个矩形区域，该区域包含了所需的三角形区域。）
                ex_x, ex_y, ex_w, ex_h = cv2.boundingRect(exorcist_pix_coords)
                face_x, face_y, face_w, face_h = cv2.boundingRect(face_pix_coords)
                cropped_exorcist = self.fake_face[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w]
                cropped_face = img[face_y:face_y + face_h, face_x:face_x + face_w]

                # 需要更新三角形的坐标
                exorcist_pix_crop_coords = exorcist_pix_coords.copy()
                face_pix_crop_coords = face_pix_coords.copy()
                exorcist_pix_crop_coords[:, 0] -= ex_x
                exorcist_pix_crop_coords[:, 1] -= ex_y
                face_pix_crop_coords[:, 0] -= face_x
                face_pix_crop_coords[:, 1] -= face_y

                # 获取裁剪后的人脸图像中三角形区域的掩码
                cropped_face_mask = np.zeros((face_h, face_w), np.uint8)
                triangle = (np.round(np.array([face_pix_crop_coords]))).astype(int)
                cv2.fillConvexPoly(cropped_face_mask, triangle, 255)

                # 将原始图像中的一个三角形区域（fake_face中的三角形）进行仿射变换，并将其映射到目标图像中的对应三角形区域（cropped_face）上 通过这个过程，可以实现将原始图像中的三角形区域无缝地贴合到目标图像中的对应位置，以达到图像的局部特征对齐的效果。通常用于实现图像合成、人脸对齐等应用。
                warp_mat = cv2.getAffineTransform(exorcist_pix_crop_coords.astype(np.float32),
                                                  face_pix_crop_coords.astype(np.float32))
                warped_triangle = cv2.warpAffine(cropped_exorcist, warp_mat, (face_w, face_h),borderMode=cv2.BORDER_REPLICATE)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_face_mask)

                # 可以将多个三角形进行逐一处理，并将它们无缝地融入到目标图像中，以生成合成图像
                cropped_new_face = converted_image[face_y:face_y + face_h, face_x:face_x + face_w]
                cropped_new_face_gray = cv2.cvtColor(cropped_new_face, cv2.COLOR_BGR2GRAY)
                _, non_filled_mask = cv2.threshold(cropped_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=non_filled_mask)
                cropped_new_face = cv2.add(cropped_new_face, warped_triangle)
                converted_image[face_y:face_y + face_h, face_x:face_x + face_w] = cropped_new_face

        # 将合成图像（converted_image）与实时摄像头图像（img）进行叠加，以将生成的面部特效或变形效果添加到摄像头图像中。
        if self.show_webcam:
            converted_image_gray = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
            _, non_drawn_mask = cv2.threshold(converted_image_gray, 1, 255, cv2.THRESH_BINARY_INV)
            # img = cv2.bitwise_and(img, img, mask=non_drawn_mask)
            non_drawn_mask = cv2.bitwise_not(non_drawn_mask)
            face_center = tuple(((np.max(face_coordinates, axis=0) + np.min(face_coordinates, axis=0)) / 2).astype(int))
            seamless_clone = cv2.seamlessClone(converted_image, img, non_drawn_mask, face_center, cv2.MIXED_CLONE)
            # converted_image = cv2.add(converted_image, img)
            # normal_clone = adjust_gamma(seamless_clone, gamma=2)
        return seamless_clone
