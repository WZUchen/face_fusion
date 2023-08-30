import cv2
from scripts.Realtime_video_face_changing import FakeFace_Driving
from draw.draw_single_face_mesh import visualize_face_landmarks
from draw.draw_many_face_mesh_swap import FaceSwapWithVirtualFace

# 实时数据流换脸
def realtime_face_swap(fake_face_url, show_webcam=True, max_people=1):
    # 初始化 FakeFace_Driving 类
    draw_fake_face = FakeFace_Driving(fake_face_url, show_webcam, max_people)
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Fake face", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            continue
        # 水平翻转图像
        frame = cv2.flip(frame, 1)
        ret, fake_face = draw_fake_face(frame)
        if not ret:
            continue
        cv2.imshow("Fake face", fake_face)
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# 静态换脸
def static_face_swap(fake_face_url,real_img_url, show_webcam=True, max_people=1):
    draw_fake_face = FakeFace_Driving(fake_face_url, show_webcam, max_people)
    real_img = cv2.imread(real_img_url)
    ret,fake_face = draw_fake_face(real_img)
    # 显示结果图像
    cv2.imshow('Face Swap Result', fake_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 处理一组图像
def draw_face_swap_results(fake_face_url, files_url, save_path):
    face_swap_instance = FaceSwapWithVirtualFace(fake_face_url)
    face_swap_instance.show_result(files_url=files_url, save_path=save_path)

if __name__ == '__main__':
    # 静态 动态换脸
    fake_face_url = "./data/img/1.jpg"
    real_img_url = './data/img/test1.png'

    # 创建类实例并调用 show_result 方法展示结果
    files_url = './draw/test_data/test_group_1'
    save_path = './draw/test_data/'

    # 提示用户选择
    choice = input("请选择要执行的操作：\n1. 实时人脸交换\n2. 静态人脸交换\n3. 生成结果\n请输入选项编号（1/3）：")
    if choice == '1':
        # 实时人脸交换 实时需等待40s左右唤醒摄像头
        realtime_face_swap(fake_face_url, show_webcam=True, max_people=1)
    elif choice == '2':
        # 静态人脸交换
        result_image = static_face_swap(fake_face_url, real_img_url, show_webcam=True, max_people=1)
    elif choice =='3':
        draw_face_swap_results(fake_face_url, files_url, save_path)
    else:
        print("无效的选项。请重新运行并输入正确的选项编号。")
