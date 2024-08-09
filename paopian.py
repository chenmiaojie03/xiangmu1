# import cv2
# import numpy as np
#
# def process_frame(frame, roi):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, [roi], 255)
#     edges_roi = cv2.bitwise_and(edges, mask)
#     return edges_roi
#
# def draw_and_detect_contours(frame, edges_roi, roi):
#     contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     output_frame = frame.copy()
#     cv2.polylines(output_frame, [roi], True, (0, 255, 0), 2)
#
#     if len(contours) < 4:
#         return None, None, None, None, output_frame
#
#     contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort from left to right
#
#     belt_left_contour = contours[1]
#     belt_right_contour = contours[-1]
#     material_left_contour = contours[1]
#     material_right_contour = contours[-4]
#
#     cv2.drawContours(output_frame, [belt_left_contour], -1, (255, 0, 0), 2)  # Blue
#     cv2.drawContours(output_frame, [belt_right_contour], -1, (0, 255, 255), 2)
#     cv2.drawContours(output_frame, [material_left_contour], -1, (0, 0, 255), 2)  # Red
#     cv2.drawContours(output_frame, [material_right_contour], -1, (255, 0, 255), 2)  # Red
#
#     return belt_left_contour, belt_right_contour, material_left_contour, material_right_contour, output_frame
#
# def calculate_deviation(belt_left_contour, belt_right_contour, material_left_contour, material_right_contour):
#     if any(contour is None for contour in [belt_left_contour, belt_right_contour, material_left_contour, material_right_contour]):
#         return 0, 0
#
#     belt_left_x = np.mean([point[0][0] for point in belt_left_contour])
#     belt_right_x = np.mean([point[0][0] for point in belt_right_contour])
#     material_left_x = np.mean([point[0][0] for point in material_left_contour])
#     material_right_x = np.mean([point[0][0] for point in material_right_contour])
#
#     belt_center = (belt_left_x + belt_right_x) / 2
#     material_center = (material_left_x + material_right_x) / 2
#
#     deviation = material_center - belt_center
#
#     return deviation
# def main():
#     video_path = r'D:\python\pythonProject\pidaipaopian.mp4'
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print("无法打开视频源，请检查路径或设备连接")
#         return
#
#     roi = np.array([[260, 80], [493, 80], [493, 225], [255, 225]], dtype=np.int32)
#     captured_alert = False
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("视频读取完毕或出现错误")
#             break
#
#         edges_roi = process_frame(frame, roi)
#         belt_left, belt_right, material_left, material_right, output_frame = draw_and_detect_contours(frame, edges_roi, roi)
#
#         deviation = calculate_deviation(belt_left, belt_right, material_left, material_right)
#
#         if abs(deviation) > 30:  # 偏离阈值，单位像素
#             print(f"物料偏移量: {deviation:.2f} 像素")
#             cv2.imwrite('offset_image.jpg', output_frame)
#             captured_alert = True
#
#         cv2.imshow("Original Frame", frame)
#         cv2.imshow("Contours in ROI", output_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()




import cv2
import numpy as np

def process_frame(frame, roi):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [roi], 255)
    edges_roi = cv2.bitwise_and(edges, mask)
    return edges_roi

def draw_and_detect_contours(frame, edges_roi, roi):
    contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_frame = frame.copy()
    cv2.polylines(output_frame, [roi], True, (0, 255, 0), 2)

    if len(contours) < 4:
        return None, None, None, None, output_frame

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort from left to right

    belt_left_contour = contours[1]
    belt_right_contour = contours[-2]
    material_left_contour = contours[2]
    material_right_contour = contours[-3]

    cv2.drawContours(output_frame, [belt_left_contour], -1, (255, 0, 0), 2)  # Blue
    cv2.drawContours(output_frame, [belt_right_contour], -1, (0, 255, 255), 2)
    cv2.drawContours(output_frame, [material_left_contour], -1, (0, 0, 255), 2)
    cv2.drawContours(output_frame, [material_right_contour], -1, (255, 0, 255), 2)

    return belt_left_contour, belt_right_contour, material_left_contour, material_right_contour, output_frame

def calculate_deviation(belt_left_contour, belt_right_contour, material_left_contour, material_right_contour):
    if any(contour is None for contour in [belt_left_contour, belt_right_contour, material_left_contour, material_right_contour]):
        return 0, 0

    belt_left_x = np.mean([point[0][0] for point in belt_left_contour])
    belt_right_x = np.mean([point[0][0] for point in belt_right_contour])
    material_left_x = np.mean([point[0][0] for point in material_left_contour])
    material_right_x = np.mean([point[0][0] for point in material_right_contour])

    belt_center = (belt_left_x + belt_right_x) / 2
    material_center = (material_left_x + material_right_x) / 2

    deviation = material_center - belt_center

    return deviation

def main():
    video_path = r'D:\python\pythonProject\pidaizhengchang2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频源，请检查路径或设备连接")
        return

    roi = np.array([[160, 300], [420, 300], [415, 225], [170, 225]], dtype=np.int32)
    captured_alert = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取完毕或出现错误")
            break

        edges_roi = process_frame(frame, roi)
        belt_left, belt_right, material_left, material_right, output_frame = draw_and_detect_contours(frame, edges_roi, roi)

        deviation = calculate_deviation(belt_left, belt_right, material_left, material_right)

        if abs(deviation) > 30:  # 偏离阈值，单位像素
            print(f"物料偏移量: {deviation:.2f} 像素")
            cv2.imwrite('offset_image.jpg', output_frame)
            captured_alert = True

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Contours in ROI", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
