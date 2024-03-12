import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# load known faces
himanshu_images = face_recognition.load_image_file("faces/himanshu.jpg")
himanshu_encoding = face_recognition.face_encodings(himanshu_images)[0]

rohan_images = face_recognition.load_image_file("faces/rohan.png")
rohan_encoding = face_recognition.face_encodings(rohan_images)[0]

known_faces_encoding = [himanshu_encoding, rohan_encoding]
known_face_name = ["himanshu", "rohan"]
# list of expected student
students = known_face_name.copy()

face_locations = []
face_encodings = []

# get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognizing faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_faces_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_name[best_match_index]

            # add the text if a person is present
            if name in known_face_name:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_text = (face_location[3] * 4, face_location[0] * 4)
                font_scale = 1
                font_color = (255, 255, 255)
                thickness = 2
                line_type = 2
                cv2.putText(frame, name + " Present", bottom_left_corner_text, font, font_scale, font_color, thickness,
                            line_type)
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow(([name, current_time]))

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
