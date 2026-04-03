import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

video = cv2.VideoCapture(0) 

current_state = "inattentive"

# UPGRADE: refine_landmarks=True unlocks the Iris tracking!
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True) as face_mesh:

    while True:
        ret, image = video.read()
        if not ret: break
        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = face_mesh.process(rgb_image)
        
        new_state = "inattentive"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Head tracking points
                nose_x = face_landmarks.landmark[1].x
                left_x = face_landmarks.landmark[234].x
                right_x = face_landmarks.landmark[454].x
                
                # Iris tracking points (468 = Right eye, 473 = Left eye)
                # (Note: "Right" means user's right side, which is left on the mirrored screen)
                right_iris = face_landmarks.landmark[468]
                left_iris = face_landmarks.landmark[473]
                
                face_width = right_x - left_x
                
                if face_width > 0: 
                    nose_center_ratio = (nose_x - left_x) / face_width
                    
                    # THE HUMANIZED THRESHOLD:
                    # 0.35 to 0.65 allows for casual glances and relaxed posture.
                    # face_width > 0.25 keeps the 60cm distance rule.
                    if 0.35 < nose_center_ratio < 0.65 and face_width > 0.25:
                        new_state = "attentive"
                
                # Draw the standard wireframe
                mp_drawing.draw_landmarks(
                    image=image, 
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
                
                # --- DRAW THE EYE TRACKERS ---
                # Convert normalized coordinates back to actual screen pixels
                h, w, _ = image.shape
                r_cx, r_cy = int(right_iris.x * w), int(right_iris.y * h)
                l_cx, l_cy = int(left_iris.x * w), int(left_iris.y * h)
                
                # Draw bright green circles on the pupils
                cv2.circle(image, (r_cx, r_cy), 3, (0, 255, 0), -1)
                cv2.circle(image, (l_cx, l_cy), 3, (0, 255, 0), -1)
        
        # State machine logic
        if new_state != current_state:
            print(f"User is now: {new_state.upper()}")
            current_state = new_state
                
        cv2.imshow("Picobot Eye Tracking", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()