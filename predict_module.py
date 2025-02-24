import cv2
import os
from ultralytics import YOLO

# Load your YOLOv8 models
gtsdb_model = YOLO('gtsdb.pt')  # GTSDB model
gtsrb_model = YOLO('gtsrb.pt')  # GTSRB model

def predict_on_frame(input_path, output_path='output_result', mode='gtsdb', conf_threshold=0.6, debug=False):
    """
    Predicts on a given input (image or video).
    
    :param input_path: Path to the input image or video.
    :param output_path: Path to save the output result (without extension).
    :param mode: Which model(s) to use for labeling. Options: 'gtsdb', 'gtsrb', 'both'.
    :param conf_threshold: Minimum confidence threshold to filter predictions.
    :param debug: If True, prints debug logs.
    :return: Dictionary containing detection results
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    detection_results = {
        'has_detections': False,
        'detections': []
    }

    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Process as an image
        img = cv2.imread(input_path)
        results = gtsdb_model(img)  # Detect with GTSDB model
        annotated_img = img.copy()  # Create a copy for annotation
        
        has_gtsdb_detections = False
        for result in results:
            # Extracting bounding boxes and labels from GTSDB model
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            labels = [result.names[idx] for idx in class_indices]

            valid_detections = [(box, score, label) for box, score, label in zip(boxes, scores, labels) if score >= conf_threshold]
            
            if valid_detections:
                has_gtsdb_detections = True
                if debug:
                    print("\n--- GTSDB Predictions ---")
                
                for box, score, label in valid_detections:
                    detection_results['has_detections'] = True
                    x1, y1, x2, y2 = map(int, box)
                    if debug:
                        print(f"Box: {box}, Label: {label}, Confidence: {score}")

                    cropped_sign = img[y1:y2, x1:x2]  # Crop the traffic sign
                    predicted_label = label

                    if mode == 'gtsrb' or mode == 'both':
                        # Use the GTSRB model to classify the cropped traffic sign
                        gtsrb_results = gtsrb_model(cropped_sign)
                        if debug:
                            print("\n--- GTSRB Predictions for Cropped Sign ---")
                        
                        g_labels = []
                        if isinstance(gtsrb_results, list):
                            for g_result in gtsrb_results:
                                if len(g_result.boxes) > 0:
                                    g_class_indices = g_result.boxes.cls.cpu().numpy().astype(int)
                                    g_labels.extend([g_result.names[idx] for idx in g_class_indices])
                        else:
                            if len(gtsrb_results.boxes) > 0:
                                g_class_indices = gtsrb_results.boxes.cls.cpu().numpy().astype(int)
                                g_labels = [gtsrb_results.names[idx] for idx in g_class_indices]

                        # If GTSRB gives a valid label, use it
                        if g_labels:
                            predicted_label = g_labels[0]

                    # Store detection information
                    detection_results['detections'].append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'label': predicted_label,
                        'confidence': float(score)
                    })

                    # Draw bounding box and label
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_img, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # If no GTSDB detections or in GTSRB/both mode, run GTSRB on whole image
        if (not has_gtsdb_detections and mode != 'gtsdb') or mode == 'gtsrb' or mode == 'both':
            if debug:
                print("\n--- GTSRB Predictions on Full Image ---")
            
            gtsrb_results = gtsrb_model(img)
            
            # Process GTSRB results on full image
            if isinstance(gtsrb_results, list):
                for g_result in gtsrb_results:
                    if len(g_result.boxes) > 0:
                        g_boxes = g_result.boxes.xyxy.cpu().numpy()
                        g_scores = g_result.boxes.conf.cpu().numpy()
                        g_class_indices = g_result.boxes.cls.cpu().numpy().astype(int)
                        
                        for j, (g_box, g_score) in enumerate(zip(g_boxes, g_scores)):
                            if g_score >= conf_threshold:
                                detection_results['has_detections'] = True
                                x1, y1, x2, y2 = map(int, g_box)
                                g_label = g_result.names[g_class_indices[j]]
                                
                                detection_results['detections'].append({
                                    'box': [x1, y1, x2, y2],
                                    'label': g_label,
                                    'confidence': float(g_score)
                                })
                                
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for GTSRB
                                cv2.putText(annotated_img, g_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save the annotated image
        cv2.imwrite(output_path + '.jpg', annotated_img)
        if debug:
            print(f"\nPrediction saved as {output_path}.jpg")

    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process as a video
        cap = cv2.VideoCapture(input_path)
        if os.name == 'nt':  # Windows
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:  # Linux/Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare the video writer for output
        out = cv2.VideoWriter(output_path + '.mp4', fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = gtsdb_model(frame)  # Detect with GTSDB model
            annotated_frame = frame.copy()  # Create a copy for annotation
            has_gtsdb_detections = False

            if debug:
                print(f"\n--- Frame {frame_count} ---")
            
            for result in results:
                # Extracting bounding boxes and labels from GTSDB model
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_indices = result.boxes.cls.cpu().numpy().astype(int)
                labels = [result.names[idx] for idx in class_indices]

                valid_detections = [(box, score, label) for box, score, label in zip(boxes, scores, labels) if score >= conf_threshold]
                
                if valid_detections:
                    has_gtsdb_detections = True
                    if debug:
                        print("--- GTSDB Predictions ---")

                    for box, score, label in valid_detections:
                        x1, y1, x2, y2 = map(int, box)
                        if debug:
                            print(f"Box: {box}, Label: {label}, Confidence: {score}")

                        cropped_sign = frame[y1:y2, x1:x2]  # Crop the traffic sign
                        predicted_label = label

                        if mode == 'gtsrb' or mode == 'both':
                            # Use the GTSRB model to classify the cropped traffic sign
                            gtsrb_results = gtsrb_model(cropped_sign)
                            
                            if debug:
                                print("--- GTSRB Predictions for Cropped Sign ---")
                            g_labels = []
                            if isinstance(gtsrb_results, list):
                                for g_result in gtsrb_results:
                                    if len(g_result.boxes) > 0:
                                        g_class_indices = g_result.boxes.cls.cpu().numpy().astype(int)
                                        g_labels.extend([g_result.names[idx] for idx in g_class_indices])
                            else:
                                if len(gtsrb_results.boxes) > 0:
                                    g_class_indices = gtsrb_results.boxes.cls.cpu().numpy().astype(int)
                                    g_labels = [gtsrb_results.names[idx] for idx in g_class_indices]

                            # If GTSRB gives a valid label, use it
                            if g_labels:
                                predicted_label = g_labels[0]

                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If no GTSDB detections or in GTSRB/both mode, run GTSRB on whole frame
            if (not has_gtsdb_detections and mode != 'gtsdb') or mode == 'gtsrb' or mode == 'both':
                if debug:
                    print("\n--- GTSRB Predictions on Full Frame ---")
                
                gtsrb_results = gtsrb_model(frame)
                
                # Process GTSRB results on full frame
                if isinstance(gtsrb_results, list):
                    for g_result in gtsrb_results:
                        if len(g_result.boxes) > 0:
                            g_boxes = g_result.boxes.xyxy.cpu().numpy()
                            g_scores = g_result.boxes.conf.cpu().numpy()
                            g_class_indices = g_result.boxes.cls.cpu().numpy().astype(int)
                            
                            for j, (g_box, g_score) in enumerate(zip(g_boxes, g_scores)):
                                if g_score >= conf_threshold:
                                    x1, y1, x2, y2 = map(int, g_box)
                                    g_label = g_result.names[g_class_indices[j]]
                                    
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for GTSRB
                                    cv2.putText(annotated_frame, g_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)
            frame_count += 1

        # Release the video capture and writer objects
        cap.release()
        out.release()
        if debug:
            print(f"\nPrediction saved as {output_path}.mp4")

    else:
        print("Unsupported file type. Please provide an image or video file.")

    return detection_results

