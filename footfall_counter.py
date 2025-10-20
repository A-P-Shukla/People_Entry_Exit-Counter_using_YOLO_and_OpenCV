# =====================================================================================
# AI Assignment: Footfall Counter using Computer Vision
#
# Description:
# This script uses the YOLOv8 object detection model to detect and track people in a
# video stream. It counts entries (left-to-right) and exits (right-to-left) across
# a vertical line. The final code is structured into functions for clarity and
# maintainability, and includes detailed comments explaining each step.
#
# Author: Akhand Pratap Shukla
# Date: 21/10/2025
# =====================================================================================

# --- 1. Imports ---
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from scipy.spatial.distance import cdist

# --- 2. Global Configuration and Constants ---

# --- Video I/O ---
VIDEO_PATH = "people-walking.mp4"  # Path to the input video, or 0 for webcam
OUTPUT_VIDEO_PATH = "output_footfall_counter.avi"

# --- Model ---
MODEL_NAME = 'yolov8n.pt'  # Pre-trained YOLOv8 nano model

# --- Frame Configuration ---
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720  # Desired resolution for processing
LINE_X_POSITION = FRAME_WIDTH // 2  # Position of the vertical counting line

# --- Tracking Configuration ---
MAX_TRACKING_DISTANCE = 75  # The max distance (pixels) to associate a detection with an existing track.
MAX_FRAMES_TO_LOSE_TRACK = 20  # The number of consecutive frames a track can be lost before it's removed.
TRACK_HISTORY_LENGTH = 30  # The number of past points to store for a track's trajectory.

# --- Visualization ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
LINE_THICKNESS = 2
CIRCLE_RADIUS = 5

# --- Heatmap (Bonus Feature) ---
HEATMAP_RADIUS = 15  # The radius of each detection point on the heatmap for better visibility.


# --- 3. Core Processing Functions ---

def process_frame(frame, model, tracked_objects, track_history, next_track_id, in_count, out_count, counted_in_ids, counted_out_ids):
    """
    Processes a single video frame for person detection, tracking, and counting.

    Args:
        frame (np.ndarray): The input video frame.
        model (YOLO): The YOLO object detection model.
        tracked_objects (dict): Dictionary storing data of currently tracked objects.
        track_history (defaultdict): Dictionary storing the trajectory of each tracked object.
        next_track_id (int): The next available ID for a new track.
        in_count (int): The current count of people entering.
        out_count (int): The current count of people exiting.
        counted_in_ids (set): Set of track IDs that have been counted as 'IN'.
        counted_out_ids (set): Set of track IDs that have been counted as 'OUT'.

    Returns:
        tuple: A tuple containing the annotated frame and all updated state variables.
    """
    # --- Step 3.1: Object Detection ---
    # Use YOLO model to detect people (class_id 0) in the frame.
    results = model(frame, classes=[0], verbose=False)

    # --- Step 3.2: Extract Detections ---
    # Create a list of dictionaries, where each dictionary represents a detected person.
    current_detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centroid_x, centroid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        current_detections.append({
            "box": [x1, y1, x2, y2],
            "centroid": (centroid_x, centroid_y)
        })

    # --- Step 3.3: Tracking Logic ---
    # This section matches current detections with existing tracks.
    if not tracked_objects:
        # If there are no tracked objects, initialize a new track for each detection.
        for det in current_detections:
            tracked_objects[next_track_id] = {**det, "lost": 0}
            track_history[next_track_id].append(det["centroid"])
            next_track_id += 1
    else:
        # If there are existing tracks, try to match them with current detections.
        tracked_ids = list(tracked_objects.keys())
        prev_centroids = np.array([tracked_objects[tid]["centroid"] for tid in tracked_ids])
        current_centroids = np.array([det["centroid"] for det in current_detections])

        if len(current_centroids) > 0:
            # Calculate the distance between every previous centroid and every current centroid.
            dist_matrix = cdist(prev_centroids, current_centroids)
            used_cols = set() # To keep track of which detections have been matched.
            
            # Greedily match the closest pairs.
            for row, tid in enumerate(tracked_ids):
                if dist_matrix.shape[1] > 0:
                    best_match_idx = np.argmin(dist_matrix[row, :])
                    # If the distance is within the threshold, it's a match.
                    if dist_matrix[row, best_match_idx] < MAX_TRACKING_DISTANCE:
                        if best_match_idx not in used_cols:
                            # Update the track with the new detection info.
                            tracked_objects[tid] = {**current_detections[best_match_idx], "lost": 0}
                            track_history[tid].append(current_detections[best_match_idx]["centroid"])
                            used_cols.add(best_match_idx)

            # Any detections that were not matched are considered new tracks.
            unmatched_indices = set(range(len(current_detections))) - used_cols
            for idx in unmatched_indices:
                det = current_detections[idx]
                tracked_objects[next_track_id] = {**det, "lost": 0}
                track_history[next_track_id].append(det["centroid"])
                next_track_id += 1

    # --- Step 3.4: Handle Lost Tracks and Count Crossings ---
    lost_ids = []
    for track_id, data in tracked_objects.items():
        # If a track was not updated in the current frame, increment its 'lost' counter.
        if data['centroid'] not in [d['centroid'] for d in current_detections]:
            data["lost"] += 1
        
        # If a track is lost for too long, mark it for deletion.
        if data["lost"] > MAX_FRAMES_TO_LOSE_TRACK:
            lost_ids.append(track_id)
        else:
            # Check for line crossings using the track's history.
            history = track_history[track_id]
            if len(history) > 1:
                prev_x = history[-2][0]
                curr_x = history[-1][0]

                # Count IN (Left -> Right)
                if prev_x < LINE_X_POSITION and curr_x >= LINE_X_POSITION and track_id not in counted_in_ids:
                    in_count += 1
                    counted_in_ids.add(track_id)
                    counted_out_ids.discard(track_id) # Allow re-counting if they return.

                # Count OUT (Right -> Left)
                elif prev_x > LINE_X_POSITION and curr_x <= LINE_X_POSITION and track_id not in counted_out_ids:
                    out_count += 1
                    counted_out_ids.add(track_id)
                    counted_in_ids.discard(track_id) # Allow re-counting if they return.

    # Remove the lost tracks from memory.
    for tid in lost_ids:
        del tracked_objects[tid]
        del track_history[tid]

    # --- Step 3.5: Return all updated values ---
    return frame, tracked_objects, track_history, next_track_id, in_count, out_count, counted_in_ids, counted_out_ids

def draw_visualizations(frame, tracked_objects, track_history, in_count, out_count, heatmap):
    """Draws all visual elements onto the frame."""
    # Draw the vertical counting line.
    cv2.line(frame, (LINE_X_POSITION, 0), (LINE_X_POSITION, FRAME_HEIGHT), (255, 0, 0), LINE_THICKNESS)

    # Draw visuals for each tracked object.
    for track_id, data in tracked_objects.items():
        box, centroid = data["box"], data["centroid"]
        x1, y1, x2, y2 = box
        
        # Update heatmap with a circle for better visibility.
        cv2.circle(heatmap, centroid, HEATMAP_RADIUS, 1, -1)
        
        # Draw bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
        # Draw centroid.
        cv2.circle(frame, centroid, CIRCLE_RADIUS, (0, 0, 255), -1)
        # Draw trajectory line.
        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 0), thickness=2)
        # Draw track ID.
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), FONT, 0.6, (255, 255, 255), FONT_THICKNESS)

    # Display IN/OUT counts on the frame.
    cv2.putText(frame, f"IN (Left->Right): {in_count}", (30, 60), FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, f"OUT (Right->Left): {out_count}", (30, 110), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)
    
    return frame

def generate_heatmap_visualization(heatmap, background_frame):
    """Generates and saves a visual heatmap image."""
    if background_frame is None:
        print("Could not generate heatmap because no frames were processed.")
        return

    # Apply a Gaussian blur to smooth the heatmap trails.
    heatmap_blurred = cv2.GaussianBlur(heatmap, (11, 11), 0)
    
    # Normalize the heatmap to a 0-255 range.
    heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply a colormap to create the classic heatmap look.
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the last frame of the video.
    superimposed_img = cv2.addWeighted(heatmap_color, 0.6, background_frame, 0.4, 0)
    
    # Display and save the final image.
    cv2.imshow('Footfall Heatmap', superimposed_img)
    cv2.imwrite("footfall_heatmap_final.png", superimposed_img)
    print("Heatmap saved as footfall_heatmap_final.png. Press any key to exit.")
    
    cv2.waitKey(0)


# --- 4. Main Execution Block ---
def main():
    """Main function to run the footfall counter."""
    
    # --- Initialization ---
    # Initialize variables to store state between frames.
    last_good_frame = None
    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    next_track_id = 0
    tracked_objects = {}
    counted_in_ids = set()
    counted_out_ids = set()
    in_count = 0
    out_count = 0
    heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

    # Load the YOLO model.
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)
    print("Model loaded successfully.")

    # Open the video source.
    print("Opening video source...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video source at {VIDEO_PATH}")
        return

    # --- Setup Video I/O and Speed Correction ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default FPS for webcams
    frame_delay_ms = int(1000 / fps)
    print(f"Original video FPS: {fps:.2f}, calculated frame delay: {frame_delay_ms}ms")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # --- Main Processing Loop ---
    print("Processing video... Press 'q' to stop.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        last_good_frame = frame.copy() # Store the last good frame for the heatmap.

        # Process the current frame to update tracking and counts.
        _, tracked_objects, track_history, next_track_id, in_count, out_count, counted_in_ids, counted_out_ids = process_frame(
            frame, model, tracked_objects, track_history, next_track_id, in_count, out_count, counted_in_ids, counted_out_ids
        )

        # Draw all visualizations on the frame.
        annotated_frame = draw_visualizations(frame, tracked_objects, track_history, in_count, out_count, heatmap)

        # Display the processed frame and write it to the output file.
        out_writer.write(annotated_frame)
        cv2.imshow("Footfall Counter", annotated_frame)

        # Wait for the calculated delay and check for the 'q' key to exit.
        if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
            break

    # --- Cleanup and Final Output ---
    print("\nCleaning up and generating heatmap...")
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    
    # Generate and display the final heatmap.
    generate_heatmap_visualization(heatmap, last_good_frame)

if __name__ == '__main__':
    main()