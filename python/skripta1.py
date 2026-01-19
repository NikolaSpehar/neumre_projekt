import json
import os
import numpy as np
from pathlib import Path

def convert_openpose_to_blender_2d(input_dir, output_file="animation_data.json", frame_rate=30):
    JOINT_NAMES = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
        "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar", "LBigToe",
        "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
    ]
    
    SIMPLIFIED_JOINTS = {
        "Head": 0,
        "Neck": 1,
        "RShoulder": 2,
        "RElbow": 3,
        "RWrist": 4,
        "LShoulder": 5,
        "LElbow": 6,
        "LWrist": 7,
        "MidHip": 8,
        "RHip": 9,
        "RKnee": 10,
        "RAnkle": 11,
        "LHip": 12,
        "LKnee": 13,
        "LAnkle": 14,
        "REye": 15,
        "LEye": 16,
        "REar": 17,
        "LEar": 18
    }
    
    BONE_CONNECTIONS = [
        # Tijelo
        ("Neck", "Head"),
        ("Neck", "RShoulder"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("Neck", "LShoulder"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("Neck", "MidHip"),
        ("MidHip", "RHip"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("MidHip", "LHip"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
        
        # Glava - dodajemo linije za lice
        ("Head", "REye"),
        ("Head", "LEye"),
        ("REye", "REar"),
        ("LEye", "LEar"),
        ("REye", "LEye")
    ]
    
    print(f"Čitam OpenPose JSON datoteke iz {input_dir}...")
    
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"Nema JSON datoteka u {input_dir}!")
        return
    
    print(f"Pronađeno {len(json_files)} frameova")
    
    frames_data = []
    
    for i, json_file in enumerate(json_files):
        frame_path = os.path.join(input_dir, json_file)
        
        try:
            with open(frame_path, 'r') as f:
                data = json.load(f)
                
            frame_keypoints = {}
            has_face_data = False
            
            if 'people' in data and len(data['people']) > 0:
                person = data['people'][0]
                keypoints_array = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
                
                for joint_name, idx in SIMPLIFIED_JOINTS.items():
                    x, y, confidence = keypoints_array[idx]
                    
                    normalized_x = (x - 320) / 500
                    normalized_y = (480 - y) / 500  
                    
                    if confidence > 0.1:
                        frame_keypoints[joint_name] = {
                            "x": float(normalized_x),
                            "y": float(normalized_y),
                            "confidence": float(confidence),
                            "type": "body"
                        }
                    else:
                        if frames_data and joint_name in frames_data[-1]["keypoints"]:
                            frame_keypoints[joint_name] = frames_data[-1]["keypoints"][joint_name]
                        else:
                            frame_keypoints[joint_name] = {
                                "x": 0.0,
                                "y": 0.0,
                                "confidence": 0.0,
                                "type": "body"
                            }
                
                if 'face_keypoints_2d' in person and person['face_keypoints_2d']:
                    has_face_data = True
                    face_points = np.array(person['face_keypoints_2d']).reshape(-1, 3)
                    
                    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]  # Kontura lica
                    
                    for j, idx in enumerate(face_indices):
                        if idx < len(face_points):
                            x, y, confidence = face_points[idx]
                            normalized_x = (x - 320) / 500
                            normalized_y = (480 - y) / 500
                            
                            if confidence > 0.1:
                                frame_keypoints[f"Face_{j}"] = {
                                    "x": float(normalized_x),
                                    "y": float(normalized_y),
                                    "confidence": float(confidence),
                                    "type": "face"
                                }
                    
                    inner_face_indices = [27, 28, 29, 30, 31, 32, 33, 34, 35] 
                    for j, idx in enumerate(inner_face_indices):
                        if idx < len(face_points):
                            x, y, confidence = face_points[idx]
                            normalized_x = (x - 320) / 500
                            normalized_y = (480 - y) / 500
                            
                            if confidence > 0.1:
                                frame_keypoints[f"Face_nose_{j}"] = {
                                    "x": float(normalized_x),
                                    "y": float(normalized_y),
                                    "confidence": float(confidence),
                                    "type": "face"
                                }
            else:
                if frames_data:
                    frame_keypoints = frames_data[-1]["keypoints"].copy()
                else:
                    frame_keypoints = {joint: {"x": 0.0, "y": 0.0, "confidence": 0.0, "type": "body"} 
                                     for joint in SIMPLIFIED_JOINTS.keys()}
            
            frames_data.append({
                "frame": i,
                "keypoints": frame_keypoints,
                "has_face": has_face_data
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Obradio {i + 1}/{len(json_files)} frameova")
                if has_face_data:
                    print(f"    (ima podatke o licu)")
                
        except Exception as e:
            print(f"Greška pri čitanju {json_file}: {e}")
            if frames_data:
                frames_data.append(frames_data[-1].copy())
            else:
                frames_data.append({
                    "frame": i,
                    "keypoints": {joint: {"x": 0.0, "y": 0.0, "confidence": 0.0, "type": "body"} 
                                 for joint in SIMPLIFIED_JOINTS.keys()},
                    "has_face": False
                })
    
    face_connections = []
    face_point_count = 27  
    
    for i in range(face_point_count - 1):
        face_connections.append((f"Face_{i}", f"Face_{i + 1}"))

    face_connections.append((f"Face_{face_point_count - 1}", "Face_0"))
    
    nose_connections = []
    for i in range(8):  # Za nos
        nose_connections.append((f"Face_nose_{i}", f"Face_nose_{i + 1}"))
    
    all_connections = BONE_CONNECTIONS + face_connections + nose_connections
    
    output_data = {
        "metadata": {
            "total_frames": len(frames_data),
            "fps": frame_rate,
            "joints": list(SIMPLIFIED_JOINTS.keys()),
            "bone_connections": all_connections,
            "has_face_data": any(frame["has_face"] for frame in frames_data)
        },
        "frames": frames_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Podaci spremljeni u: {output_file}")
    print(f"Ukupno frameova: {len(frames_data)}")
    
    face_frames = sum(1 for frame in frames_data if frame["has_face"])
    if face_frames > 0:
        print(f"Frameova s podacima o licu: {face_frames}")
    
if __name__ == "__main__":
    INPUT_DIR = "openpose_json/video10"
    OUTPUT_FILE = "opoenpose_video10.json"
    

    convert_openpose_to_blender_2d(INPUT_DIR, OUTPUT_FILE, frame_rate=30)
