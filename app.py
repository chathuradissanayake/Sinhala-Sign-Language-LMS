from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import json
import asyncio
from datetime import datetime
from typing import List, Dict
import random

app = FastAPI(title="Sinhala Sign Language LMS API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model Configuration ----------
MODEL_PATH = "hand_gesture_model.h5"
ACTIONS_PATH = "actions.npy"
SEQUENCE_LENGTH = 60
KEYPOINT_LENGTH = 126
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW = 5

# Sinhala letter mapping
LETTER_MAPPING = {
    'a_': 'අ',
    'aa_': 'ආ',
    'aae_': 'ඇ',
    'ae_': 'ඈ',
    'e_': 'එ',
    'ee_': 'ඒ',
    'i_': 'ඉ',
    'ii_': 'ඊ',
    'u_': 'උ',
    'uu_': 'ඌ',
    'unknown': '?'
}

# Load model and actions
try:
    model = load_model(MODEL_PATH)
    actions = np.load(ACTIONS_PATH)
    print(f"✅ Model loaded with actions: {list(actions)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    actions = []

# MediaPipe setup
mp_hands = mp.solutions.hands


class GestureRecognizer:
    """Handles real-time gesture recognition"""

    def __init__(self):
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def extract_hand_keypoints(self, results):
        """Extract hand landmarks"""
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z]
                                  for lm in hand_landmarks.landmark]).flatten()

                if hand_label == "Left":
                    lh = coords
                else:
                    rh = coords

        return np.concatenate([lh, rh])

    def normalize_keypoints(self, keypoints):
        """Normalize relative to wrist"""
        if not np.any(keypoints):
            return keypoints

        keypoints_reshaped = keypoints.reshape(2, 21, 3)

        for hand_idx in range(2):
            hand = keypoints_reshaped[hand_idx]
            if np.any(hand):
                wrist = hand[0]
                hand = hand - wrist
                max_dist = np.max(np.abs(hand))
                if max_dist > 0:
                    hand = hand / max_dist
                keypoints_reshaped[hand_idx] = hand

        return keypoints_reshaped.flatten()

    def process_frame(self, frame_data):
        """Process a single frame and return prediction"""
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # MediaPipe detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Extract and normalize keypoints
            keypoints = self.extract_hand_keypoints(results)
            keypoints = self.normalize_keypoints(keypoints)

            # Add to sequence
            self.sequence.append(keypoints)

            # Make prediction if buffer is full
            if len(self.sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(
                    np.array(self.sequence), axis=0).astype(np.float32)
                probabilities = model.predict(input_data, verbose=0)[0]
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])

                if confidence > CONFIDENCE_THRESHOLD:
                    self.prediction_history.append(pred_idx)

                # Smooth predictions
                if len(self.prediction_history) >= SMOOTHING_WINDOW // 2:
                    from collections import Counter
                    recent = list(self.prediction_history)[-SMOOTHING_WINDOW:]
                    most_common = Counter(recent).most_common(1)[0]

                    if most_common[1] >= SMOOTHING_WINDOW * 0.5:
                        predicted_action = actions[most_common[0]]
                        return {
                            'predicted': predicted_action,
                            'confidence': confidence,
                            'probabilities': {actions[i]: float(probabilities[i]) for i in range(len(actions))},
                            'buffer_filled': True
                        }

            return {
                'predicted': None,
                'confidence': 0.0,
                'probabilities': {},
                'buffer_filled': len(self.sequence) == SEQUENCE_LENGTH
            }

        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                'predicted': None,
                'confidence': 0.0,
                'probabilities': {},
                'buffer_filled': False,
                'error': str(e)
            }

    def reset(self):
        """Reset sequence buffer"""
        self.sequence.clear()
        self.prediction_history.clear()


# Store active sessions
active_sessions: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "running",
        "service": "Sinhala Sign Language LMS",
        "model_loaded": model is not None,
        "available_actions": list(actions)
    }


@app.get("/api/letters")
async def get_letters():
    """Get all available Sinhala letters"""
    available_letters = [
        {
            'id': action,
            'sinhala': LETTER_MAPPING.get(action, action),
            'english': action
        }
        for action in actions if action != 'unknown'
    ]
    return {
        "letters": available_letters,
        "total": len(available_letters)
    }


@app.post("/api/practice/start")
async def start_practice(request: dict):
    """Start a new practice session"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    letters = request.get('letters', [])

    # If no letters specified, use all available
    if not letters:
        letters = [action for action in actions if action != 'unknown']

    # Validate letters
    invalid = [l for l in letters if l not in actions]
    if invalid:
        raise HTTPException(
            status_code=400, detail=f"Invalid letters: {invalid}")

    # Create session
    session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    active_sessions[session_id] = {
        'mode': 'practice',
        'letters': letters,
        'current_index': 0,
        'completed': [],
        'failed_attempts': {},
        'start_time': datetime.now().isoformat(),
        'recognizer': GestureRecognizer()
    }

    print(
        f"✅ Created practice session {session_id} with {len(letters)} letters")

    return {
        "session_id": session_id,
        "total_letters": len(letters),
        "letters": [
            {
                'id': l,
                'sinhala': LETTER_MAPPING.get(l, l),
                'english': l
            }
            for l in letters
        ]
    }


@app.post("/api/exam/start")
async def start_exam():
    """Start a new exam session with 5 random letters"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Get 5 random letters (excluding 'unknown')
    available_letters = [action for action in actions if action != 'unknown']
    if len(available_letters) < 5:
        raise HTTPException(
            status_code=400, detail="Not enough letters available for exam")

    letters = random.sample(available_letters, 5)

    # Create session
    session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    active_sessions[session_id] = {
        'mode': 'exam',
        'letters': letters,
        'current_index': 0,
        'results': [],  # Store results for each letter
        'start_time': datetime.now().isoformat(),
        'recognizer': GestureRecognizer()
    }

    print(f"✅ Created exam session {session_id} with letters: {letters}")

    return {
        "session_id": session_id,
        "total_letters": len(letters),
        "letters": [
            {
                'id': l,
                'sinhala': LETTER_MAPPING.get(l, l),
                'english': l
            }
            for l in letters
        ]
    }


@app.websocket("/ws/practice/{session_id}")
async def websocket_practice(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for practice mode"""
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.send_json({
            'type': 'error',
            'message': f'Invalid session ID: {session_id}. Please start a new practice session.'
        })
        await websocket.close()
        return

    session = active_sessions[session_id]
    if session['mode'] != 'practice':
        await websocket.send_json({
            'type': 'error',
            'message': 'Invalid session mode'
        })
        await websocket.close()
        return

    recognizer = session['recognizer']
    print(f"✅ WebSocket connected for practice session: {session_id}")

    try:
        # Send initial letter
        current_letter = session['letters'][session['current_index']]
        await websocket.send_json({
            'type': 'next_letter',
            'letter': {
                'id': current_letter,
                'sinhala': LETTER_MAPPING.get(current_letter, current_letter),
                'english': current_letter
            },
            'progress': {
                'current': session['current_index'] + 1,
                'total': len(session['letters'])
            }
        })

        while True:
            data = await websocket.receive_json()

            if data.get('type') == 'frame':
                result = recognizer.process_frame(data['frame'])

                await websocket.send_json({
                    'type': 'prediction',
                    'predicted': result['predicted'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'buffer_filled': result['buffer_filled']
                })

                # Check if correct gesture
                if result['predicted'] == current_letter and result['confidence'] > CONFIDENCE_THRESHOLD:
                    session['completed'].append({
                        'letter': current_letter,
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().isoformat()
                    })

                    await websocket.send_json({
                        'type': 'success',
                        'letter': {
                            'id': current_letter,
                            'sinhala': LETTER_MAPPING.get(current_letter, current_letter)
                        },
                        'confidence': result['confidence']
                    })

                    session['current_index'] += 1
                    recognizer.reset()
                    await asyncio.sleep(1)

                    if session['current_index'] < len(session['letters']):
                        current_letter = session['letters'][session['current_index']]
                        await websocket.send_json({
                            'type': 'next_letter',
                            'letter': {
                                'id': current_letter,
                                'sinhala': LETTER_MAPPING.get(current_letter, current_letter),
                                'english': current_letter
                            },
                            'progress': {
                                'current': session['current_index'] + 1,
                                'total': len(session['letters'])
                            }
                        })
                    else:
                        total = len(session['letters'])
                        completed = len(session['completed'])
                        success_rate = (completed / total) * 100

                        await websocket.send_json({
                            'type': 'practice_complete',
                            'results': {
                                'total': total,
                                'completed': completed,
                                'success_rate': success_rate,
                                'details': session['completed']
                            }
                        })
                        break

            elif data.get('type') == 'reset':
                recognizer.reset()
                await websocket.send_json({
                    'type': 'reset_complete'
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]


@app.websocket("/ws/exam/{session_id}")
async def websocket_exam(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for exam mode"""
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.send_json({
            'type': 'error',
            'message': f'Invalid session ID: {session_id}. Please start a new exam.'
        })
        await websocket.close()
        return

    session = active_sessions[session_id]
    if session['mode'] != 'exam':
        await websocket.send_json({
            'type': 'error',
            'message': 'Invalid session mode'
        })
        await websocket.close()
        return

    recognizer = session['recognizer']
    print(f"✅ WebSocket connected for exam session: {session_id}")

    try:
        # Send initial letter
        current_letter = session['letters'][session['current_index']]
        await websocket.send_json({
            'type': 'next_letter',
            'letter': {
                'id': current_letter,
                'sinhala': LETTER_MAPPING.get(current_letter, current_letter),
                'english': current_letter
            },
            'progress': {
                'current': session['current_index'] + 1,
                'total': len(session['letters'])
            },
            'time_limit': 10
        })

        # Track predictions during the 10 seconds
        letter_start_time = asyncio.get_event_loop().time()
        best_match = None
        best_confidence = 0.0

        while True:
            data = await websocket.receive_json()

            if data.get('type') == 'frame':
                result = recognizer.process_frame(data['frame'])

                # Send prediction for display only
                await websocket.send_json({
                    'type': 'prediction',
                    'predicted': result['predicted'],
                    'confidence': result['confidence'],
                    'buffer_filled': result['buffer_filled']
                })

                # Track best match during time window
                if result['predicted'] == current_letter and result['confidence'] > best_confidence:
                    best_match = result['predicted']
                    best_confidence = result['confidence']

                # Check if 10 seconds elapsed
                elapsed = asyncio.get_event_loop().time() - letter_start_time
                if elapsed >= 10:
                    # Save result
                    is_correct = best_match == current_letter and best_confidence > CONFIDENCE_THRESHOLD
                    session['results'].append({
                        'letter': current_letter,
                        'correct': is_correct,
                        'confidence': best_confidence,
                        'predicted': best_match
                    })

                    # Move to next letter
                    session['current_index'] += 1
                    recognizer.reset()

                    if session['current_index'] < len(session['letters']):
                        current_letter = session['letters'][session['current_index']]
                        letter_start_time = asyncio.get_event_loop().time()
                        best_match = None
                        best_confidence = 0.0

                        await websocket.send_json({
                            'type': 'next_letter',
                            'letter': {
                                'id': current_letter,
                                'sinhala': LETTER_MAPPING.get(current_letter, current_letter),
                                'english': current_letter
                            },
                            'progress': {
                                'current': session['current_index'] + 1,
                                'total': len(session['letters'])
                            },
                            'time_limit': 10
                        })
                    else:
                        # Exam complete
                        total = len(session['letters'])
                        correct = sum(
                            1 for r in session['results'] if r['correct'])
                        success_rate = (correct / total) * 100

                        await websocket.send_json({
                            'type': 'exam_complete',
                            'results': {
                                'total': total,
                                'correct': correct,
                                'success_rate': success_rate,
                                'details': session['results']
                            }
                        })
                        break

            elif data.get('type') == 'reset':
                recognizer.reset()
                await websocket.send_json({
                    'type': 'reset_complete'
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
