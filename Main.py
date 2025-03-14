import os
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
import time
from PIL import Image
import io

# Set the API key via environment variable first, then configure
os.environ["GOOGLE_API_KEY"] = "AIzaSyA6qNHA5pIYUB0sRXAmROBAs52fmxZpjRk"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Try to initialize the model after configuration
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Model initialized successfully")
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None

# Initialize the HandDetector class
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Constants
DRAW_COLOR = (255, 0, 255)  # Purple color for drawing
DRAW_THICKNESS = 10
ERASER_COLOR = (0, 0, 0)  # Black color for eraser
ERASER_THICKNESS = 40

# Status variables
drawing_mode = True  # True for drawing, False for erasing
last_ai_trigger_time = 0  # To prevent multiple rapid AI triggers
ai_cooldown = 3  # Cooldown in seconds between AI triggers
processing_ai = False  # Flag to indicate AI is processing
has_drawing = False  # Flag to track if there's a drawing on the canvas
ai_response = "No response yet"  # Store the AI response


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList, img
    else:
        return None, None, img


def draw(img, info, prev_pos, canvas):
    global drawing_mode, has_drawing
    fingers, lmList = info
    current_pos = None
    drawing_occurred = False

    # Drawing mode - index finger up, others down
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(map(int, lmList[8][0:2]))
        if prev_pos is not None:
            color = DRAW_COLOR if drawing_mode else ERASER_COLOR
            thickness = DRAW_THICKNESS if drawing_mode else ERASER_THICKNESS
            cv2.line(canvas, current_pos, prev_pos, color, thickness)
            if drawing_mode:
                drawing_occurred = True

    # Switch between drawing and erasing - thumb and little finger up
    elif fingers == [1, 0, 0, 0, 1]:
        drawing_mode = not drawing_mode
        print(f"Mode switched to {'Drawing' if drawing_mode else 'Erasing'}")
        time.sleep(0.5)

    # Clear canvas - all fingers up
    elif fingers == [1, 1, 1, 1, 1]:
        canvas[:] = 0
        has_drawing = False
        print("Canvas cleared")
        time.sleep(0.5)

    if drawing_occurred:
        has_drawing = True

    return current_pos, canvas


def prepare_image_for_ai(canvas):
    # Create a white background image
    white_bg = np.ones((canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8) * 255

    # Copy non-black pixels from the canvas to the white background
    mask = np.any(canvas > 10, axis=2)
    white_bg[mask] = canvas[mask]

    # Convert to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
    return img_pil


def sendToAI(model, canvas, fingers):
    global last_ai_trigger_time, processing_ai, ai_response
    current_time = time.time()

    # Trigger AI with all fingers except pinky up
    if fingers == [1, 1, 1, 1, 0] and (current_time - last_ai_trigger_time) > ai_cooldown and not processing_ai:
        print("AI trigger gesture detected!")
        last_ai_trigger_time = current_time

        # Check if model is initialized
        if model is None:
            ai_response = "AI model not initialized. Cannot generate response."
            print(ai_response)
            return

        # Check if canvas has any drawing
        if has_drawing:
            processing_ai = True
            ai_response = "Generating AI response..."
            print(ai_response)

            try:
                # Prepare the image for AI
                img_for_ai = prepare_image_for_ai(canvas)

                # Send image to Gemini with a prompt
                response = model.generate_content(["Solve this math problem", img_for_ai])
                ai_response = response.text
                print("AI Response:", ai_response)

            except Exception as e:
                error_msg = f"Error generating AI content: {e}"
                ai_response = error_msg
                print(error_msg)
            finally:
                processing_ai = False
        else:
            ai_response = "Canvas is empty. Draw something first!"
            print(ai_response)


def main():
    # Initialize the webcam to capture video
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_pos = None
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Text setup for instruction display
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    line_type = 2

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image from camera")
                break

            img = cv2.flip(img, 1)

            # Get hand information and update image
            fingers, lmList, img = getHandInfo(img)

            # Display instructions on screen
            instructions = [
                "INDEX FINGER UP: Draw",
                "THUMB + PINKY UP: Switch Draw/Erase",
                "ALL FINGERS UP: Clear Canvas",
                "ALL EXCEPT PINKY UP: Get AI Feedback"
            ]

            for i, instruction in enumerate(instructions):
                y_position = 30 + i * 25
                cv2.putText(img, instruction, (10, y_position), font, font_scale, font_color, line_type)

            # Display current mode
            mode_text = f"Mode: {'Drawing' if drawing_mode else 'Erasing'}"
            cv2.putText(img, mode_text, (10, 130), font, font_scale, (0, 255, 0), line_type)

            # Display drawing status
            status_text = f"Drawing Status: {'Content present' if has_drawing else 'Canvas empty'}"
            cv2.putText(img, status_text, (10, 160), font, font_scale, (0, 255, 255), line_type)

            # Display AI Response (truncated if too long)
            max_length = 50  # Maximum length for a single line
            if ai_response:
                # Split into multiple lines if needed
                response_lines = [ai_response[i:i + max_length] for i in range(0, len(ai_response), max_length)]
                for j, line in enumerate(response_lines[:4]):  # Show at most 4 lines
                    cv2.putText(img, line, (10, 190 + j * 25), font, font_scale, (255, 165, 0), line_type)

                # Show indicator if more text is available
                if len(response_lines) > 4:
                    cv2.putText(img, "...", (10, 190 + 4 * 25), font, font_scale, (255, 165, 0), line_type)

            # Process hand gestures if hand is detected
            if lmList is not None:
                info = (fingers, lmList)
                current_pos, canvas = draw(img, info, prev_pos, canvas)

                # Check for AI trigger gesture
                if fingers == [1, 1, 1, 1, 0]:
                    cv2.putText(img, "AI TRIGGER DETECTED!", (10, 290), font, font_scale, (0, 0, 255), line_type)
                    sendToAI(model, canvas, fingers)

                prev_pos = current_pos
            else:
                prev_pos = None

            # Combine the webcam image with the canvas
            image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

            # Show the combined image
            cv2.imshow("Math With Gesture", image_combined)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    main()