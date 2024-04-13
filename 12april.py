import cv2
import numpy as np

def main():
    # Load input video
    video_capture = cv2.VideoCapture('input_video.mp4')

    # Load advertisement image
    ad_image = cv2.imread('Advertisement_Image.png', cv2.IMREAD_UNCHANGED)

    # Check if the video capture and image were loaded correctly
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    if ad_image is None:
        print("Error: Could not load image.")
        return

    # Get original video frame dimensions
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the desired output video resolution (reduce size by approximately 38%)
    output_width = int(frame_width * 0.38)
    output_height = int(frame_height * 0.38)

    # Resize the advertisement image to fit within the video frame
    ad_width = output_width // 7
    ad_height = output_height // 4
    ad_image_resized = cv2.resize(ad_image, (ad_width, ad_height))

    # Define the region on the left side where the advertisement will be placed
    ad_x, ad_y = 0, 0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 10.0, (output_width, output_height))

    # Process video frames
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the frame to the desired output resolution
        resized_frame = cv2.resize(frame, (output_width, output_height))

        # Split advertisement image into RGB and alpha channels
        ad_bgr = ad_image_resized[..., :3]  # Color channels
        ad_alpha = ad_image_resized[..., 3] / 255.0  # Alpha channel, normalized

        # Ensure to extract the region of interest (ROI) from the frame where ad will be placed
        roi = resized_frame[ad_y:ad_y + ad_height, ad_x:ad_x + ad_width]

        # Convert alpha channel to the same size as roi for blending
        ad_alpha = ad_alpha[:, :, None]  # Reshape alpha for broadcasting

        # Blend the advertisement with the ROI based on alpha channel
        blended = (roi * (1 - ad_alpha) + ad_bgr * ad_alpha).astype(np.uint8)

        # Place the blended image back into the frame
        resized_frame[ad_y:ad_y + ad_height, ad_x:ad_x + ad_width] = blended

        # Write the frame with advertisement overlay to the output video file
        out.write(resized_frame)

        # Display the frame with advertisement overlay
        cv2.imshow('Frame with Advertisement', resized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

