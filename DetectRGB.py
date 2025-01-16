import cv2
import numpy as np

# taking the input from webcam
vid = cv2.VideoCapture(0)

# running while loop just to make sure that
# our program keeps running until we stop it
while True:
    # capturing the current frame
    _, frame = vid.read()

    # displaying the current frame
    cv2.imshow("frame", frame)

    # setting values for base colors
    b = frame[:, :, 0]  # Blue channel
    g = frame[:, :, 1]  # Green channel
    r = frame[:, :, 2]  # Red channel

    # computing the mean
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    # displaying the most prominent color
    if b_mean > g_mean and b_mean > r_mean:
        print("Blue")
    elif g_mean > r_mean and g_mean > b_mean:
        print("Green")
    else:
        print("Red")

    # breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing the video capture object and closing all windows
vid.release()
cv2.destroyAllWindows()
