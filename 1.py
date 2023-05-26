import cv2 as cv
# import threading
import time
import multiprocessing

if __name__ == "__main__":
    # video = cv.VideoCapture("it.mp4")
    # while video.isOpened():
    #     # Read video capture
    #     ret, frame = video.read()
    #     # Display each frame
    #     cv.imshow("video", frame)
    #     # show one frame at a time
    #     key = cv.waitKey(0)
    #     while key not in [ord('q'), ord('k')]:
    #         key = cv.waitKey(0)
    #     # Quit when 'q' is pressed
    #     if key == ord('q'):
    #         break

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    def q(img):
        cv.imshow("video", img)
        cv.waitKey(0)

    img = mpimg.imread('demo/features.jpg')
    # import matplotlib
    # matplotlib.get_backend()

    # axes = plt.axes()

    # axes.plot([1, 2], [1,2])
    # axes.imshow(img)
    for i in range(5):
        t = multiprocessing.Process(target=q, args=(img,))
        t.start()
        time.sleep(2)
        t.terminate()
        t.join()

        print("d")
    # t = multiprocessing.Process(target=q, args=(img,))
    # t.start()
    # time.sleep(2)
    # t.terminate()
    # t.join()

    # plt.show()
    # axes.imshow(img)
    # plt.show()
