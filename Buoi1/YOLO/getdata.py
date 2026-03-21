import cv2

def FrameCapture(path):
    vid0bj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = vid0bj.read()
        if count % 10 == 0:
            cv2.imwrite("dataset\\frame%d.jpg" % (count/10), image)
        count += 1
if __name__ == '__main__':
    FrameCapture("test.mp4")