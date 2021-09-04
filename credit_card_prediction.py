import cv2
import imutils
import keras
import numpy as np
from tensorflow.keras.models import load_model


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def doc_scan(image):
    ratio = image.shape[0] / 500.0

    orig = image.copy()
    image = imutils.resize(image, height = 500)
    orig_height, orig_width = image.shape[:2]
    original_area = orig_height * orig_width
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)

    contours, _  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        if area < (original_area/3):
            print("Error Image Invalid")
            return("ERROR")
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break
    
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    cv2.resize(warped, (640,403), interpolation = cv2.INTER_AREA)
    cv2.imwrite("test_card/credit_card_color.jpg", warped)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = warped.astype("uint8") * 255
    # cv2.imshow("Extracted Credit Card", warped)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped

image = cv2.imread('test_card/test_card.jpg')
image = doc_scan(image)

# hard coded
region = [(55, 210), (640, 290)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

roi = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
# cv2.imshow("Region", roi)
# cv2.imwrite("test_card/credit_card_extracted_digits.jpg", roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

classifier = load_model('models/creditcard.h5')

def x_cord_contour(contours):
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass

img = cv2.imread('test_card/credit_card_extracted_digits.jpg')
orig_img = cv2.imread('test_card/credit_card_color.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:13]
contours = sorted(contours, key = x_cord_contour, reverse = False)

full_number = []

def pre_process(image, inv = False):
    if (len(image.shape) == 3):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if inv == False:
        _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return cv2.resize(threshold, (32,32), interpolation = cv2.INTER_AREA)

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)    
    if w >= 5 and h >= 25 and cv2.contourArea(c) < 1000:
        roi = blurred[y:y + h, x:x + w]
        cv2.imshow("ROI1", roi)
        cv2.waitKey(0)
        roi_otsu = pre_process(roi, True)
        cv2.imshow("ROI2", roi_otsu)
        cv2.waitKey(0)
        roi_otsu = cv2.cvtColor(roi_otsu, cv2.COLOR_GRAY2RGB)
        roi_otsu = keras.preprocessing.image.img_to_array(roi_otsu)
        roi_otsu = roi_otsu * 1./255
        roi_otsu = np.expand_dims(roi_otsu, axis=0)
        image = np.vstack([roi_otsu])
        label = str(np.argmax(classifier.predict(image)))
        print(label)
        (x, y, w, h) = (x+region[0][0], y+region[0][1], w, h)
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x , y + 90), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow("image", orig_img)
        cv2.waitKey(0) 
        
cv2.destroyAllWindows()
