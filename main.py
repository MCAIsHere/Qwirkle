import cv2 as cv
import numpy as np

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(image):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Brown HSV range
    lower_brown = np.array([0, 40, 30])
    upper_brown = np.array([25, 200, 200])

    # Detect brown pixels
    brown_mask = cv.inRange(hsv, lower_brown, upper_brown)
    # Anything NOT brown
    not_brown_mask = cv.bitwise_not(brown_mask)

    # Highlight the non-brown regions
    highlight = img.copy()
    highlight[not_brown_mask > 0] = (0, 255, 0)  # mark non-brown pixels
    show_image('image_sharpened', highlight)

    # Detectam marginile (50–100 det mai multe margini; 300-600 det mai putine margini)
    edges = cv.Canny(highlight, 200, 300)
    show_image('edges', edges)
    # analizează o imagine binară (0 și 255) și caută forme închise sau margini continue.
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    # Selecteaza cel mai mare patrat
    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    # Alegem dimensiunea patratului scos (pentru quirkle ar fi 50x16 = 800)
    width = 1600
    height = 1600

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_HSV2BGR)
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
    show_image("detected corners", image_copy)

    source = np.array([top_left, bottom_left, top_right, bottom_right], dtype="float32")
    dest = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype="float32")
    M = cv.getPerspectiveTransform(source, dest)  # Creaza matricea de transformare
    result = cv.warpPerspective(image, M, (width, height))

    return result
img = cv.imread('antrenare/5_20.jpg')
img = cv.resize(img, (1600, 1600), interpolation=cv.INTER_AREA)
careu = extrage_careu(img)
show_image("Careu afisare",careu)

patratele = []
for i in range(0,1600,100):
    for j in range(0,1600,100):
        patratele.append(careu[i:i+100,j:j+100])

def diagonal_detector(patratele):
    template = cv.imread("Piese/doi.jpg",0)
    pozitii = [17,25,145,153]
    rez = []
    for pozitie in pozitii:
        pozitie_gray = cv.cvtColor(patratele[pozitie],cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(pozitie_gray, template, cv.TM_CCOEFF_NORMED)
        if result > 0.1:
            rez.append("secundara")
        else:
            rez.append("principala")
        print(result)
    return rez
def color_detector(patrat):
    patrat_hsv = cv.cvtColor(patrat, cv.COLOR_BGR2HSV)
    h, s, v = patrat_hsv[50][50]
    if 0 <= s <= 40 and 70 <= v: return "W"

    if h <= 20: return "O"
    if 25 <= h <= 35: return "Y"
    if 55 <= h <= 80: return "G"
    if 90 <= h <= 120: return "B"
    if 165 <= h: return "R"
def form_detector(patrat):
    patrat_hsv = cv.cvtColor(patrat, cv.COLOR_BGR2HSV)
    V = patrat_hsv[:,:,2]
    S = patrat_hsv[:,:,1]
    V_mean = np.mean(V)
    S_mean = np.mean(S)

    if V_mean > 150 and S_mean < 130:
        return '-'
    else:
        lista_corr = []
        patrat_gray = cv.cvtColor(patrat, cv.COLOR_BGR2GRAY)
        patrat_gray = cv.adaptiveThreshold(patrat_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,21,4)
        for i in range(1,7):
            piece_template = cv.imread(f"Piese/forma_{i}.jpg",0)
            piece_template = cv.GaussianBlur(piece_template,(5,5),0)
            _, piece_template = cv.threshold(piece_template,40,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
            kernel = np.ones((3,3), dtype="uint8")
            piece_template = cv.morphologyEx(piece_template,cv.MORPH_OPEN,kernel)

            max_corr = -1
            for rotate in range(4):
                corr = cv.matchTemplate(patrat_gray, piece_template, cv.TM_CCOEFF_NORMED)
                piece_template = cv.rotate(piece_template,cv.ROTATE_90_CLOCKWISE)


                if corr > max_corr: max_corr = corr
            lista_corr.append(max_corr)

        return int(np.argmax(lista_corr))+1
def detectare_piesa(patratele):
    rez = []
    for i in range(16):
        row = []
        for j in range(16):
            row.append(form_detector(patratele[i*16+j]))
        rez.append(row)
    return rez

matrice = detectare_piesa(patratele)
for linie in matrice:
    print(linie)

# fk = cv.imread("image.png",0)
# fk = cv.resize(fk,(100,100),interpolation=cv.INTER_AREA)
# cv.imwrite("Piese/forma_1.jpg", fk)




####################################

# PHOTO_SERIE = 1
# VALUORI = [['-' for _ in range(16)] for _ in range(16)]
#
# for i in range(21):
#     img = cv.imread('antrenare/' + str(PHOTO_SERIE) + f"_{i}.jpg")
#     img = cv.resize(img, (1600, 1600), interpolation=cv.INTER_AREA)
#     careu = extrage_careu(img)
#
#     patratele = []
#     for i in range(0, 1600, 100):
#         for j in range(0, 1600, 100):
#             patratele.append(careu[i:i + 100, j:j + 100])
#
#     if i == 0:
#         diagonale = diagonal_detector(patratele)




