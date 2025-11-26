import cv2 as cv
import numpy as np

def show_image(image,title="image"):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def extrage_careu(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Brown HSV range
    lower_brown = np.array([0, 40, 30])
    upper_brown = np.array([25, 200, 200])

    # Detect brown pixels
    brown_mask = cv.inRange(hsv, lower_brown, upper_brown)
    # Anything NOT brown
    not_brown_mask = cv.bitwise_not(brown_mask)

    # Highlight the non-brown regions
    highlight = image.copy()
    highlight[not_brown_mask > 0] = (0, 255, 0)  # mark non-brown pixels

    # Detectam marginile (50–100 det mai multe margini; 300-600 det mai putine margini)
    edges = cv.Canny(highlight, 200, 300)
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
    show_image(image_copy,"detected corners")

    source = np.array([top_left, bottom_left, top_right, bottom_right], dtype="float32")
    dest = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype="float32")
    M = cv.getPerspectiveTransform(source, dest)  # Creaza matricea de transformare
    result = cv.warpPerspective(image, M, (width, height))

    return result
def diagonal_detector(patratele):
    template = cv.imread("Piese/doi.jpg",0)
    pozitii = [17,25,145,153]
    rez = []
    for pozitie in pozitii:
        pozitie_gray = cv.cvtColor(patratele[pozitie],cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(pozitie_gray, template, cv.TM_CCOEFF_NORMED)
        if result > 0.3:
            rez.append("secundara")
        else:
            rez.append("principala")
        print(result)
    return rez
def piece_detector(patrat):
    patrat_gray = cv.cvtColor(patrat, cv.COLOR_BGR2GRAY)
    patrat_gray = cv.resize(patrat_gray[2:99, 2:99], (100,100), interpolation=cv.INTER_CUBIC)
    if np.sum(patrat_gray < 80) < 3200:
        return '-'
    else:
        return 'X'
def color_classifier(patrat):
    patrat_hsv = cv.cvtColor(patrat, cv.COLOR_BGR2HSV)
    h, s, v = patrat_hsv[50][50]
    if 0 <= s <= 40 and 70 <= v: return "W"

    if h <= 20: return "O"
    if 25 <= h <= 35: return "Y"
    if 55 <= h <= 80: return "G"
    if 90 <= h <= 120: return "B"
    if 165 <= h: return "R"
def piece_classifier(patrat):
    patrat_gray = cv.cvtColor(patrat, cv.COLOR_BGR2GRAY)

    max_corr = -1
    max_corr_index = -1
    for index in range(1, 8):
        template = cv.imread("Piese/f" + str(index) + ".jpg", 0)

        corr = cv.matchTemplate(patrat_gray, template, cv.TM_CCOEFF_NORMED)
        if np.max(corr) > max_corr:
            max_corr_index = index
            max_corr = np.max(corr)

    show_image(patrat_gray)
    template = cv.imread("Piese/f" + str(max_corr_index) + ".jpg", 0)
    show_image(template)

    if max_corr < 0.3: return "-"
    return (max_corr_index - 1) % 6 + 1
def score_calculator(VALORI, piese_noi):
    rows_checked = set()
    cols_checked = set()
    score = 0

    for piesa_noua in piese_noi:
        if piesa_noua[0] not in rows_checked:
            streak = 0

            index = piesa_noua[1] - 1
            while index >= 0 and VALORI[piesa_noua[0]][index] == 'X':
                streak += 1
                index -= 1
            index = piesa_noua[1] + 1
            while index <= 15 and VALORI[piesa_noua[0]][index] == 'X':
                streak += 1
                index += 1

            score += streak
            if streak != 0: score += 1
            if streak == 5: score += 6
        if piesa_noua[1] not in cols_checked:
            streak = 0

            index = piesa_noua[0] - 1
            while index >= 0 and VALORI[index][piesa_noua[1]] == 'X':
                streak += 1
                index -= 1
            index = piesa_noua[0] + 1
            while index <= 15 and VALORI[index][piesa_noua[1]] == 'X':
                streak += 1
                index += 1

            score += streak
            if streak != 0: score += 1
            if streak == 5: score += 6
        rows_checked.add(piesa_noua[0])
        cols_checked.add(piesa_noua[1])
    return score

#########################################
PHOTO_SERIE = 5
VALORI = [['-' for _ in range(16)] for _ in range(16)]

for photo_index in range(21):
    PHOTO_INDEX_STR = str(photo_index) if photo_index >= 10 else "0"+str(photo_index)

    img = cv.imread('antrenare/' + str(PHOTO_SERIE) + "_" + PHOTO_INDEX_STR + ".jpg")
    img = cv.resize(img, (1600, 1600), interpolation=cv.INTER_AREA)
    careu = extrage_careu(img)
    show_image(careu, "Careu afisare")

    patratele = []
    for i in range(0, 1600, 100):
        for j in range(0, 1600, 100):
            patratele.append(careu[i:i + 100, j:j + 100])

    if photo_index == 0:
        diagonale = diagonal_detector(patratele)
        spacing = [[0, 0], [0, 8], [8, 0], [8, 8]]
        for square in range(4):
            x = spacing[square][0]
            y = spacing[square][1]
            if diagonale[square] == "principala":
                for i in range(1, 7):
                    VALORI[i + x][i + y] = 'X'
                for i in range(1, 6):
                    VALORI[i + x][i + 1 + y] = '1'
                    VALORI[i + 1 + x][i + y] = '1'
                VALORI[1 + x][6 + y] = '2'
                VALORI[6 + x][1 + y] = '2'
            else:
                for i in range(1, 7):
                    VALORI[i + x][7 + y - i] = 'X'
                for i in range(1, 6):
                    VALORI[i + x][7 + y - i - 1] = '1'
                    VALORI[i + 1 + x][7 + y - i] = '1'
                VALORI[1 + x][1 + y] = '2'
                VALORI[6 + x][6 + y] = '2'
    else:
        piese_noi = []
        scor = 0
        for i in range(16):
            for j in range(16):
                if VALORI[i][j] != 'X' and piece_detector(patratele[i*16+j]) == 'X':
                    piese_noi.append([i,j])

                    if VALORI[i][j] == '1': scor += 1
                    elif VALORI[i][j] == '2': scor += 2
                    VALORI[i][j] = 'X'

        scor += score_calculator(VALORI,piese_noi)

        # Afiseaza in fisier locatia / tipul, dupa scorul
        with open(f"result/{PHOTO_SERIE}_{PHOTO_INDEX_STR}.txt", "w") as file:
            for piesa in piese_noi:
                x = piesa[0]
                y = piesa[1]
                file.write(f"{piesa[0]+1}{chr(piesa[1]+ord('A'))} "
                           f"{piece_classifier(careu[max(0,x*100-30):min(1600,x*100+130), max(0,y*100-30):min(1600,y*100+130)])}"
                           f"{color_classifier(patratele[piesa[0]*16+piesa[1]])}\n")
            file.write(str(scor))

        if photo_index == 10: break
for linie in VALORI:
    print(linie)

#
# f = cv.imread("dede.png")
# f = cv.resize(f,(100,100),cv.INTER_CUBIC)
# cv.imwrite("Piese/f8.jpg", f)


