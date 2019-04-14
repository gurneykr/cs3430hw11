from PIL import Image, ImageDraw
import math
import numpy as np
import cv2


def gd_detect_edges(rgb_img, magn_thresh=20):
    # gray scale image
    greyed = rgb_img.convert('L')
    img = Image.new('L', greyed.size)

    for row in range(img.size[0]):
        if row > 0:
            for col in range(img.size[1]):
                if col > 1 and col < greyed.size[1]-1 and row > 1 and row < greyed.size[0]-1:
                    above = greyed.getpixel((row-1, col))
                    below = greyed.getpixel((row+1, col))
                    right = greyed.getpixel((row, col+1))
                    left = greyed.getpixel((row, col-1))
                    dy = above - below
                    dx = right - left
                    if dx == 0:
                        dx = 1
                    G = math.sqrt(dy**2 + dx**2)
                    if G > magn_thresh:
                        img.putpixel((row, col), 255)
                    else:
                        img.putpixel((row, col), 0)

    return img


def create_hough_matrix(image, rho_resolution=1, theta_resolution=1):
    height, width = image.size  # we need height and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height ** 2 + width ** 2))  # a**2 + b**2 = c**2

    # y axis
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)

    # x axis
    thetas = np.deg2rad(np.arange(0, 180, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Get all of the pixels with non-zero values - which will be the edges
    y_indexes, x_indexes = np.nonzero(image)

    for i in range(len(x_indexes)):  # cycle through edge points
        x = x_indexes[i]
        y = y_indexes[i]

        for j in range(len(thetas)):  # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)

            # Increment the position each time it is found
            H[rho, j] += 1

    return H, rhos, thetas


def find_hough_peaks(H, spl=30, threshold=0, nhood_size=3):
    # This will hold the coordinates of where the maxima were found
    indicies = []

    # Make a copy of the Hough matrix so that it can be modified without touching the original
    H1 = np.copy(H)

    # Keep going until the specified number of peaks has been found
    for i in range(spl):
        # Flatten the matrix and find the maximum value
        idx = np.argmax(H1)

        # Get the index where this maximum occurred
        H1_idx = np.unravel_index(idx, H1.shape)

        # Save this index to return
        indicies.append(H1_idx)

        # Now that we have the maximum, supress points in the neighborhood - the nhood_size
        idx_y, idx_x = H1_idx

        # Don't go too far to the left - ie negative values
        if (idx_x - (nhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (nhood_size / 2) + 1

        # Don't go too far to the right - past the size of the image
        if (idx_x + (nhood_size / 2) + 1) > H.shape[1]:
            # Stay at the right edge of the image
            max_x = H.shape[1]
        else:
            max_x = idx_x + (nhood_size / 2) + 1

        # Do the same for the y values
        if (idx_y - (nhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (nhood_size / 2)
        if (idx_y + (nhood_size / 2) + 1) > H.shape[0]:
            max_y = H.shape[0]
        else:
            max_y = idx_y + (nhood_size / 2) + 1

        # Now set the values in the neighborhood to zero so that they won't be considered next time thru the loop
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H. This ensures that they are actual peaks.
                if x == min_x or x == (max_x - 1):
                    H[y, x] = 255
                if y == min_y or y == (max_y - 1):
                    H[y, x] = 255

    return indicies


def draw_hough_lines(image, indicies, thetas, rhos):
    lnimg = image.copy()

    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        draw = ImageDraw.Draw(lnimg)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 255))

    return lnimg

def ht_detect_lines(image_file_path, magn_thresh, spl=20, nhood_size=3):
    # Load the file
    image = Image.open(image_file_path)

    # Get the edge file
    edge_image = gd_detect_edges(image, magn_thresh)

    # Create the hough transform matrix
    ht, rhos, thetas = create_hough_matrix(edge_image)

    # Find the peaks within the Hough Matrix
    indicies = find_hough_peaks(ht, spl, nhood_size)

    # Draw the lines on the image
    lnimg = draw_hough_lines(image, indicies, thetas, rhos)

    return image, lnimg, edge_image, ht
