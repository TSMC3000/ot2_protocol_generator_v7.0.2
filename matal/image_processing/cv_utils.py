import numpy as np
import cv2

from .contours import sort_contours, grab_contours


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)


def normalize_img(src_img, lo=16, hi=240):
    img = cv2.normalize(src_img, None, lo, hi, cv2.NORM_MINMAX)
    return img


def denoise_img(src_img, h=20, tmpl=5, srch=5):
    assert src_img.dtype == np.uint8
    img = cv2.fastNlMeansDenoisingColored(src_img, None, h, h, tmpl, srch)
    return img


def filter_hsv(src_img, lo=(0, 0, 0), hi=(255, 255, 255), erode_rounds=0):
    assert src_img.dtype == np.uint8
    assert src_img.shape[2] == 3

    hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    img = mask.astype(np.uint8)

    if erode_rounds:
        img = cv2.erode(img, None, iterations=erode_rounds)
        img = cv2.dilate(img, None, iterations=erode_rounds)
    return img


def filter_hls(src_img, lo=(0, 0, 0), hi=(255, 255, 255), erode_rounds=0):
    assert src_img.dtype == np.uint8
    assert src_img.shape[2] == 3

    hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hsv, lo, hi)
    img = mask.astype(np.uint8)

    if erode_rounds:
        img = cv2.erode(img, None, iterations=erode_rounds)
        img = cv2.dilate(img, None, iterations=erode_rounds)
    return img


def find_correction_mat(src_img, return_counters=False, sort_by_area=True):
    assert src_img.dtype == np.uint8
    assert len(src_img.shape) == 2

    iw, ih = src_img.shape
    output_size = min(iw, ih)

    cnts = cv2.findContours(src_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # noqa E501
    cnts = grab_contours(cnts)
    if sort_by_area:
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:4]
    cnts = sort_contours(cnts)[0]

    homo_src, homo_dst = [], []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        sx, sy = x + w // 2, y + h // 2
        homo_src.append((sx, sy))
    homo_src = np.array(homo_src, dtype=np.float32)
    
    xmid, ymid = homo_src.mean(axis=0)
    for i, (sx, sy) in enumerate(homo_src):
        dx = 0 if sx < xmid else output_size
        dy = 0 if sy < ymid else output_size
        homo_dst.append((dx, dy))
    
    homo_dst = np.array(homo_dst, dtype=np.float32)

    h, mask = cv2.findHomography(homo_src, homo_dst)
    if return_counters:
        return h, cnts
    else:
        return h


def apply_correction_mat(src_img, h):
    iw, ih, *_ = src_img.shape
    output_size = min(iw, ih)
    return cv2.warpPerspective(src_img.astype(np.uint8), h, (output_size, output_size))


def calc_area(src_img):
    iw, ih, *_ = src_img.shape
    area_per_pixel = 2.54 * 2.54 / (iw * ih)
    white_pixels = np.sum(src_img) / 255.0
    area = white_pixels * area_per_pixel
    return area


def draw_mask(src_img, mask, color=(0, 0, 0), alpha=0.3):
    mask_img = np.zeros(src_img.shape, src_img.dtype)
    mask_img[:, :] = color
    mask_img = cv2.bitwise_and(mask_img, mask_img, mask=mask)
    img = cv2.addWeighted(mask_img, alpha, src_img, 1, 0, mask_img)
    return img


def draw_counters(src_img, cnts, lw=8):
    img = src_img.copy()
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(img, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), lw)
        cv2.putText(img, "{}".format(i + 1), (x + w // 4, y + 3 * h // 4), cv2.FONT_HERSHEY_SIMPLEX, (w + h) / 80,
                    (0, 0, 255), lw)
    return img
