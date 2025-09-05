from PIL import Image, ImageCms
import matplotlib as mpl

import numpy as np
from ..utils.log import auto_log

# Tested environment:
# - Python 3.7.2 with Pillow 5.4.1
# - Python 2.7.15 with Pillow 5.4.1


def _sanitize(img):
    """
    Convert image color mode to *RGB*, *RGBA*, *L* or *LA*. So it is easier for future
    processing and the returned image is always safe to be saved as PNG file.

    If the input file is in *Lab* color space, it is converted to *sRGB*.

    **Parameters**

        img: :class:`PIL.Image`
            Input image.

    **Returns**

        img: :class:`PIL.Image`
            sanitized image.
    """

    if img.mode in ['RGB', 'RGBA', 'L', 'LA']:
        img2 = img.copy()
    elif img.mode == 'RGBa':
        img2 = img.convert('RGBA')
    elif img.mode in ['1', 'I', 'F']:
        img2 = img.convert('L')
    elif img.mode in ['CMYK', 'YCbCr', 'HSV']:
        img2 = img.convert('RGB')
    elif img.mode == 'LAB':
        # PIL cannot convert Lab to RGB automatically.
        # Codes from https://stackoverflow.com/questions/52767317/
        srgb_p = ImageCms.createProfile('sRGB')
        lab_p  = ImageCms.createProfile('LAB')
        lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, 'LAB', 'RGB')
        img2 = ImageCms.applyTransform(img, lab2rgb)
    elif img.mode == 'P':
        # Seems that PIL have a bug when handling palette greyscale images with alpha
        # channel. Although it can be decoded properly, img.palette.mode will give 'RGB' 
        # instead of 'RGBA', even if alpha channel is present. This means we can not 
        # determine whether the original image palette has alpha information or not. 
        # So here the image will be forced to convert to 'RGBA'.
        img2 = img.convert('RGBA')
    else:
        print('[_sanitize] Unknown color mode: %s' % img.mode)
        img2 = img.convert('RGBA')

    auto_log('%s(%s) --> %s.' % (img.format, img.mode, img2.mode), level='debug')

    return img2


def _calc_avg_luminance(sanitized_img):
    """ 
    Calculate average luminance of a given sanitized image object


    **Parameters**

        img: :class:`PIL.Image`
            Input image (8-bit file, color mode must be *RGB*, *RGBA*, *L* or *LA*).

    **Returns**

        img: *float*
            average luminance.

    """

    from colorsys import rgb_to_hsv

    width, height = sanitized_img.size
    sanitized_data = np.asarray(sanitized_img, dtype=np.double)
    has_alpha = sanitized_img.mode.endswith('A')

    # flattened_data = sanitized_data.reshape((width * height, len(sanitized_data[0][0])))

    # # Because sanitized_img is always an 8-bit image, we can hard code 255 as limit.
    # if sanitized_img.mode in ['L', 'LA']: luminance_data = [(x[0]/255.0) for x in flattened_data]
    # else: luminance_data = [rgb_to_hsv(*(x/255.0)[:3])[2] for x in flattened_data]

    # If image has alpha information, alpha value is used to weight average calculation.
    # So transparent pixels will have less influences in calculating average luminance.
    if not has_alpha: 
        return np.average(sanitized_data.max(axis=2)) / 255.0
    else: 
        return np.average(sanitized_data[:,:,:-1].max(axis=2) / 255.0, weights=sanitized_data[:,:,-1] / 255.0)


def blur(original_img):
    """
    Blur input image, and save the result as a new PNG file.

    **Parameters**

        inname: *str*
            path to input image.

        suffix: *str, optional*
            suffix of output file name of blurred image, default is 'blur'.

    """
    width, height = original_img.size

    auto_log('load %s (%dx%d).' % (original_img.format, width, height), level='debug')

    sanitized_img = _sanitize(original_img)
    sanitized_data = np.asarray(sanitized_img, dtype=np.double)

    blurred_img = sanitized_img.copy()
    
    # Function for getting data of one pixel. And: 
    # - Limit x and y inside image dimensions for edge processing. This equals to
    #   pad the image out around the edges.
    # - A quick IF check for better performance, so we don't need to do 4 min/max
    #   calls for pixels that are not out of boundaries.
    get_px = lambda x, y: sanitized_data[y][x] if 0 <= x < width and 0 <= y < height else \
                          sanitized_data[max(min(height - 1, y), 0)][max(min(width - 1, x), 0)]
    
    # Function for calculating result data of one pixel
    def calc_px(x, y):
        temp = (get_px(x-1, y-1) + get_px( x , y-1) + get_px(x+1, y-1)
              + get_px(x-1,  y ) + get_px( x ,  y ) + get_px(x+1,  y )
              + get_px(x-1, y+1) + get_px( x , y+1) + get_px(x+1, y+1)) / 9.0
        return tuple(map(int, temp))

    # Apply manipulating function to the entire image
    blurred_data = [calc_px(x, y) for y in range(height) for x in range(width)]
    blurred_img.putdata(blurred_data)
    
    return blurred_img


def set_luminance(original_img, target_luminance, mixing_coeff=1.0):
    """
    Set luminance of input image to a target value. Save the result as a new PNG file.

    There are two ways to adjust the luminance of a given image. One is to multiply
    luminance of each pixels with a fixed scaling factor. This will create more nature
    looks, but are easier to be limited by highlight pixels. Another one is to add a
    fixed shift value to luminance of each pixels. This may result in absences of pure 
    black and white in the output image.

    This function can use both of those methods, and default to use the "fixed scaling" 
    method. you can use the optional argument *mixing_coeff* to control which method are
    used, or use both methods and mix their results together.

    The final luminance *L'* of a pixel which has original luminance *L* is:
    *L' = a \* (L \* scale) + b \* (L + shift)*, where *a* and *b* are weights controlled by 
    *mixing_coeff* and *a* + *b* is always 1.

    **Parameters**

        inname: *str*
            path to input image.

        target_luminance: *float*
            target luminance, on scale from 0.0 to 1.0.

        mixing_coeff: *float, optional*
            control mixing of results of two methods, on scale from -3.0 to 3.0. Default is 1.0,
            which is only use "fixed scaling" method. Value of 0.0 will use the average of 
            two methods. Positive value increase weight of "fixed scaling" method, while 
            negative value increase weight of "fixed shift" method. Value of 1 and -1 will 
            effectively "turn off" the other method. And value beyond 1 or -1 will resulting 
            in negative value of *b* or *a*, which usually leads to wired result but can be 
            useful in some cases.

        suffix: *str, optional*
            suffix of output file name, default is 'luminance'.

    **Returns**

        luminance: *float*
            average luminance of output image. Note that this may differ from *target_luminance*.
            When this happens it means there are pixels in the output image that already hit 
            pure black or white, so that the luminance adjustment is limited because those 
            pixels can not be darker or brighter.

    """

    from colorsys import rgb_to_hsv, hsv_to_rgb

    # Limit target_luminance and mixing_coeff in range
    target_luminance = max(min(1.0, target_luminance), 0.0)
    mixing_coeff = max(min(3.0, mixing_coeff), -3.0)

    # original_img = Image.open(inname)
    width, height = original_img.size
    sanitized_img = _sanitize(original_img)
    sanitized_data = np.asarray(sanitized_img, dtype=np.double)
    modified_img = sanitized_img.copy()

    # Calculate average luminance of input image, determine coefficients for luminance transform function.
    input_avg_luminance = _calc_avg_luminance(sanitized_img)
    auto_log(f'load {original_img.format} ({width}x{height}) with average luminance: {input_avg_luminance}.')

    # luminance transform function L' = f(L) = alpha * (L * scale)  +  beta * (L + shift)
    alpha, beta = (1 + mixing_coeff) / 2.0, (1 - mixing_coeff) / 2.0
    scale = target_luminance / input_avg_luminance
    shift = target_luminance - input_avg_luminance
    f = lambda L: alpha * (L * scale) + beta * (L + shift)
    auto_log("mixing_coeff = %.2f, L' = f(L) = %.2f * (L * %.4f) + %.2f * (L + %.4f)" % 
             (mixing_coeff, alpha, scale, beta, shift), level='debug')

    # Function for calculating result data of one pixel
    # Because sanitized_img is always an 8-bit image, we can hard code 255 as limit.
    # 'r', 'g', 'b' for 3 channels in RGB; 'h', 's', 'v' for 3 channels in HSV, 'a' for Alpha channel
    # if sanitized_img.mode == 'L': 
    #     def calc_px(x, y):
    #         v = sanitized_data[y][x][0]
    #         v = f(v)
    #         return int(v),
    # elif sanitized_img.mode == 'LA':
    #     def calc_px(x, y):
    #         v, a = sanitized_data[y][x]
    #         v = f(v)
    #         return int(v), int(a)
    # elif sanitized_img.mode == 'RGB':
    #     def calc_px(x, y):
    #         h, s, v = rgb_to_hsv(*(sanitized_data[y][x][:3] / 255.0))
    #         v = f(v)
    #         r, g, b = map(int, np.array(hsv_to_rgb(h, s, v)) * 255.0)
    #         return r, g, b
    # else:   # 'RGBA'
    #     def calc_px(x, y):
    #         h, s, v = rgb_to_hsv(*(sanitized_data[y][x][:3] / 255.0))
    #         v = f(v)
    #         r, g, b = map(int, np.array(hsv_to_rgb(h, s, v)) * 255.0)
    #         return r, g, b, int(sanitized_data[y][x][3])
    #
    # Apply manipulating function to the entire image
    # modified_data = [calc_px(x, y) for y in range(height) for x in range(width)]
    # modified_img.putdata(modified_data)
    
    img_data = mpl.colors.rgb_to_hsv(sanitized_data / 255.0)
    img_data[:,:,2] = f(img_data[:,:,2]).clip(0, 1)
    img_data = (mpl.colors.hsv_to_rgb(img_data).clip(0, 1) * 255.0).astype(np.uint8)
    modified_img = Image.fromarray(img_data)

    output_avg_luminance = _calc_avg_luminance(modified_img)
    auto_log('output average luminance: %.4f' % output_avg_luminance)
    if abs(target_luminance - output_avg_luminance) > 0.01:
        auto_log('luminance adjustment is limited by highlight and/or shadow clipping.', level='warning')

    return modified_img

