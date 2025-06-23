import numpy as np
import scipy
import scipy.signal as scisig
import scipy.ndimage # Import ndimage directly for morphology operations


# naming conventions:
# ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
# ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2'])
# [        1,    2,      3,       4,     5,    6,    7,    8,     9,      10,            11,      12,     13]) #gdal
# [        0,    1,      2,       3,     4,    5,    6,    7,     8,      9,            10,      11,     12]) #numpy
# [              BB      BG       BR                       BNIR                                  BSWIR1    BSWIR2

# ge. Bands 1, 2, 3, 8, 11, and 12 were utilized as BB , BG , BR , BNIR , BSWIR1 , and BSWIR2, respectively.

# Naming conventions for Sentinel-2 bands (numpy 0-indexed)
# This mapping is crucial and should be clearly documented.
# QA60: 0
# B1 (Coastal Aerosol): 1
# B2 (Blue): 2
# B3 (Green): 3
# B4 (Red): 4
# B5 (VEG_RedEdge1): 5
# B6 (VEG_RedEdge2): 6
# B7 (VEG_RedEdge3): 7
# B8 (NIR): 8
# B8A (NIR_Narrow): 9
# B9 (Water Vapor): 10
# B10 (Cirrus): 11
# B11 (SWIR1): 12
# B12 (SWIR2): 13

# Note: Your original code's band mapping comment seems to have a mismatch.
# For example, B8 (NIR) is at index 7 in your old comment but 8 in S2 naming.
# BSWIR1 is at 11, BSWIR2 is at 12.
# Let's verify the band indices you are using.
# Original comment: ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2'])
# Original comment: [0,    1,      2,       3,     4,    5,    6,    7,     8,      9,            10,      11,     12]) #numpy
# Let's assume your numpy indices are actually:
# QA60: 0
# B1 (Coastal Aerosol): 1 ('cb')
# B2 (Blue): 2 ('blue')
# B3 (Green): 3 ('green')
# B4 (Red): 4 ('red')
# B5 (RedEdge1): 5 ('re1')
# B6 (RedEdge2): 6 ('re2')
# B7 (RedEdge3): 7 ('re3')
# B8 (NIR): 8 ('nir')
# B8A (NIR2): 9 ('nir2')
# B9 (WaterVapor): 10 ('waterVapor')
# B10 (Cirrus): 11 ('cirrus')
# B11 (SWIR1): 12 ('swir1')
# B12 (SWIR2): 13 ('swir2')

# Based on get_cloud_mask usage:
# data_image[1] -> B1 (Coastal Aerosol)
# data_image[0] -> QA60
# data_image[10] -> B9 (Water Vapor)
# data_image[3] -> B3 (Green)
# data_image[2] -> B2 (Blue)
# data_image[7] -> B7 (RedEdge3) or B8?
# data_image[11] -> B10 (Cirrus) or B11 (SWIR1)?

# It seems your mapping might be shifted or custom. For Sentinel-2 common use:
# B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6, B8=7, B8A=8, B9=9, B10=10, B11=11, B12=12.
# If your data is 0-indexed and bands are as per common Sentinel-2 (B1=0, B2=1, ..., B12=11),
# then data_image[1] is B2 (Blue), data_image[7] is B8 (NIR), data_image[11] is B12 (SWIR2).
# I'll use your original indices in the code, but you might want to double-check this mapping for correctness with your actual data bands.

def get_rescaled_data(data, limits):
    return (data - limits[0]) / (limits[1] - limits[0])


def get_normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    # get data between 0 and 1
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    shadow_mask = np.zeros((r, c)).astype('float32')

    BB = data_image[1]
    BNIR = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3 / 4
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadow_mask[shadow_tf] = -1
    shadow_mask = scisig.medfilt2d(shadow_mask, 5)

    return shadow_mask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, get_rescaled_data(data_image[1], [0.1, 0.5]))
    score = np.minimum(score, get_rescaled_data(data_image[0], [0.1, 0.3]))
    score = np.minimum(score, get_rescaled_data((data_image[0] + data_image[10]), [0.15, 0.2]))
    # Clouds are reasonably bright in all visible bands.
    score = np.minimum(score, get_rescaled_data((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))

    if use_moist_check:
        # Clouds are moist
        ndmi = get_normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, get_rescaled_data(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = get_normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, get_rescaled_data(ndsi, [0.8, 0.6]))

    box_size = 7
    box = np.ones((box_size, box_size)) / (box_size ** 2)
    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold] = 0

    return score


def get_cloud_cloudshadow_mask(data_image, cloud_threshold):
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(data_image)

    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = 0
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    return cloud_cloudshadow_mask