import imageio
import cv2


def imread_func(path):
    # Returns RGB
    spath = str(path)
    image = cv2.imread(spath)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif spath.endswith('.gif'):
        image = imageio.mimread(spath)[0]
    else:
        raise RuntimeError(f'Can not read {path}')
    # Deal with channels
    assert len(image.shape) in (2, 3)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image
