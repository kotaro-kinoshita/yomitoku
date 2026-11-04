from yomitoku.dewarper import Dewarper


class Preprocessor:
    """Image corrections before analysis: geometric correction (dewarping of curled, curved, tilted or keystoned
    pages). Same interface as yomitoku-pro's Preprocessor (which also corrects the rotation first).

    Args:
        dewarp_coordinate (str): "dewarped" (results are in the dewarped image, default) or "photo" (the caller maps
            them back to the input image with `dewarper.last_result`).
    """

    def __init__(self, configs={}, device="cuda", dewarp_coordinate="dewarped"):
        dewarper_kwargs = {"device": device}

        if isinstance(configs, dict):
            if "dewarper" in configs:
                dewarper_kwargs.update(configs["dewarper"])
        else:
            raise ValueError("configs must be a dictionary")

        if dewarp_coordinate not in ("dewarped", "photo"):
            raise ValueError("dewarp_coordinate must be 'dewarped' or 'photo'")

        self.dewarper = Dewarper(**dewarper_kwargs)
        self.dewarp_coordinate = dewarp_coordinate
        self.rotated_img = None  # input of the dewarper (named as in pro, where it is the rotation-corrected image)

    def run(self, img):
        self.rotated_img = img
        dewarp_outputs, img = self.dewarper(img, self.dewarp_coordinate)
        return dewarp_outputs, img

    def __call__(self, img):
        dewarp_outputs, corrected_img = self.run(img)
        return {"dewarp": dewarp_outputs}, corrected_img
