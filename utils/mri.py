# How to load and create images from Dicom files.

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as pltRectangle
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple
from skimage import exposure
from pathlib import Path
from loguru import logger

from utils.enums import Laterality as Lat, View
from utils.models import Point3D, Size3D, Rectangle, Box


class MRISeries:
    """MRI Series"""

    def __init__(self, itk_image: sitk.Image, is_sub: bool = False):
        # Get the original direction matrix
        original_direction = itk_image.GetDirection()
        identity_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        
        # Log the original direction for debugging
        logger.info(f"Original direction: {np.round(original_direction, 2)}")

        # If direction is not identity, reorient the image
        if not np.allclose(original_direction, identity_direction, atol=1e-1):
            logger.info("Image has non-standard direction, reorienting...")
            # itk_image = self.force_identity_direction(itk_image)
            itk_image = self.reorient_image(itk_image)
            logger.info(f"New direction after reorientation: {np.round(itk_image.GetDirection(), 2)}")

        self.sub = is_sub
        self.itk_image = itk_image
        self.dims = self.itk_image.GetSize()
        self.spacing = self.itk_image.GetSpacing()
        self.origin = self.itk_image.GetOrigin()
        self.direction = self.itk_image.GetDirection()
        self.z_x_ratio = self.get_z_x_voxel_size_ratio(self.itk_image)
        self.resampled_z_dim = int(self.dims[2] * self.z_x_ratio)

    @classmethod
    def from_files(cls, filepaths: List[str]):
        """Loads a Series from Dicom files. They need to be ordered"""
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filepaths)
        itk_volume = reader.Execute()
        del reader
        return cls(itk_volume)

    @classmethod
    def from_path(cls, path: Path):
        files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(str(path))
        return cls.from_files(files)

    @staticmethod
    def try_axis_permutations(image: sitk.Image) -> sitk.Image:
        """
        Try different axis permutations to find one that results in a direction matrix
        closest to the identity matrix.
        """
        identity_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        best_image = image
        best_diff = float('inf')
        
        # List of possible permutations
        permutations = [
            (0, 1, 2),  # Original
            (0, 2, 1),  # Swap y and z
            (1, 0, 2),  # Swap x and y
            (1, 2, 0),  # Cycle left
            (2, 0, 1),  # Cycle right
            (2, 1, 0),  # Swap x and z
        ]
        
        # Try each permutation
        for perm in permutations:
            # Skip identity permutation if image already has bad direction
            if perm == (0, 1, 2):
                continue
                
            # Apply permutation
            permuted = sitk.PermuteAxes(image, perm)
            
            # Check how close it is to identity
            direction = permuted.GetDirection()
            diff = sum((a - b) ** 2 for a, b in zip(direction, identity_direction))
            
            if diff < best_diff:
                best_diff = diff
                best_image = permuted
                logger.info(f"Found better permutation: {perm}, diff: {diff}")
                
            # If we find a good enough match, stop searching
            if np.allclose(direction, identity_direction, atol=1e-1):
                logger.info(f"Found good permutation: {perm}")
                return permuted
        
        # Try flipping axes for the best permutation
        for axis in range(3):
            # Create flip tuple (SimpleITK requires tuples for vector parameters)
            flip = (False, False, False)
            flip = tuple(True if i == axis else False for i in range(3))
            
            flipped = sitk.Flip(best_image, flip)
            direction = flipped.GetDirection()
            diff = sum((a - b) ** 2 for a, b in zip(direction, identity_direction))
            
            if diff < best_diff:
                best_diff = diff
                best_image = flipped
                logger.info(f"Improved by flipping axis {axis}, diff: {diff}")
        
        return best_image

    @staticmethod
    def reorient_image(itk_image):
        # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DICOMOrientImageFilter.html
        logger.info("reorienting image")
        filter = sitk.DICOMOrientImageFilter()
        filter.SetDesiredCoordinateOrientation("LPS")
        return filter.Execute(itk_image)

    def show(self):
        """Shows the MRI series."""
        sitk.Show(self.itk_image)

    def get_sagittal_slice(self, index):
        """Obtains a sagittal slice at the specified index"""
        assert np.allclose(
                self.direction,
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                atol=1e-1
            ), f"direction={np.round(self.direction, 2)}"
        x, y, z = self.dims
        assert 0 <= index <= x
        img = self._extract_image_filter(self.itk_image, (index, 0, 0), (0, y, z))
        return img

    def get_chest_start(self, threshold: int = 20):
        """get the index of the coronal plane where the chest starts"""
        index = self.dims[0] // 2
        middle_sagittal = self.get_sagittal_slice(index)
        middle_sagittal = sitk.RescaleIntensity(middle_sagittal, 0, 255)
        middle_sagittal = sitk.Cast(middle_sagittal, sitk.sitkUInt8)
        array = sitk.GetArrayFromImage(middle_sagittal)
        y_size = self.dims[2]
        # use only the central strip bc there is noise on the edges
        c_start, c_end = 2 * y_size // 5, 3 * y_size // 5
        col_mean = np.mean(array[c_start:c_end, :], axis=-2)
        position = np.argwhere(col_mean > threshold)[0][0]
        return int(position)

    def get_breasts_bounding_rectangles(
        self, chest_start
    ) -> Tuple[Rectangle, Rectangle]:
        """Get the bounding boxes of the breasts in the x-y plane"""

        def _get_breast_start_y(lat: sitk.Image):
            THRESHOLD = 10
            PERCENTILE = 99
            # use only the central column bc there is noise on the edges
            x, y, z = lat.GetSize()
            x_start, x_size = 2 * x // 5, x // 5
            z_start, z_size = 2 * z // 5, z // 5

            column = self._extract_image_filter(
                lat, (x_start, 10, z_start), (x_size, y - 10, z_size)
            )

            array = sitk.GetArrayFromImage(column)
            # the axis in the returned array in the order z, y, x
            average = np.mean(array, axis=(0, 2))

            from scipy.ndimage import gaussian_filter1d
            smoothed_signal = gaussian_filter1d(average, sigma=2)
            derivative = np.diff(smoothed_signal)
            threshold = 5
            start = np.where(np.abs(derivative) > threshold)[0][0]
            assert start > 0, f"start={start}, derivative={derivative}"
            return start

        def _get_breast_x_limits(
            lat: sitk.Image, breast_start, chest_start
        ) -> Tuple[Rectangle, Rectangle]:
            """Get the 2D start and size for the breast"""
            THRESHOLD = 5
            PERCENTILE = 90
            y_prominence = chest_start - breast_start
            if y_prominence > 0:
                # get an "axial" "thick slice" of pixels for averaging
                x, y, z = lat.GetSize()
                y_thickness = y_prominence // 5
                y_start = chest_start - 2 * y_thickness
                z_start, z_size = 2 * z // 5, z // 5
                slice = self._extract_image_filter(
                    lat,
                    (0, int(y_start), int(z_start)),
                    (int(x), int(y_thickness), int(z_size)),
                )
                array = sitk.GetArrayFromImage(slice)
                max_proxy = np.percentile(array, PERCENTILE, axis=(0, 1))
                binary = max_proxy > THRESHOLD
                change_indices = np.flatnonzero(binary[1:] != binary[:-1])
                if len(change_indices) < 2:
                    logger.warning("Could not find horizontal (x) breast position")
                    return 0, 0
                start, end = change_indices[0], change_indices[-1]
                return start, end
            else:
                # breast is inexistent or something wrong happened.
                raise ValueError(
                    f"breast_start={breast_start}, chest_start={chest_start}"
                )

        def _process_laterality(windowed, laterality: Lat):
            """returns one point and one dimension (start_x, start_y), (size_x, size_y)"""
            lat = MRISeries.extract_laterality_filter(windowed, laterality)
            start_y = _get_breast_start_y(lat)
            if start_y < chest_start:
                start_x, x_end = _get_breast_x_limits(lat, start_y, chest_start)
                start = (start_x, start_y)
                size = (x_end - start_x, chest_start - start_y)
                return start, size
            else:
                # breast is inexistent or something wrong happened.
                return (0, 0), (0, 0)

        # window the image with a very steep contrast
        assert self.sub == False
        low_value, high_value = self.get_pixel_value_range_of_interest_method_1(
            self.itk_image, 70
        )
        windowed = self.window_image(self.itk_image, low_value, high_value)
        r_start, r_size = _process_laterality(windowed, Lat.RIGHT)
        l_start, l_size = _process_laterality(windowed, Lat.LEFT)
        # the left breast x position is according to half of the image, add the remaining
        x, _, _ = windowed.GetSize()
        l_start = (l_start[0] + x // 2, l_start[1])

        return (r_start, r_size), (l_start, l_size)

    def get_bounding_boxes(
        self, bounding_rectangles: Tuple[Rectangle, Rectangle]
    ) -> Tuple[Box, Box]:
        """
        Get the bounding rectangles of the breasts.
        Considers the z/x voxel resampled size for choosing the mayor_axis.
        The following conditions are met
        - big_size = right_Y_size == right_Z_size == left_Y_size == left_Z_size
        - small_size = right_X_size == left_X_size
        - The right box never contains with any voxels in the left half of the volume and viceversa.

        This allows in the end to stitch a single image like this:
        ----------------------------
        |             |             |
        |             |             |
        |  <-r_sag    |    l_sag->  |
        |          r_Z|l_Z          |
        |    r_Y      |    l_Y      |
        ----------------------------
        |    r_Y      |    l_Y      |
        |  <-r_ax  r_X|l_X  l_ax->  |
        |             |             |
        ----------------------------
        "->" shows the final direction of the breast after fliping and rotating the images.

        """
        (r_start, r_size), (l_start, l_size) = bounding_rectangles
        rectangles_start = np.array([r_start, l_start])
        rectangles_size = np.array([r_size, l_size])
        # 1. Find boxes according to the resampled Z dim
        # Since the scan usually does not contain much air in the z dim, all of z dim will be included.
        x, y, z = self.dims
        S = big_size = self.resampled_z_dim
        s = small_size = x // 2
        boxes_start = np.array([[0, 0, 0], [s, 0, 0]])  # r_x, r_y, r_z  # l_x, l_y, l_z
        boxes_size = np.array([[s, S, S], [s, S, S]])  # r_X, r_Y, r_Z  # l_X, l_Y, l_Z
        # Center the boxes in the rectangle's Y center
        delta = boxes_size[:, 1] - rectangles_size[:, 1]
        boxes_start[:, 1] = rectangles_start[:, 1] - delta // 2
        # Detect out_of_bounds and translate box if necessary
        # in y_end
        boxes_start[:, 1] = np.min([boxes_start[:, 1], y - boxes_size[:, 1]], axis=0)
        # in y_0
        boxes_start[:, 1] = np.max([boxes_start[:, 1], [0, 0]], axis=0)
        r_box_start = boxes_start[0].astype(int).tolist()
        l_box_start = boxes_start[1].astype(int).tolist()
        r_box_size = boxes_size[0].astype(int).tolist()
        l_box_size = boxes_size[1].astype(int).tolist()
        return (r_box_start, r_box_size), (l_box_start, l_box_size)

    @staticmethod
    def extract_laterality_filter(image: sitk.Image, laterality: Lat):
        """
        Obtain a sitk image with the volume of the selected laterality.
        """
        x, y, z = image.GetSize()
        if laterality == Lat.RIGHT:
            start = (0, 0, 0)
            size = (x // 2, y, z)
        if laterality == Lat.LEFT:
            start = (x // 2, 0, 0)
            size = ((x - x // 2), y, z)
        if laterality == Lat.BILATERAL:
            return image
        return MRISeries._extract_image_filter(image, start, size)

    def get_image_without_chest(self, chest_start: int):
        """Get the volume that goes from the first coronal plane until the plane where chest starts"""
        x, _, z = self.dims
        start = (0, 0, 0)
        size = (x, chest_start, z)
        return self._extract_image_filter(self.itk_image, start, size)

    @staticmethod
    def _contrast_stretching_filter(image: sitk.Image, percentile_low, percentile_high):
        """apply contrast stretching to image for the values in the selected percentiles."""
        array = sitk.GetArrayFromImage(image)
        p_low, p_high = np.percentile(
            array, (percentile_low, percentile_high), method="lower"
        )
        print(p_low, p_high)
        array_rescale = exposure.rescale_intensity(array, in_range=(float(p_low), float(p_high)))
        image = sitk.GetImageFromArray(array_rescale)
        return image

    @staticmethod
    def get_pixel_value_range_of_interest_method_1(
        image, percentile_high=99.9
    ) -> Tuple[float, float]:
        """
        Expected histogram from a normal image
        min ...........
        x1  ..         |
        x   ......     |
        x   .....      | Region of interest
        x   ......     |
        x2  ...        |
        x   .
        max .
        """
        im_array = sitk.GetArrayFromImage(image).flatten()
        hist, bin_edges = np.histogram(im_array, bins=100, density=True)
        # get the deltas to find where x1 is
        deltas = hist[1:] - hist[:-1]
        # find where is the first bin that the values increase
        start_value_index = np.argwhere(deltas > 0)[0][0]
        # Sometimes there is no early increase, make_sure black it is below 5%.
        if start_value_index > 5:
            start_value_index = 5
        start_value = bin_edges[start_value_index]
        # rm black to be able to select what percentage of the pixels we want to not clip
        no_black = im_array[im_array > start_value]
        stop_value: float = np.percentile(no_black, percentile_high)  # type: ignore
        return start_value, stop_value

    @staticmethod
    def get_pixel_value_range_of_interest_method_2(
        image, percentile_high=99.999
    ) -> Tuple[float, float]:
        """
        Expected histogram from a subtraction image
        min .
        x1  ..
        x   ...
        x   ...........
        x   ....       | Region of interest
        x2  ..         |
        x   .
        max .
        """
        im_array = sitk.GetArrayFromImage(image).flatten()
        hist, bin_edges = np.histogram(im_array, bins=100, density=True)
        plt.hist(hist, bin_edges)
        start_index = np.argmax(hist)
        start_value = (
            bin_edges[start_index + 1] if bin_edges[start_index + 1] >= 0 else 0
        )
        no_black = im_array[im_array > start_value]
        stop_value: float = np.percentile(no_black, percentile_high)  # type: ignore
        return start_value, stop_value

    @staticmethod
    def _extract_image_filter(image: sitk.Image, start: Point3D, size: Size3D):
        extractor = sitk.ExtractImageFilter()
        extractor.SetIndex(start)
        extractor.SetSize(size)
        return extractor.Execute(image)

    @staticmethod
    def window_image(image: sitk.Image, min, max):
        executor = sitk.IntensityWindowingImageFilter()
        executor.SetWindowMinimum(int(min))
        executor.SetWindowMaximum(int(max))
        return executor.Execute(image)

    @staticmethod
    def mip_filter(image: sitk.Image, view: View):
        dims = {
            View.AX: 2,
            View.SAG: 0,
            View.COR: 1,
        }
        executor = sitk.MaximumProjectionImageFilter()
        executor.SetProjectionDimension(dims[view])
        return executor.Execute(image)

    def __sub__(self, other: "MRISeries"):
        assert self.dims == other.dims
        assert self.direction == other.direction
        executor = sitk.SubtractImageFilter()
        img = executor.Execute(
            sitk.Cast(self.itk_image, sitk.sitkInt32),
            sitk.Cast(other.itk_image, sitk.sitkInt32),
        )
        return MRISeries(img, is_sub=True)

    def _choose_pixel_range_method(self):
        return (
            self.get_pixel_value_range_of_interest_method_1
            if not self.sub
            else self.get_pixel_value_range_of_interest_method_2
        )

    def get_mip_pixel_range(
        self, chest_start: int, extra_pixel_percentage: float = 0.3
    ) -> Tuple[float, float]:
        """Get the suggested pixel value range for the mip according to the image (sub or normal).

        :param chest_start: the pixel position of the beginning of the chest
         as seen from posterior to anterior in the Y dimension
        :param extra_pixel_percentage: float between in the range [0,1] for the extra pixels that are
         included in the calculation of the windowing. It is clipped to satisfy the inequation:
         chest_start + y_size * extra_pixel_percentage <= y_size
        :return: low and high values for windowing the image.
        """
        y_size = self.dims[2]
        extra_pixels = y_size * extra_pixel_percentage
        extra_pixels = min(extra_pixels, y_size - chest_start)
        no_chest = self.get_image_without_chest(int(chest_start + extra_pixels))
        pixel_range_func = self._choose_pixel_range_method()
        low_value, high_value = pixel_range_func(no_chest)
        return low_value, high_value

    def get_full_mip(self, laterality: Lat, view: View, window: Tuple[int, int]):
        im = self.itk_image
        low_value, high_value = window
        im = self.window_image(im, low_value, high_value)
        if view == View.COR or view == View.SAG:
            im = self.resample_z_filter(im)
        im = self.extract_laterality_filter(im, laterality)
        mip = self.mip_filter(im, view)
        return mip

    def get_cropped_mips(
        self, bounding_boxes: Tuple[Box, Box], pixel_value_range: Tuple[int, int]
    ) -> Tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image]:
        """
        Obtain the Axial and Sagittal mips for each laterality.
        Using the volume inside the bounding_boxes.
        return (r_ax, l_ax, r_sag, l_sag)
        """

        def _process_mip(box: sitk.Image, view: View, low_value: int, high_value: int):
            mip = MRISeries.mip_filter(box, view)
            mip = MRISeries.window_image(mip, low_value, high_value)
            return mip

        low, high = pixel_value_range
        r_box, l_box = self._extract_boxes_filter(bounding_boxes)

        r_ax = _process_mip(r_box, View.AX, low, high)
        r_sag = _process_mip(r_box, View.SAG, low, high)
        l_ax = _process_mip(l_box, View.AX, low, high)
        l_sag = _process_mip(l_box, View.SAG, low, high)
        return r_ax, l_ax, r_sag, l_sag

    def _extract_boxes_filter(
        self, bounding_boxes: Tuple[Box, Box]
    ) -> Tuple[sitk.Image, sitk.Image]:
        """
        Extract the volumes that are within the bounding boxes. First resampling to get 'isometric' voxels.
        """
        (r_start, r_size), (l_start, l_size) = bounding_boxes
        # resampled image
        image = self.resample_z_filter(self.itk_image)
        # right box
        r_box = MRISeries._extract_image_filter(image, r_start, r_size)
        # left box
        l_box = MRISeries._extract_image_filter(image, l_start, l_size)
        return r_box, l_box

    @staticmethod
    def get_z_x_voxel_size_ratio(image: sitk.Image):
        """how many x pixels is worth one z pixel"""
        spacing = image.GetSpacing()
        return spacing[2] / spacing[0]

    @staticmethod
    def resample_z_filter(image: sitk.Image):
        """Resample Image to get isotropic voxels assuming that X ans Y have the same spacing."""
        executor = sitk.ResampleImageFilter()
        new_size, new_spacing = MRISeries.calculate_z_resampling_size_and_spacing(image)
        executor.SetSize(new_size)
        executor.SetOutputSpacing(new_spacing)
        executor.SetInterpolator(sitk.sitkNearestNeighbor)
        executor.SetOutputOrigin(image.GetOrigin())
        return executor.Execute(image)

    @staticmethod
    def calculate_z_resampling_size_and_spacing(image: sitk.Image):
        og_dims = image.GetSize()
        assert np.isclose(og_dims[0], og_dims[1]), (
            f"Assumption that X and Y are equal is false Image.Spacing:{og_dims}"
        )
        og_spacing = image.GetSpacing()
        ratio_z_x = MRISeries.get_z_x_voxel_size_ratio(image)
        output_z_size = np.floor(og_dims[2] * ratio_z_x)
        new_size = [int(og_dims[0]), int(og_dims[1]), int(output_z_size)]
        new_spacing = [og_spacing[0], og_spacing[1], og_spacing[0]]
        return new_size, new_spacing

    def get_z_resampling_size_and_spacing(self):
        return self.calculate_z_resampling_size_and_spacing(self.itk_image)

    @staticmethod
    def force_identity_direction(image: sitk.Image) -> sitk.Image:
        """
        Force the image to have an identity direction matrix by resampling it.
        This properly handles any arbitrary orientation.
        """
        # Create a new reference image with identity direction
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        pixel_type = image.GetPixelID()
        
        # Create an identity direction image with the same properties
        reference_image = sitk.Image(size, pixel_type)
        reference_image.SetSpacing(spacing)
        reference_image.SetOrigin(origin)
        # Direction is already identity by default
        
        # Create a transform that incorporates the current direction matrix
        dimension = image.GetDimension()
        transform = sitk.AffineTransform(dimension)
        
        # Set the matrix from the direction and apply it to the transform
        direction = np.array(image.GetDirection()).reshape(dimension, dimension)
        transform.SetMatrix(direction.flatten())
        
        # Set up the resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform.GetInverse())
        
        # Resample the image with identity direction
        resampled_image = resampler.Execute(image)
        
        return resampled_image


def debug_pipeline(mri1: MRISeries, mri2: MRISeries):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
    # MIP of first MRI
    # Get measurements
    chest_start = mri1.get_chest_start()
    window1 = mri1.get_mip_pixel_range(chest_start)
    mip1 = sitk.GetArrayFromImage(
        mri1.get_full_mip(Lat.BILATERAL, View.AX, window1)
    ).squeeze()

    bounding_rectangles = mri1.get_breasts_bounding_rectangles(chest_start)
    (
        (r_start, (r_size_x, r_size_y)),
        (
            l_start,
            (l_size_x, l_size_y),
        ),
    ) = bounding_rectangles
    rectangle_r = pltRectangle(
        r_start, r_size_x, r_size_y, edgecolor="green", facecolor="none", lw=1
    )
    rectangle_l = pltRectangle(
        l_start, l_size_x, l_size_y, edgecolor="red", facecolor="none", lw=1
    )

    bounding_boxes = mri1.get_bounding_boxes(bounding_rectangles)
    (r_start, r_size), (l_start, l_size) = bounding_boxes
    box_r = pltRectangle(
        r_start[:2], r_size[0], r_size[1], edgecolor="green", facecolor="none", lw=1
    )
    box_l = pltRectangle(
        l_start[:2], l_size[0], l_size[1], edgecolor="red", facecolor="none", lw=1
    )

    ax = axes[0, 0]
    ax.imshow(mip1, cmap="gray")
    ax.axhline(chest_start)
    ax.add_patch(rectangle_r)
    ax.add_patch(rectangle_l)
    ax.add_patch(box_r)
    ax.add_patch(box_l)
    ax.set_title("Series 1 MIP")

    ax = axes[2, 0]
    ax.set_title("MRI1 Voxel vals")
    ax.hist(sitk.GetArrayFromImage(mri1.itk_image).flatten(), bins=100, density=True)
    window = mri1.get_pixel_value_range_of_interest_method_1(mri1.itk_image)
    ax.axvspan(*window, color="blue", alpha=0.2)
    ax.set_ylim([0, 0.01])

    # MIP of second MRI
    window2 = mri1.get_mip_pixel_range(chest_start)
    mip2 = sitk.GetArrayFromImage(
        mri2.get_full_mip(Lat.BILATERAL, View.AX, window2)
    ).squeeze()
    ax = axes[1, 0]
    ax.imshow(mip2, cmap="gray")
    ax.set_title("Series 2 MIP")

    ax = axes[2, 1]
    ax.set_title("MRI2 Voxel vals")
    ax.hist(sitk.GetArrayFromImage(mri2.itk_image).flatten(), bins=100, density=True)
    window = mri2.get_pixel_value_range_of_interest_method_1(mri2.itk_image)
    ax.axvspan(*window, color="blue", alpha=0.2)
    ax.set_ylim([0, 0.01])

    # Create subtraction
    sub_mri = mri2 - mri1
    window = sub_mri.get_mip_pixel_range(chest_start)
    r_ax, l_ax, r_sag, l_sag = sub_mri.get_cropped_mips(bounding_boxes, window)

    ax = axes[2, 2]
    ax.set_title("SUB Voxel values")
    ax.hist(sitk.GetArrayFromImage(sub_mri.itk_image).flatten(), bins=100, density=True)
    ax.axvspan(*window, color="blue", alpha=0.2)
    ax.set_ylim([0, 0.01])

    # Right Axial
    mip_ra = sitk.GetArrayFromImage(r_ax)  # .squeeze()
    print(mip_ra.shape)
    mip_ra = mip_ra[0, :, :]
    ax = axes[0, 1]
    ax.imshow(mip_ra, cmap="gray")
    ax.set_title("Sub right axial MIP")

    # Left Axial
    mip_la = sitk.GetArrayFromImage(l_ax).squeeze()
    ax = axes[0, 2]
    ax.imshow(mip_la, cmap="gray")
    ax.set_title("Sub left sagittal MIP")

    # Right sag
    mip_rs = sitk.GetArrayFromImage(r_sag).squeeze()
    ax = axes[1, 1]
    ax.imshow(mip_rs, cmap="gray")
    ax.set_title("Sub right sagittal MIP")

    # Left sag
    mip_ls = sitk.GetArrayFromImage(l_sag).squeeze()
    ax = axes[1, 2]
    ax.imshow(mip_ls, cmap="gray")
    ax.set_title("Sub left sagittal MIP")

    # Create squared images that surround the breast.

    plt.show()
    return bounding_boxes, window


def pipeline(mri1: MRISeries, mri2: MRISeries):
    """
    Process the MRI series without visualizing the results.
    Returns the extracted data and images for testing.
    
    Args:
        mri1: The first MRI series (pre-contrast)
        mri2: The second MRI series (post-contrast)
        
    Returns:
        A dictionary containing:
        - chest_start: The detected chest start position
        - window1: The window range for mri1
        - window2: The window range for mri2
        - bounding_rectangles: The rectangles around the breasts
        - bounding_boxes: The 3D boxes around the breasts
        - sub_window: The window range for the subtraction
        - mips: A tuple of (mip1, mip2, mip_ra, mip_la, mip_rs, mip_ls)
    """
    # Get measurements from first MRI
    chest_start = mri1.get_chest_start()
    window1 = mri1.get_mip_pixel_range(chest_start)
    mip1 = sitk.GetArrayFromImage(
        mri1.get_full_mip(Lat.BILATERAL, View.AX, window1)
    ).squeeze()

    # Get breast bounding rectangles
    bounding_rectangles = mri1.get_breasts_bounding_rectangles(chest_start)
    
    # Get 3D bounding boxes
    bounding_boxes = mri1.get_bounding_boxes(bounding_rectangles)
    
    # Process second MRI
    window2 = mri1.get_mip_pixel_range(chest_start)
    mip2 = sitk.GetArrayFromImage(
        mri2.get_full_mip(Lat.BILATERAL, View.AX, window2)
    ).squeeze()
    
    # Create subtraction
    sub_mri = mri2 - mri1
    sub_window = sub_mri.get_mip_pixel_range(chest_start)
    r_ax, l_ax, r_sag, l_sag = sub_mri.get_cropped_mips(bounding_boxes, sub_window)

    # Extract MIP arrays
    mip_ra = sitk.GetArrayFromImage(r_ax)
    mip_ra = mip_ra[0, :, :]
    mip_la = sitk.GetArrayFromImage(l_ax).squeeze()
    mip_rs = sitk.GetArrayFromImage(r_sag).squeeze()
    mip_ls = sitk.GetArrayFromImage(l_sag).squeeze()
    
    # Return all the data as a dictionary for easier testing
    return {
        "chest_start": chest_start,
        "window1": window1,
        "window2": window2,
        "bounding_rectangles": bounding_rectangles,
        "bounding_boxes": bounding_boxes,
        "sub_window": sub_window,
        "mips": (mip1, mip2, mip_ra, mip_la, mip_rs, mip_ls),
        "sub_mri": sub_mri
    }


if __name__ == "__main__":
    
    def subtraction_from_paths(series_path_1, series_path_2):
        mri1 = MRISeries.from_path(series_path_1)
        mri2 = MRISeries.from_path(series_path_2)
        subtraction = mri2 - mri1
        return subtraction

    path_to_test_data = Path(__file__).parent.parent / "tests" / "data"

    series_path_1 = path_to_test_data / "3.000000-ax dyn pre-93877"
    series_path_2 = path_to_test_data / "5.000000-ax dyn 1st pass-59529"

    mri1 = MRISeries.from_path(series_path_1)
    mri2 = MRISeries.from_path(series_path_2)

    bounding_boxes, window = debug_pipeline(mri1, mri2)

