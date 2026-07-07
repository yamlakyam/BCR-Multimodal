#!/usr/bin/env python3
"""
TensorFlow-optional patch extractor for WSITools.

This version keeps the original WSITools behavior for:
- tissue detection
- patch extraction to .png / .jpg / .h5

and only requires TensorFlow if you choose save_format=".tfrecord".
"""

import logging
import os
import sys
import concurrent.futures

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.color import rgb2lab

# Optional imports
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import cupy  # type: ignore
except Exception:
    cupy = None

try:
    import cucim  # type: ignore
except Exception:
    cucim = None

try:
    import openslide  # type: ignore
except Exception:
    openslide = None


logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\x1b[80D\x1b[1A\x1b[K%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False

# WSITools originally tried to detect CUDA via TensorFlow.
# For your current pipeline (.png / .h5), keep this off by default.
device_list = []
is_cuda_gpu_available = False

# Used by parallel_save_patches
patch_cnt = 0


class ExtractorParameters:
    """
    Class for establishing & validating parameters for patch extraction
    """

    def __init__(
        self,
        save_dir=None,
        log_dir="./",
        save_format=".tfrecord",
        sample_cnt=-1,
        patch_filter_by_area=None,
        with_anno=True,
        threads=20,
        rescale_rate=128,
        patch_size=128,
        stride=128,
        patch_rescale_to=None,
        extract_layer=0,
        randomize_order=False,
    ):
        if save_dir is None:
            raise Exception("Must specify a directory to save the extraction")

        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_format = save_format
        self.with_anno = with_anno
        self.rescale_rate = rescale_rate
        self.patch_size = patch_size
        self.stride = stride
        self.patch_rescale_to = patch_rescale_to
        self.extract_layer = extract_layer
        self.patch_filter_by_area = patch_filter_by_area
        self.sample_cnt = sample_cnt
        self.randomize_order = randomize_order
        self.threads = threads

        os.makedirs(self.save_dir, exist_ok=True)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)


class PatchExtractor:
    """
    Class that sets up the remaining info for patch extraction,
    and contains the function to extract them
    """

    def __init__(self, detector=None, parameters=None, feature_map=None, annotations=None):
        if parameters is None:
            raise ValueError("parameters must be provided")

        self.tissue_detector = detector
        self.threads = parameters.threads
        self.save_dir = parameters.save_dir
        self.log_dir = parameters.log_dir
        self.rescale_rate = parameters.rescale_rate
        self.patch_size = parameters.patch_size
        self.stride = parameters.stride
        self.patch_rescale_to = parameters.patch_rescale_to
        self.extract_layer = parameters.extract_layer
        self.save_format = parameters.save_format
        self.patch_filter_by_area = parameters.patch_filter_by_area
        self.sample_cnt = parameters.sample_cnt
        self.randomize_order = parameters.randomize_order
        self.feature_map = feature_map
        self.annotations = annotations

        if self.save_format == ".tfrecord":
            if tf is None:
                raise ImportError(
                    "TensorFlow is required only for .tfrecord output. "
                    "Use .png, .jpg, or .h5 to avoid this dependency."
                )
            if feature_map is not None:
                self.with_feature_map = True
            else:
                raise Exception("A Feature map must be specified when you create tfRecords")
        else:
            if feature_map is not None:
                logger.info("No need to specify feature_map ... ignoring.")
            self.with_feature_map = False

        self.with_anno = annotations is not None

        os.makedirs(self.save_dir, exist_ok=True)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def get_case_info(wsi_fn):
        """
        Converts the WSI filename into a slide object and returns it plus metadata
        """
        if is_cuda_gpu_available and cucim is not None:
            wsi_obj = cucim.CuImage(wsi_fn)
        else:
            if openslide is None:
                raise ImportError(
                    "openslide is not available. Install openslide-python and openslide-bin."
                )
            wsi_obj = openslide.open_slide(wsi_fn)

        root_dir, fn = os.path.split(wsi_fn)
        uuid, ext = os.path.splitext(fn)
        case_info = {"fn_str": uuid, "ext": ext, "root_dir": root_dir}
        return wsi_obj, case_info

    def get_thumbnail(self, wsi_obj):
        """
        Given a slide object, return a down-sampled thumbnail image
        """
        if is_cuda_gpu_available and cupy is not None and cucim is not None:
            wsi_w, wsi_h = wsi_obj.shape[1], wsi_obj.shape[0]
            wsi_numpy = cupy.asarray(
                wsi_obj.read_region((0, 0), size=(wsi_w, wsi_h), num_workers=6),
                dtype="uint8",
            ).get()
            thumbnail = Image.fromarray(wsi_numpy[:: self.rescale_rate, :: self.rescale_rate, :])
        else:
            wsi_w, wsi_h = wsi_obj.dimensions
            thumb_size_x = max(1, int(wsi_w / self.rescale_rate))
            thumb_size_y = max(1, int(wsi_h / self.rescale_rate))
            thumbnail = wsi_obj.get_thumbnail([thumb_size_x, thumb_size_y]).convert("RGB")

        return thumbnail

    def get_patch_locations(self, wsi_thumb_mask, level_downsamples):
        """
        Return all positive patch coordinates from a thumbnail mask
        """
        wsi_thumb_mask = ndimage.binary_erosion(wsi_thumb_mask)
        pos_indices = np.where(wsi_thumb_mask > 0)

        if len(pos_indices[0]) == 0:
            return [[], []]

        loc_y = (np.array(pos_indices[0]) * self.rescale_rate).astype(np.int32)
        loc_x = (np.array(pos_indices[1]) * self.rescale_rate).astype(np.int32)

        loc_x_selected = []
        loc_y_selected = []

        x_lim = [int(min(loc_x)), int(max(loc_x))]
        y_lim = [int(min(loc_y)), int(max(loc_y))]

        step = int(self.stride * level_downsamples[self.extract_layer])

        for x in range(x_lim[0], x_lim[1], step):
            for y in range(y_lim[0], y_lim[1], step):
                x_idx = int(x / self.rescale_rate)
                y_idx = int(y / self.rescale_rate)
                x_idx_1 = int((x + self.patch_size * level_downsamples[self.extract_layer]) / self.rescale_rate)
                y_idx_1 = int((y + self.patch_size * level_downsamples[self.extract_layer]) / self.rescale_rate)

                if x_idx_1 >= wsi_thumb_mask.shape[1]:
                    x_idx_1 = x_idx
                if y_idx_1 >= wsi_thumb_mask.shape[0]:
                    y_idx_1 = y_idx

                if np.count_nonzero(wsi_thumb_mask[y_idx:y_idx_1, x_idx:x_idx_1]) > 0:
                    loc_x_selected.append(int(x))
                    loc_y_selected.append(int(y))

        if self.randomize_order and len(loc_x_selected) > 0:
            index = np.arange(len(loc_x_selected))
            np.random.shuffle(index)
            loc_x_selected = [loc_x_selected[k] for k in index]
            loc_y_selected = [loc_y_selected[k] for k in index]

        return [loc_x_selected, loc_y_selected]

    def get_patch_locations_from_ROIs(self, ROIs, level_downsamples):
        """
        Given a ROI list, return patch coordinates inside ROIs
        """
        loc_x_selected = []
        loc_y_selected = []

        step = int(self.stride * level_downsamples[self.extract_layer])

        for roi in ROIs:
            x_lim = [roi[0], roi[2]]
            y_lim = [roi[1], roi[3]]
            for x in range(x_lim[0], x_lim[1], step):
                for y in range(y_lim[0], y_lim[1], step):
                    loc_x_selected.append(int(x))
                    loc_y_selected.append(int(y))

        if self.randomize_order and len(loc_x_selected) > 0:
            index = np.arange(len(loc_x_selected))
            np.random.shuffle(index)
            loc_x_selected = [loc_x_selected[k] for k in index]
            loc_y_selected = [loc_y_selected[k] for k in index]

        return [loc_x_selected, loc_y_selected]

    def validate_extract_locations(self, case_info, locations, thumbnail, level_downsamples):
        """
        Save a validation image with extracted patch grid overlaid
        """
        if self.log_dir is None:
            print("log dir is None, validation image will not be saved")
            return

        os.makedirs(self.log_dir, exist_ok=True)

        loc_x_selected, loc_y_selected = locations
        if len(loc_x_selected) == 0:
            logger.warning("No patch locations found; skipping validation image.")
            return

        thumb_fn = os.path.join(
            self.log_dir,
            case_info["fn_str"] + "_extraction_grid_" + str(len(loc_x_selected)) + ".png",
        )
        if os.path.exists(thumb_fn):
            return

        thumb_copy = thumbnail.copy()
        draw = ImageDraw.Draw(thumb_copy)

        for i in range(len(loc_x_selected)):
            xy = [
                int(loc_x_selected[i] / self.rescale_rate),
                int(loc_y_selected[i] / self.rescale_rate),
                int((loc_x_selected[i] + self.patch_size * level_downsamples[self.extract_layer]) / self.rescale_rate),
                int((loc_y_selected[i] + self.patch_size * level_downsamples[self.extract_layer]) / self.rescale_rate),
            ]
            draw.rectangle(xy, outline="green")

        print("Grids numbers in total: %d" % len(loc_x_selected))
        thumb_copy.save(thumb_fn)

    @staticmethod
    def filter_by_content_area(rgb_image_array, area_threshold=0.4, brightness=85):
        """
        Return True if patch contains enough tissue
        """
        lab_img = rgb2lab(rgb_image_array)
        l_img = lab_img[:, :, 0]
        binary_img_array_1 = np.array(0 < l_img)
        binary_img_array_2 = np.array(l_img < brightness)
        binary_img = np.logical_and(binary_img_array_1, binary_img_array_2) * 255
        tissue_size = np.where(binary_img > 0)[0].size
        tissue_ratio = tissue_size * 3 / rgb_image_array.size
        return tissue_ratio > area_threshold

    def get_patch_label(self, patch_loc, Center=True):
        """
        Get annotation label for a patch location
        """
        if self.annotations is None:
            return -1, "None"

        if Center:
            pix_loc = (patch_loc[0] + self.patch_size, patch_loc[1] + self.patch_size)
        else:
            pix_loc = patch_loc

        label_id, label_txt = self.annotations.get_pixel_label(pix_loc)
        return label_id, label_txt

    def generate_patch_fn(self, case_info, patch_loc, label_text=None):
        if label_text is None:
            filename = (
                f"{case_info['fn_str']}_{int(patch_loc[0])}_{int(patch_loc[1])}{self.save_format}"
            )
        else:
            filename = (
                f"{case_info['fn_str']}_{int(patch_loc[0])}_{int(patch_loc[1])}_{label_text}{self.save_format}"
            )

        return os.path.join(self.save_dir, filename)

    def generate_tfRecords_fp(self, case_info):
        """
        Create TFRecord writer. Only valid if TensorFlow is installed.
        """
        if tf is None:
            raise ImportError(
                "TensorFlow is not installed. TFRecord output requires tensorflow, "
                "but your current workflow can use .png, .jpg, or .h5 instead."
            )

        tmp = case_info["fn_str"] + self.save_format
        fn = os.path.join(self.save_dir, tmp)
        writer = tf.io.TFRecordWriter(fn)
        return writer, fn

    def img_patch_generator(self, x, y, wsi_obj, case_info, tf_writer=None):
        """Return image patches if they have enough tissue"""
        patch = wsi_obj.read_region(
            (x, y),
            self.extract_layer,
            (self.patch_size, self.patch_size),
        ).convert("RGB")

        if self.patch_rescale_to:
            patch = patch.resize([self.patch_rescale_to, self.patch_rescale_to])

        content_rich = True
        if self.patch_filter_by_area:
            content_rich = self.filter_by_content_area(np.array(patch), area_threshold=self.patch_filter_by_area)

        if not content_rich:
            logger.debug("No content found in image patch x: {} y: {}".format(x, y))
            return

        global patch_cnt
        patch_cnt += 1

        if self.with_anno:
            label_id, label_txt = self.get_patch_label([x, y])
        else:
            label_txt = "None"
            label_id = -1

        if self.with_feature_map:
            if tf is None:
                raise ImportError("TensorFlow is required for tfrecord output.")
            values = []
            for eval_str in self.feature_map.eval_str:
                values.append(eval(eval_str))
            features = self.feature_map.update_feature_map_eval(values)
            example = tf.train.Example(features=tf.train.Features(feature=features))
            tf_writer.write(example.SerializeToString())
            sys.stdout.flush()
        else:
            fn = self.generate_patch_fn(case_info, (x, y), label_text=label_txt)
            if os.path.exists(fn):
                logger.error("You already wrote this image file")
            if self.save_format == ".jpg":
                patch.save(fn)
            elif self.save_format == ".png":
                patch.convert("RGBA").save(fn)
            else:
                raise Exception("Can't recognize save format")
            sys.stdout.flush()

    def parallel_save_patches(self, wsi_obj, case_info, indices):
        global patch_cnt
        patch_cnt = 0

        if self.with_feature_map:
            tf_writer, _ = self.generate_tfRecords_fp(case_info)
        else:
            tf_writer = None

        loc_x, loc_y = indices

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [
                executor.submit(self.img_patch_generator, x, y, wsi_obj, case_info, tf_writer)
                for x, y in zip(loc_x, loc_y)
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except NameError:
                    pass

        if self.with_feature_map and tf_writer is not None:
            tf_writer.close()

        logger.info("Found {} image patches".format(patch_cnt))

    def save_patch_without_annotation(self, wsi_obj, case_info, indices):
        """
        Save patches when annotations are not used
        """
        patch_cnt = 0

        if self.with_feature_map:
            tf_writer, _ = self.generate_tfRecords_fp(case_info)
        else:
            tf_writer = None

        loc_x, loc_y = indices
        for idx, _ in enumerate(loc_x):
            patch = wsi_obj.read_region(
                (loc_x[idx], loc_y[idx]),
                self.extract_layer,
                (self.patch_size, self.patch_size),
            ).convert("RGB")

            content_rich = True
            if self.patch_filter_by_area:
                content_rich = self.filter_by_content_area(np.array(patch), area_threshold=self.patch_filter_by_area)

            if content_rich:
                patch_cnt += 1
                if self.with_feature_map:
                    if tf is None:
                        raise ImportError("TensorFlow is required for tfrecord output.")
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    tf_writer.write(example.SerializeToString())
                    logger.info("\rWrote {} to tfRecords ".format(patch_cnt))
                    sys.stdout.flush()
                else:
                    fn = self.generate_patch_fn(case_info, (loc_x[idx], loc_y[idx]))
                    if self.save_format == ".jpg":
                        patch.save(fn)
                    elif self.save_format == ".png":
                        patch.convert("RGBA").save(fn)
                    else:
                        raise Exception("Can't recognize save format")
                    logger.info("\rWrote {} to image files ".format(patch_cnt))
                    sys.stdout.flush()
            else:
                logger.debug("No content found in image patch x: {} y: {}".format(loc_x[idx], loc_y[idx]))

        if tf_writer is not None:
            tf_writer.close()

        return patch_cnt

    def save_patches_h5file(self, wsi_obj, case_info, indices):
        """
        Save patches to .h5 format
        """
        patch_cnt = 0

        if self.save_format != ".h5":
            print("Wrong file format. Not saving to h5 file")
            return patch_cnt

        loc_x, loc_y = indices
        total_patch_num = len(loc_x)

        tmp = case_info["fn_str"] + "_loc" + self.save_format
        loc_fn = os.path.join(self.save_dir, tmp)
        loc_hdf5_file_w = h5py.File(loc_fn, mode="w")
        loc_hdf5_file_w.create_dataset(name="location", shape=[total_patch_num, 2], dtype=int, data=indices)
        loc_hdf5_file_w.close()

        tmp = case_info["fn_str"] + self.save_format
        fn = os.path.join(self.save_dir, tmp)
        hdf5_file_w = h5py.File(fn, mode="w")

        if self.patch_rescale_to:
            key_shape = [total_patch_num, 3, self.patch_rescale_to, self.patch_rescale_to]
        else:
            key_shape = [total_patch_num, 3, self.patch_size, self.patch_size]

        img_storage = hdf5_file_w.create_dataset(
            name="image",
            shape=key_shape,
            dtype=np.uint8,
            chunks=(1, 3, key_shape[2], key_shape[3]),
            compression="gzip",
        )

        for idx, _ in enumerate(loc_x):
            patch = wsi_obj.read_region(
                (loc_x[idx], loc_y[idx]),
                self.extract_layer,
                (self.patch_size, self.patch_size),
            ).convert("RGB")

            if self.patch_rescale_to:
                patch = patch.resize([self.patch_rescale_to, self.patch_rescale_to])

            img_arr = np.array(patch)[:, :, 0:3].astype(np.uint8).transpose(2, 0, 1)

            img_storage[idx] = img_arr
            patch_cnt += 1
            logger.info("\rWrote {} to h5 file ".format(patch_cnt))
            sys.stdout.flush()

        hdf5_file_w.close()
        return patch_cnt

    def save_patches(self, wsi_obj, case_info, indices):
        """
        Save patches to jpg/png or tfrecord
        """
        patch_cnt = 0

        if self.with_feature_map:
            tf_writer, _ = self.generate_tfRecords_fp(case_info)
        else:
            tf_writer = None

        loc_x, loc_y = indices
        for idx, _ in enumerate(loc_x):
            patch = wsi_obj.read_region(
                (loc_x[idx], loc_y[idx]),
                self.extract_layer,
                (self.patch_size, self.patch_size),
            ).convert("RGB")

            if self.patch_rescale_to:
                patch = patch.resize([self.patch_rescale_to, self.patch_rescale_to])

            content_rich = True
            if self.patch_filter_by_area:
                content_rich = self.filter_by_content_area(np.array(patch), area_threshold=self.patch_filter_by_area)

            if content_rich:
                patch_cnt += 1

                if self.with_anno:
                    label_id, label_txt = self.get_patch_label([loc_x[idx], loc_y[idx]])
                else:
                    label_txt = "None"
                    label_id = -1

                if self.with_feature_map:
                    if tf is None:
                        raise ImportError("TensorFlow is required for tfrecord output.")
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    tf_writer.write(example.SerializeToString())
                    logger.info("\rWrote {} to tfRecords ".format(patch_cnt))
                    sys.stdout.flush()
                else:
                    fn = self.generate_patch_fn(case_info, (loc_x[idx], loc_y[idx]), label_text=label_txt)
                    if not os.path.exists(os.path.split(fn)[0]):
                        os.makedirs(os.path.split(fn)[0], exist_ok=True)

                    if self.save_format == ".jpg":
                        patch.save(fn)
                    elif self.save_format == ".png":
                        patch.convert("RGBA").save(fn)
                    else:
                        raise Exception("Can't recognize save format")

                    logger.info("\rWrote {} to image files ".format(patch_cnt))
                    sys.stdout.flush()

                if self.sample_cnt == patch_cnt:
                    if tf_writer is not None:
                        tf_writer.close()
                    return patch_cnt
            else:
                logger.debug("No content found in image patch x: {} y: {}".format(loc_x[idx], loc_y[idx]))

        if tf_writer is not None:
            tf_writer.close()

        return patch_cnt

    def extract(self, wsi_fn):
        """
        Extract image patches from foreground tissue
        """
        wsi_obj, case_info = self.get_case_info(wsi_fn)
        wsi_fn_short = os.path.split(wsi_fn)[1]
        case_finished_fn = os.path.join(self.save_dir, "%s_case_finished.txt" % wsi_fn_short)

        if os.path.exists(case_finished_fn):
            print("Patch already extracted: %s" % wsi_fn_short)
            patches_cnt = 0
            try:
                with open(case_finished_fn, "r") as fp:
                    line = fp.readline().strip()
                    if ":" in line:
                        patches_cnt = int(line.split(":")[1].strip())
            except Exception:
                patches_cnt = 0
            return patches_cnt

        wsi_thumb = self.get_thumbnail(wsi_obj)
        wsi_thumb_mask = self.tissue_detector.predict(wsi_thumb)

        if is_cuda_gpu_available and hasattr(wsi_obj, "resolutions"):
            level_downsamples = wsi_obj.resolutions["level_downsamples"]
        else:
            level_downsamples = wsi_obj.level_downsamples

        extract_locations = self.get_patch_locations(wsi_thumb_mask, level_downsamples)
        self.validate_extract_locations(case_info, extract_locations, wsi_thumb, level_downsamples)

        if self.save_format == ".h5":
            patches_cnt = self.save_patches_h5file(wsi_obj, case_info, extract_locations)
        else:
            patches_cnt = self.save_patches(wsi_obj, case_info, extract_locations)

        with open(case_finished_fn, "w") as fp:
            fp.write("Patch Num: %d " % patches_cnt)

        return patches_cnt

    def extract_ROIs(self, wsi_fn, ROIs):
        """
        Extract patches from a list of ROIs
        """
        wsi_obj, case_info = self.get_case_info(wsi_fn)

        if is_cuda_gpu_available and hasattr(wsi_obj, "resolutions"):
            level_downsamples = wsi_obj.resolutions["level_downsamples"]
        else:
            level_downsamples = wsi_obj.level_downsamples

        extract_locations = self.get_patch_locations_from_ROIs(ROIs, level_downsamples)
        wsi_thumb = self.get_thumbnail(wsi_obj)
        self.validate_extract_locations(case_info, extract_locations, wsi_thumb, level_downsamples)
        return self.save_patches(wsi_obj, case_info, extract_locations)