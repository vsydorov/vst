import cv2
import copy
import yacs
import os
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch

from detectron2.data import (
        DatasetCatalog, MetadataCatalog,)
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers.nms import nms, batched_nms
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as d2_transforms
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Boxes, Instances, pairwise_iou

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.tools import snippets
from thes.data.external_dataset import (
        DatasetDALY)
from thes.detectron.cfg import (
        D2DICT_GPU_SCALING_DEFAULTS,
        d2dict_gpu_scaling,
        set_detectron_cfg_base,
        set_detectron_cfg_train,
        set_detectron_cfg_test,
        )
from thes.detectron.internals import launch_w_logging
from thes.detectron.externals import (
        simple_d2_setup, DalyVideoDatasetTrainer,
        get_frame_without_crashing)
from thes.detectron.daly import (
        get_daly_split_vids, simplest_daly_to_datalist_v2,
        daly_to_datalist_pfadet, get_category_map_o100, make_datalist_o100,
        get_similar_action_objects_DALY,
        make_datalist_objaction_similar_merged)
from thes.eval_tools import (
        voclike_legacy_evaluation, voclike_legacy_evaluation_v2)


log = logging.getLogger(__name__)


def _set_defcfg_detectron(cfg):
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    """)
    cfg.set_typecheck(""" dataset.subset: ['train', 'test'] """)


def _set_defcfg_detectron_train(cfg):
    cfg.set_defaults(""" dataset.subset: 'train' """)
    cfg.set_deftype("""
    num_gpus: [~, int]
    d2:
        SOLVER.CHECKPOINT_PERIOD: [2500, int]
        TEST.EVAL_PERIOD: [0, int]
        # ... anything ...
    """)
    cfg.set_defaults(D2DICT_GPU_SCALING_DEFAULTS)


def _set_defcfg_detectron_test(cfg):
    cfg.set_defaults(""" dataset.subset: 'test' """)
    cfg.set_deftype("""
    conf_thresh: [0.0, float]
    nms:
        enable: [True, bool]
        batched: [False, bool]
        thresh: [0.3, float]
    """)


def _set_defcfg_object_hacks(cfg):
    cfg.set_deftype("""
    object_hacks:
        dataset: ['normal', ['normal', 'o100', 'action_object']]
        action_object:
            merge: ['sane', ['sane',]]
    """)


def _figure_out_disturl(add_args):
    if '--port_inc' in add_args:
        ind = add_args.index('--port_inc')
        port_inc = int(add_args[ind+1])
    else:
        port_inc = 0
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14 + port_inc
    dist_url = "tcp://127.0.0.1:{}".format(port)
    return dist_url


def _detectron_train_function(d_cfg, cf, nargs):
    simple_d2_setup(d_cfg)

    dataset_name = nargs.TRAIN_DATASET_NAME
    DatasetCatalog.register(dataset_name, lambda: nargs.datalist)
    MetadataCatalog.get(dataset_name).set(
            thing_classes=nargs.cls_names)

    trainer = nargs.trainer(d_cfg)
    trainer.resume_or_load(resume=nargs.resume)
    trainer.train()


def _datalist_hacky_converter(cf, dataset):
    if cf['object_hacks.dataset'] == 'normal':
        num_classes = 43
        object_names = dataset.object_names

        def datalist_converter(datalist):
            return datalist

    elif cf['object_hacks.dataset'] == 'o100':
        o100_objects, category_map = get_category_map_o100(dataset)
        num_classes = len(o100_objects)
        assert len(o100_objects) == 16
        object_names = o100_objects

        def datalist_converter(datalist):
            datalist = make_datalist_o100(datalist, category_map)
            return datalist

    elif cf['object_hacks.dataset'] == 'action_object':
        assert cf['object_hacks.action_object.merge'] == 'sane'
        action_object_to_object = get_similar_action_objects_DALY()
        object_names = sorted([x
            for x in set(list(action_object_to_object.values())) if x])
        num_classes = len(object_names)

        def datalist_converter(datalist):
            datalist = make_datalist_objaction_similar_merged(
                    datalist, dataset.object_names, object_names,
                    action_object_to_object)
            return datalist

    else:
        raise NotImplementedError()
    return num_classes, object_names, datalist_converter


def _train_routine(cf, cf_add_d2, out,
        cls_names, TRAIN_DATASET_NAME,
        datalist, add_args):

    num_classes = len(cls_names)
    d2_output_dir = str(small.mkdir(out/'d2_output'))
    d_cfg = set_detectron_cfg_base(
            d2_output_dir, num_classes, cf['seed'])
    d_cfg = set_detectron_cfg_train(
            d_cfg, TRAIN_DATASET_NAME, cf_add_d2)

    num_gpus = cf['num_gpus']

    nargs = argparse.Namespace()
    nargs.datalist = datalist
    nargs.TRAIN_DATASET_NAME = TRAIN_DATASET_NAME
    nargs.cls_names = cls_names
    nargs.resume = True
    nargs.trainer = DalyVideoDatasetTrainer

    dist_url = _figure_out_disturl(add_args)
    launch_w_logging(_detectron_train_function,
            num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, nargs))


def train_daly_action(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_train(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')
    cf_add_d2 = d2dict_gpu_scaling(cf, cf_add_d2, cf['num_gpus'])

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)

    cls_names = dataset.action_names

    TRAIN_DATASET_NAME = 'daly_pfadet_train'
    _train_routine(cf, cf_add_d2, out,
        cls_names, TRAIN_DATASET_NAME, datalist, add_args)


def train_daly_object(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_train(cfg)
    _set_defcfg_object_hacks(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')
    cf_add_d2 = d2dict_gpu_scaling(cf, cf_add_d2, cf['num_gpus'])

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    o100_objects, category_map = get_category_map_o100(dataset)
    assert len(o100_objects) == 16
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)

    num_classes, cls_names, datalist_converter = \
            _datalist_hacky_converter(cf, dataset)
    datalist = datalist_converter(datalist)

    TRAIN_DATASET_NAME = 'daly_objaction_train'
    _train_routine(cf, cf_add_d2, out,
        cls_names, TRAIN_DATASET_NAME, datalist, add_args)


def _eval_foldname_hack(cf):
    if cf['eval_hacks.model_to_eval'] == 'what':
        model_to_eval = cf['what_to_eval']
    elif cf['eval_hacks.model_to_eval'] == 'what+foldname':
        import dervo
        EPATH = dervo.experiment.EXPERIMENT_PATH
        model_name = EPATH.resolve().name
        model_to_eval = str(Path(cf['what_to_eval'])/model_name)
    else:
        raise NotImplementedError('Wrong eval_hacks.model_to_eval')
    return model_to_eval


def _eval_routine(cf, cf_add_d2, out,
        cls_names, TEST_DATASET_NAME,
        datalist, model_to_eval):

    num_classes = len(cls_names)
    d2_output_dir = str(small.mkdir(out/'d2_output'))
    d_cfg = set_detectron_cfg_base(
            d2_output_dir, num_classes, cf['seed'])
    d_cfg = set_detectron_cfg_test(
            d_cfg, TEST_DATASET_NAME,
            model_to_eval, cf['conf_thresh'], cf_add_d2)

    simple_d2_setup(d_cfg)
    predictor = DefaultPredictor(d_cfg)
    cpu_device = torch.device("cpu")

    def eval_func(dl_item):
        video_path = dl_item['video_path']
        frame_number = dl_item['video_frame_number']
        frame_time = dl_item['video_frame_number']
        frame_u8 = get_frame_without_crashing(
            video_path, frame_number, frame_time)
        predictions = predictor(frame_u8)
        cpu_instances = predictions["instances"].to(cpu_device)
        return cpu_instances

    df_isaver = snippets.Simple_isaver(
            small.mkdir(out/'isaver'), datalist, eval_func, '::50')
    predicted_datalist = df_isaver.run()

    if cf['nms.enable']:
        nms_thresh = cf['nms.thresh']
        nmsed_predicted_datalist = []
        for pred_item in predicted_datalist:
            if cf['nms.batched']:
                keep = batched_nms(pred_item.pred_boxes.tensor,
                        pred_item.scores, pred_item.pred_classes, nms_thresh)
            else:
                keep = nms(pred_item.pred_boxes.tensor,
                        pred_item.scores, nms_thresh)
            nmsed_item = pred_item[keep]
            nmsed_predicted_datalist.append(nmsed_item)
        predicted_datalist = nmsed_predicted_datalist

    log.info('AP v1:')
    voclike_legacy_evaluation(cls_names, datalist, predicted_datalist)

    log.info('AP v2:')
    voclike_legacy_evaluation_v2(cls_names, datalist, predicted_datalist)


def eval_daly_action(workfolder, cfg_dict, add_args):
    """
    Evaluation code with hacks
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_test(cfg)
    cfg.set_deftype("""
    what_to_eval: [~, str]
    eval_hacks:
        model_to_eval: ['what', ['what', 'what+foldname']]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)
    cls_names = dataset.action_names

    TEST_DATASET_NAME = 'daly_pfadet_test'

    model_to_eval = _eval_foldname_hack(cf)
    _eval_routine(cf, cf_add_d2, out, cls_names,
        TEST_DATASET_NAME, datalist, model_to_eval)


def eval_daly_object(workfolder, cfg_dict, add_args):
    """
    Evaluation code with hacks
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_test(cfg)
    _set_defcfg_object_hacks(cfg)
    cfg.set_deftype("""
    what_to_eval: [~, str]
    eval_hacks:
        model_to_eval: ['what', ['what', 'what+foldname']]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)

    num_classes, cls_names, datalist_converter = \
            _datalist_hacky_converter(cf, dataset)
    datalist = datalist_converter(datalist)

    TEST_DATASET_NAME = 'daly_objaction_test'

    model_to_eval = _eval_foldname_hack(cf)
    _eval_routine(cf, cf_add_d2, out, cls_names,
        TEST_DATASET_NAME, datalist, model_to_eval)


def _cpu_and_thresh_instances(predictions, post_thresh, cpu_device):
    instances = predictions["instances"].to(cpu_device)
    good_scores = instances.scores > post_thresh
    tinstances = instances[good_scores]
    return tinstances


def _d2vis_draw_gtboxes(frame_u8, tinstances, metadata):
    visualizer = Visualizer(frame_u8, metadata=metadata, scale=1)
    img_vis = visualizer.draw_instance_predictions(
            predictions=tinstances)
    img = img_vis.get_image()
    return img


def _predict_rcnn_given_box_resized_proposals(
        box4, frame_u8, transform_gen, model):

    o_height, o_width = frame_u8.shape[:2]
    got_transform = transform_gen.get_transform(frame_u8)

    # Transform image
    image = got_transform.apply_image(frame_u8)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    imshape = tuple(image.shape[1:3])

    # / Transform box
    assert box4.shape == (4,)
    boxes_unscaled = box4[None]
    t_boxes = torch.as_tensor(boxes_unscaled.astype("float32"))
    transformed_t_boxes = got_transform.apply_box(t_boxes)
    # // Proposals w.r.t transformed imagesize
    proposal = Instances(imshape)
    tb_boxes = Boxes(transformed_t_boxes)
    proposal.proposal_boxes = tb_boxes

    inputs = {
            "image": image,
            "proposals": proposal,
            "height": o_height,
            "width": o_width}

    with torch.no_grad():
        predictions = model([inputs])[0]
    return predictions


def eval_daly_tubes_RGB_with_pfadet_demovis(workfolder, cfg_dict, add_args):
    """
    Run out own trained model on tubes
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cfg.set_deftype("""
    compute:
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    save_period: ['::10', str]
    """)
    cfg.set_deftype("""
    some_tubes:
        N: [50, int]
        seed: [0, int]
    conf_thresh: [0.0, float]
    trained_d2_model: [~, ~]
    seed: 42
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    cls_names = dataset.action_names
    num_classes = len(cls_names)

    # Dataset
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)
    TEST_DATASET_NAME = 'daly_pfadet_train'
    DatasetCatalog.register(TEST_DATASET_NAME, lambda: datalist)
    MetadataCatalog.get(TEST_DATASET_NAME).set(
            thing_classes=cls_names)
    metadata = MetadataCatalog.get(TEST_DATASET_NAME)

    # / Define d2 conf
    d2_output_dir = str(small.mkdir(out/'d2_output'))
    d_cfg = set_detectron_cfg_base(
            d2_output_dir, num_classes, cf['seed'])
    d_cfg = set_detectron_cfg_test(
            d_cfg, TEST_DATASET_NAME,
            cf['trained_d2_model'], cf['conf_thresh'], cf_add_d2,
            freeze=False)
    d_cfg2 = copy.deepcopy(d_cfg)
    d_cfg.freeze()
    # / Start d2
    simple_d2_setup(d_cfg)
    # Predictor with proposal generator
    predictor = DefaultPredictor(d_cfg)

    # Predictor without proposal generator
    d_cfg2.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
    d_cfg2.freeze()
    model = build_model(d_cfg2)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(d_cfg2.MODEL.WEIGHTS)
    MIN_SIZE_TEST = d_cfg2.INPUT.MIN_SIZE_TEST
    MAX_SIZE_TEST = d_cfg2.INPUT.MAX_SIZE_TEST
    transform_gen = d2_transforms.ResizeShortestEdge(
        [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)
    cpu_device = torch.device("cpu")

    # Load tubes
    tubes_per_video = _set_tubes(cf, dataset)
    some_tubes = sample_some_tubes(
            tubes_per_video, N=cf['some_tubes.N'],
            NP_SEED=cf['some_tubes.seed'])
    # k = ('S_PwpNZWgpk', 0, 19)
    # some_tubes = {k: tubes_per_video[k]}

    post_thresh = 0.2

    for k, tube in tqdm(some_tubes.items(), 'tubes vis'):
        (vid, bunch_id, tube_id) = k
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        frame_inds = tube['frame_inds']
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        tube_prefix = '{}_B{:02d}T{:02d}'.format(vid, bunch_id, tube_id)
        tube_fold = small.mkdir(out/tube_prefix)
        for i, (frame_ind, frame_u8) in enumerate(zip(frame_inds, frames_u8)):
            frame_prefix = tube_fold/'I{}_F{:03d}'.format(i, frame_ind)

            predictions = predictor(frame_u8)
            tinstances = \
                    _cpu_and_thresh_instances(predictions, post_thresh, cpu_device)
            img = _d2vis_draw_gtboxes(frame_u8, tinstances, metadata)
            cv2.imwrite(str(out/f'{frame_prefix}_frcnn.jpg'), img)

            # Get tube box, pass tube box through the rcnn part
            box4 = tube['boxes'][i]
            predictions = _predict_rcnn_given_box_resized_proposals(
                    box4, frame_u8, transform_gen, model)
            tinstances = \
                    _cpu_and_thresh_instances(predictions, post_thresh, cpu_device)
            img = _d2vis_draw_gtboxes(frame_u8, tinstances, metadata)
            snippets.cv_put_box_with_text(
                    img, box4, text='philtube')
            cv2.imwrite(str(out/f'{frame_prefix}_rcnn.jpg'), img)


def eval_daly_tubes_RGB_with_pfadet(workfolder, cfg_dict, add_args):
    """
    Run out own trained model on tubes
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    # _set_tubecfg
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cfg.set_deftype("""
    compute:
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
        equal_split: ['frames', ['frames', 'tubes']]
    save_period: ['::10', str]
    """)
    cfg.set_deftype("""
    some_tubes:
        N: [50, int]
        seed: [0, int]
    conf_thresh: [0.0, float]
    trained_d2_model: [~, ~]
    seed: 42
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    cls_names = dataset.action_names
    num_classes = len(cls_names)

    # Dataset
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)
    TEST_DATASET_NAME = 'daly_pfadet_train'
    DatasetCatalog.register(TEST_DATASET_NAME, lambda: datalist)
    MetadataCatalog.get(TEST_DATASET_NAME).set(
            thing_classes=cls_names)
    metadata = MetadataCatalog.get(TEST_DATASET_NAME)

    # / Define d2 conf
    d2_output_dir = str(small.mkdir(out/'d2_output'))
    d_cfg = set_detectron_cfg_base(
            d2_output_dir, num_classes, cf['seed'])
    d_cfg = set_detectron_cfg_test(
            d_cfg, TEST_DATASET_NAME,
            cf['trained_d2_model'], cf['conf_thresh'], cf_add_d2,
            freeze=False)
    d_cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
    d_cfg.freeze()

    # / Start d2
    simple_d2_setup(d_cfg)

    # Predictor without proposal generator
    model = build_model(d_cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(d_cfg.MODEL.WEIGHTS)
    MIN_SIZE_TEST = d_cfg.INPUT.MIN_SIZE_TEST
    MAX_SIZE_TEST = d_cfg.INPUT.MAX_SIZE_TEST
    transform_gen = d2_transforms.ResizeShortestEdge(
        [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)
    cpu_device = torch.device("cpu")

    # Load tubes
    tubes_per_video = _set_tubes(cf, dataset)
    ctubes_per_video = _parcel_management(cf, tubes_per_video)

    def tube_eval_func(k):
        tube = ctubes_per_video[k]
        (vid, bunch_id, tube_id) = k
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        frame_inds = tube['frame_inds']
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)

        instances_per_frame = []
        for i, (frame_ind, frame_u8) in enumerate(zip(frame_inds, frames_u8)):
            # Get tube box, pass tube box through the rcnn part
            box4 = tube['boxes'][i]
            predictions = _predict_rcnn_given_box_resized_proposals(
                    box4, frame_u8, transform_gen, model)
            instances = predictions["instances"].to(cpu_device)
            # Simply record all predictions
            instances_per_frame.append(instances)
        return instances_per_frame

    df_isaver = snippets.Simple_isaver(
            small.mkdir(out/'tube_eval_isaver'),
            list(ctubes_per_video.keys()),
            tube_eval_func, cf['save_period'], 120)
    predicted_tube_instances = df_isaver.run()
    tube_instances_dict = dict(zip(
        ctubes_per_video.keys(),
        predicted_tube_instances))
    small.save_pkl(out/'tube_instances_dict.pkl', tube_instances_dict)
