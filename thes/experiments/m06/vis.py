def _vis_scoresorted_tubes(out, dataset, wnms_av_stubes):
    action = 'Drinking'
    vfold = small.mkdir(out/action)
    v_stubes = wnms_av_stubes[action]
    flat_tubes = []
    for vid, stubes in v_stubes.items():
        for i_stube, stube in enumerate(stubes):
            flat_tubes.append({'tube': stube, 'ind': (vid, i_stube)})
    sorted_flat_tubes = sorted(flat_tubes,
            key=lambda x: x['tube']['score'], reverse=True)

    for i_sorted, flat_tube in enumerate(sorted_flat_tubes):
        vid, i_stube = flat_tube['ind']
        tube = flat_tube['tube']
        score = tube['score']
        sf, ef = tube['start_frame'], tube['end_frame']
        frame_inds = tube['frame_inds']
        video_fold = small.mkdir(vfold/f'{i_sorted:04d}_vid{vid}_{sf}_to_{ef}_score{score:02f}')
        video_path = dataset.videos[vid]['path']

        # Extract
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        # Draw
        drawn_frames_u8 = []
        for i, (find, frame_BGR) in enumerate(zip(frame_inds, frames_u8)):
            image = frame_BGR.copy()
            box = tube['boxes'][i]
            snippets.cv_put_box_with_text(image, box,
                text='{} {} {:.2f}'.format(
                    i, action, score))
            drawn_frames_u8.append(image)

        # # Save as images
        # for find, image in zip(frame_inds, drawn_frames_u8):
        #     cv2.imwrite(str(
        #         video_fold/'Fr{:05d}.png'.format(find)), image)

        # Save as video
        snippets.qsave_video(video_fold/'overlaid.mp4', drawn_frames_u8)

def faster_sample_via_pyav(video_path, finds_to_sample, threading=True):
    # A bit broken
    margin = 1024
    skip_offset = 2048
    stream_name = {"video": 0}
    buffer_size = 0

    container = av.open(str(video_path))
    frames_length = container.streams.video[0].frames
    # how many time_base units
    duration = container.streams.video[0].duration
    timebase = int(duration/frames_length)
    pts_to_sample = timebase * finds_to_sample

    stream = container.streams.video[0]

    iter_pts_to_sample = iter([int(p) for p in pts_to_sample])
    next_pts = next(iter_pts_to_sample)
    # Initial seek
    seek_offset = max(next_pts - margin, 0)
    container.seek(seek_offset, any_frame=False,
            backward=True, stream=stream)
    if threading:
        container.streams.video[0].thread_type = 'AUTO'
    # Decode some
    buffer_count = -1
    pts_to_frames = {}
    pts_to_frames_packet = {}
    for packet in container.demux(**stream_name):
        if (next_pts - packet.pts) > skip_offset:
            continue
        for frame in packet.decode():
            pts_to_frames[frame.pts] = frame
            pts_to_frames_packet[frame.pts] = packet
        max_pts_to_frames = max(pts_to_frames, default=0)
        try:
            if max_pts_to_frames >= next_pts:
                next_pts = next(iter_pts_to_sample)
        except StopIteration:
            buffer_count += 1
        if buffer_count >= buffer_size:
            break
    container.close()

    pts_we_got = np.array(list(pts_to_frames.keys()))
    ssorted_indices = np.searchsorted(pts_we_got, pts_to_sample)

    sampled_frames = []
    for pts in pts_we_got[ssorted_indices]:
        sampled_frames.append(pts_to_frames[pts])

    sampled_frames_np = [frame.to_rgb().to_ndarray()
            for frame in sampled_frames]
    sampled_frames_np = np.stack(sampled_frames_np)
    return sampled_frames_np


def _perframe_detection_display(out, test_kfs, Y_conf_scores_sm, dataset):
    # Display our detections
    det2_fold = small.mkdir(out/'det2')
    state = np.random.RandomState(400)
    iter_index = state.permutation(np.arange(len(test_kfs)))[:400]
    for i, ii in enumerate(tqdm(iter_index)):
        kf = test_kfs[ii]
        scores = Y_conf_scores_sm[ii]
        vid = kf['vid']
        frame0 = kf['frame0']
        pred_box = kf['bbox']
        video_path = dataset.videos[vid]['path']
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, [frame0], debug_filename=video_path)
        frame_u8 = frames_u8[0]
        act_scores = pd.Series(dict(zip(dataset.action_names, scores)))
        act_scores = act_scores.sort_values(ascending=False)[:3]
        stract = ' '.join([f'{x[:5]}: {y:.2f}' for x, y in act_scores.items()])
        rec_color = (0, 0, 80)
        good = kf['action_name'] == act_scores.index[0]
        if good:
            rec_color = (0, 80, 0)
        snippets.cv_put_box_with_text(frame_u8, pred_box,
            text='{}'.format(stract), rec_color=rec_color)
        cv2.imwrite(str(det2_fold/f'{i:05d}_{vid}_frame{frame0:05d}.jpg'), frame_u8)


def _tube_detection_display(out, av_stubes_, dataset):
    action = 'Drinking'
    vfold = small.mkdir(out/'det3_tube'/action)
    v_stubes = av_stubes_[action]
    flat_tubes = []
    for vid, stubes in v_stubes.items():
        for i_stube, stube in enumerate(stubes):
            flat_tubes.append({'tube': stube, 'ind': (vid, i_stube)})
    sorted_flat_tubes = sorted(flat_tubes,
            key=lambda x: x['tube']['score'], reverse=True)
    sorted_flat_tubes = sorted_flat_tubes[:10]

    for i_sorted, flat_tube in enumerate(tqdm(sorted_flat_tubes)):
        vid, i_stube = flat_tube['ind']
        tube = flat_tube['tube']
        score = tube['score']
        sf, ef = tube['start_frame'], tube['end_frame']
        frame_inds = tube['frame_inds']
        video_fold = small.mkdir(vfold/f'{i_sorted:04d}_vid{vid}_{sf}_to_{ef}_score{score:02f}')
        video_path = dataset.videos[vid]['path']

        # Extract
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        # Draw
        drawn_frames_u8 = []
        for i, (find, frame_BGR) in enumerate(zip(frame_inds, frames_u8)):
            image = frame_BGR.copy()
            box = tube['boxes'][i]
            snippets.cv_put_box_with_text(image, box,
                text='{} {} {:.2f}'.format(
                    i, action, score))
            drawn_frames_u8.append(image)

        snippets.qsave_video(video_fold/'overlaid.mp4', drawn_frames_u8)
        break
