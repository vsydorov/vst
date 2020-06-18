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
