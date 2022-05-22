# encoding: utf-8
'''
2022-05-17
From: wufan - gsd
'''
import os
import sys
import time
import cv2
import numpy as np
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import mediapipe as mp
from copy import deepcopy
from dtw import dtw

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
all_mark_list = [16, 14, 12, 11, 13, 15]

watermark_fn = 'watermark.png'
template_start_fn = 'start.jpg'
template_end_fn = 'end.jpg'

# 用来给线程池用合并帧
def make_frame(frame_num, frame_list, rate, shape, video_frame, watermark, detail, score_line,
		area_shape, area_pos, area_step, area_scale, detail_list):
	less = 1
	b_c = g_c = r_c = a_c = np.zeros(shape).astype('uint8')
	bg_frame = cv2.merge((b_c, g_c, r_c, a_c))
	for frame in frame_list:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
		frame = frame * less * rate
		bg_frame = bg_frame + frame
		less = less - (less * rate)
		bg_frame = bg_frame.astype('uint8')
	# bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGRA2BGR)

	if detail:
		f_s = '%.4f'%np.std(score_line)
		f_m = np.mean(score_line)
		cv2.putText(bg_frame, f_s, (10, shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		detail_list.append((area_pos[0] + int(frame_num * area_step), area_pos[1] - int(f_m * area_scale)))
		bg_info = cv2.merge((b_c, g_c, r_c, a_c))
		for d in detail_list:
			cv2.circle(bg_info, d, 4, (255, 0, 0), cv2.FILLED)
	if watermark is not None:
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
		bg_frame = cv2.addWeighted(bg_frame, alpha=1, src2=watermark, beta=0.3, gamma=1)
		bg_frame = cv2.addWeighted(bg_frame, alpha=1, src2=bg_info, beta=0.5, gamma=1)
		bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGRA2BGR)

	video_frame.append((frame_num, bg_frame))
	return frame_num, bg_frame

# 过滤分割成多段
def filted_data(data_list):
	win_size = 30
	output_list = deepcopy(data_list)
	for idx in range(len(output_list)):
		tmp_list = output_list[idx: idx+win_size]
		tmp_list = [i[1] for i in tmp_list]
		mean = np.mean(tmp_list, axis=0)
		std = np.std(tmp_list, axis=0)
		floor = mean - 3*std
		upper = mean + 3*std
		for t, val in enumerate(tmp_list):
			value = float(np.where(((val>upper) | (val<floor)), mean, val))
			output_list[idx + t][1] = value
	total_mean = np.mean([i[1] for i in output_list])
	tmp = [list(g) for k, g in groupby(output_list, lambda x: x[1] > total_mean) if not k]
	output = []
	for i in tmp:
		s = i[0][0]
		e = i[-1][0]
		output.append(list(filter(lambda x: x[0] >= s and x[0] <= e, data_list)))
	tmp_value = [len(i) for i in tmp]
	value = np.mean(tmp_value) / 2
	output = list(filter(lambda x: len(x) > value, output))
	output = [sorted(g, key=lambda x: x[1])[0] for g in output]
	return output

# 获取关键点和相对参数
def getValueNp(pose_val, width, height):
	lm_list = []
	nose = pose_val.pose_landmarks.landmark[0]
	nose_x = nose.x * width
	nose_y = nose.y * height
	pos_list = [(int(round(nose_x)), int(round(nose_y)))]
	for idx in all_mark_list:
		lm = pose_val.pose_landmarks.landmark[idx]
		pos_list.append((int(round(lm.x * width)), int(round(lm.y * height))))
		lm_list.append([(lm.x * width) - nose_x, (lm.y * height) - nose_y])
	return pos_list, np.array(lm_list)

# DTW
def dtwFrameList(template, query):
	alignment = dtw(template, query, step_pattern='asymmetric', keep_internals=True)
	return alignment.index1, alignment.index2

def main(filename, line, sync, move, detail):
	st = time.time()
	cameraCapture = cv2.VideoCapture(filename)
	width = int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	area_width = int(width * 4 / 5)
	area_height = int(height * 1 / 4)
	area_pos = (int((width - area_width) / 2), int(height * 3 / 4))
	print(width, height)

	if os.path.isfile(watermark_fn):
		watermark = np.zeros((height, width, 4)).astype('uint8')
		wt = cv2.imread(watermark_fn, cv2.IMREAD_UNCHANGED)
		sca = max([wt.shape[0] / float(height), wt.shape[1] / float(width)])
		wt = cv2.resize(wt, (int(wt.shape[1] / sca), int(wt.shape[0] / sca)))
		watermark[int((height - wt.shape[0]) / 2): int((height + wt.shape[0]) / 2), int((width - wt.shape[1]) / 2): int((width + wt.shape[1]) / 2)] = wt
		watermark[(watermark[:,:,3] == 0), 0] = 0
		watermark[(watermark[:,:,3] == 0), 1] = 0
		watermark[(watermark[:,:,3] == 0), 2] = 0
	else:
		watermark = None

	start_img = cv2.cvtColor(cv2.imread(template_start_fn), cv2.COLOR_BGR2RGB)
	start_val = pose.process(start_img)
	base_pos_list, start_val = getValueNp(start_val, width, height)
	end_img = cv2.cvtColor(cv2.imread(template_end_fn), cv2.COLOR_BGR2RGB)
	end_val = pose.process(end_img)
	_, end_val = getValueNp(end_val, width, height)
	print(start_val)
	print(end_val)

	success = True
	score_list_start = []
	score_list_end = []
	frame_num = 0
	frame_move_dict = {}
	frame_line_dict = {}
	while success:
		success, frame = cameraCapture.read()
		if frame is None:
			continue
		frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pose_val = pose.process(frameRGB)
		if not pose_val.pose_landmarks:
			frame_move_dict[frame_num] = frame_move_dict.get(frame_num - 1, (0, 0))
			continue
		pos_list, pose_val = getValueNp(pose_val, width, height)
		if move:
			frame_move_dict[frame_num] = (base_pos_list[0][0] - pos_list[0][0], base_pos_list[0][1] - pos_list[0][1])
		if line:
			frame_line_dict[frame_num] = pos_list
		score = np.sqrt(np.sum(np.square(pose_val - start_val)))
		score_list_start.append([frame_num, score ** 2])
		score = np.sqrt(np.sum(np.square(pose_val - end_val)))
		score_list_end.append([frame_num, score ** 2])
		frame_num += 1
	score_list_start = np.array(score_list_start)
	score_list_end = np.array(score_list_end)
	np.savetxt('log_start.txt', score_list_start)
	np.savetxt('log_end.txt', score_list_end)
	score_list_start[score_list_start[:,1] > 100000, 1] = np.float64(100000)
	score_list_end[score_list_end[:,1] > 100000, 1] = np.float64(100000)
	start_list = filted_data(score_list_start)
	end_list = filted_data(score_list_end)
	print('start: %s; end: %s'%(len(start_list), len(end_list)))

	if len(start_list) != len(end_list):
		raise Exception('ERR_SIZE: %s-%s'%(len(start_list), len(end_list)))
	result = list(zip([i[0] for i in start_list], [i[0] for i in end_list]))
	print(len(result), time.time() - st)

	if sync:
		template_val = result.pop(0)
		frame_num_result = [list(range(int(template_val[0]), int(template_val[1])))]
		template = score_list_start[(score_list_start[:,0] >= template_val[0]) & (score_list_start[:,0] < template_val[1]), 1]
		score_val_result = [template]
		for s, e in result:
			tmp_idx = []
			tmp_val = []
			query = score_list_start[(score_list_start[:,0] >= s) & (score_list_start[:,0] < e), 1]
			_, index2 = dtwFrameList(template, query)
			for i in index2:
				tmp_idx.append(int(s + i))
				tmp_val.append(query[i])
			frame_num_result.append(tmp_idx)
			score_val_result.append(tmp_val)
		result = np.array(frame_num_result)
		score_val_result = np.array(score_val_result)
		result = result.T
		print('score:', np.std(score_val_result))
		score_val_result = score_val_result.T
	else:
		frame_num_result = []
		score_val_result = []
		max_frames = int(max([b - a for a, b in result]))
		for s, e in result:
			tmp_idx = []
			tmp_val = []
			for n in range(max_frames):
				f_n = s + n
				if f_n > e:
					f_n = e
				s_v = score_list_start[f_n]
				tmp_idx.append(f_n)
				tmp_val.append(s_v)
			frame_num_result.append(tmp_idx)
			score_val_result.append(tmp_val)
		result = np.array(frame_num_result)
		score_val_result = np.array(score_val_result)
		result = result.T
		score_val_result = score_val_result.T
	print('frame_num:', result.shape, len(frame_line_dict))

	area_step = area_width / result.shape[0]
	area_scale = area_height / 100000

	cameraCapture = cv2.VideoCapture(filename)

	# pool = ThreadPoolExecutor(max_workers=20)
	# all_work = []
	detail_list = []
	rate = 0.5
	video_frame = []
	for idx, val in enumerate(result):
		if idx % 30 == 0:
			print('frame: %s'%idx)
		# if idx == 60:
		# 	break
		frame_list = []
		for f_n in val:
			cameraCapture.set(cv2.CAP_PROP_POS_FRAMES, f_n)
			success, frame = cameraCapture.read()
			if line:
				fpl = frame_line_dict.get(f_n)
				if fpl:
					zl = zip(fpl[0:-1], fpl[1::])
					for pt1, pt2 in zl:
						cv2.line(frame, pt1, pt2, (0, 255, 0), 5)
					for pt in fpl:
						cv2.circle(frame, pt, 8, (0, 0, 255), cv2.FILLED)
			if move:
				imgShift = np.float32([[1, 0, frame_move_dict[f_n][0]],[0, 1, frame_move_dict[f_n][1]]])
				frame = cv2.warpAffine(frame, imgShift, (width, height))

			frame_list.append(frame)

		make_frame(idx, frame_list, rate, (height, width), video_frame, watermark, detail, score_val_result[idx], (area_height, area_width), area_pos, area_step, area_scale, detail_list)
		# work = pool.submit(make_frame, idx, frame_list, rate, (height, width), video_frame, watermark, detail, score_val_result[idx], (area_height, area_width), area_pos, area_step, area_scale)
		# all_work.append(work)
	# wait(all_work, return_when=ALL_COMPLETED)
	print(len(video_frame))
	video_frame = sorted(video_frame, key=lambda x:x[0])
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outfn = os.path.splitext(filename)
	outfn = outfn[0] + '_output' + outfn[1]
	videoWriter = cv2.VideoWriter(outfn, fourcc, 30.0, (width, height), True)
	for _, f in video_frame:
		videoWriter.write(f)
	videoWriter.release()


if __name__ == '__main__':
	st = time.time()
	line = False
	sync = False
	move = False
	detail = False
	video_fn = sys.argv[1]
	if len(sys.argv) == 3:
		if '1' in sys.argv[2]:
			line = True
		if '2' in sys.argv[2]:
			sync = True
		if '3' in sys.argv[2]:
			move = True
		if '4' in sys.argv[2]:
			detail = True
	main(video_fn, line, sync, move, detail)
	print(time.time() - st)