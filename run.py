import subprocess

import cv2
import numpy as np


def get_board_image(screenshot_file=None):
	if screenshot_file is None:
		adb = "./platform-tools/adb"
		pipe = subprocess.Popen(
			f"{adb} shell screencap -p",
       		stdin=subprocess.PIPE,
       	 	stdout=subprocess.PIPE, shell=True
    	)
		image_bytes = pipe.stdout.read()  # .replace(b'\r\n', b'\n')
		image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
	else:
		image = np.fromfile(screenshot_file, dtype=np.uint8
			).reshape((2400,1080))

	borders = np.nonzero(255 == image[:, image.shape[1] // 2])[0]
	top, bottom = borders[0], borders[-1]

	borders = np.nonzero(255 == image[(top + bottom) // 2, :])[0]
	left, right = borders[0], borders[-1]

	board = image[top:bottom, left:right]

	board -= np.min(board)
	board = board / np.max(board)

	return board

def show(image):
	image = cv2.resize(image, (540, 700))
	cv2.imshow("window", image)
	cv2.waitKey(0)
	cv2.destroyWindow("window")


def image_to_patches(board, rows, columns):
	border = 13
	board = board[border:-border, border:-border]
	h = board.shape[0] / rows
	w = board.shape[1] / columns

	out = []
	for y_i in range(0, rows):
		row = []
		for x_i in range(0, columns):
			y1, y2 = int(y_i * h), int((y_i + 1) * h)
			x1, x2 = int(x_i * w), int((x_i + 1) * w)
			patch = board[y1:y2, x1:x2]
			while patch.size and np.all(patch[0] == 1):
				patch = patch[1:]
			while patch.size and np.all(patch[-1] == 1):
				patch = patch[:-1]
			while patch.size and np.all(patch[:, 0] == 1):
				patch = patch[:, 1:]
			while patch.size and np.all(patch[:, -1] == 1):
				patch = patch[:, :-1]
			if patch.size == 0:
				patch = None
			else:
				patch = cv2.resize(patch, (100, 100))
			row.append(patch)
		out.append(row)

	return out


def create_number_references(file_descs):

	reference = np.zeros((8, 100, 100), dtype=np.float32)

	for (filename, n_rows, n_columns) in file_descs:
		image = get_board_image(filename)
		patches = image_to_patches(image, n_rows, n_columns)
		for row in patches:
			for patch in row:
				if patch is None:
					continue
				cv2.imshow("window", patch)
				key = cv2.waitKey(0)
				idx = key - ord('1')
				reference[idx] += patch

	for i in range(8):
		reference[i] -= np.min(reference[i])
		maximum = np.max(reference[i])
		if maximum > 0:
			reference[i] /= maximum
		cv2.imshow("labels", reference[i])
		cv2.waitKey(0)

	reference.tofile("ocr_reference.npy")
			

global ocr_reference
ocr_reference = None

def ocr(patch):
	if patch is None:
		return None

	global ocr_reference
	if ocr_reference is None:
		ocr_reference = np.fromfile(
			'ocr_reference.npy', 
			dtype=np.float32
		).reshape((8, 100, 100))

	min_error = 10001
	val = None
	for i, target in enumerate(ocr_reference):
		d = patch - target
		mse = np.sum(d * d)
		if mse < min_error:
			min_error = mse
			val = i + 1
	return val


def digitise(image, rows, columns):

	patch_grid = image_to_patches(image, rows, columns)
	for row in patch_grid:
		for patch in row:
			value = ocr(patch)
			print(value)
			if patch is not None:
				cv2.imshow("window", patch)
				cv2.waitKey(0)

			

# create_number_references([
# 	("10x7_2400-1080.npy", 10, 7),
# 	("10x7_2_2400-1080.npy", 10, 7),
# 	("13x9_2400-1080.npy", 13, 9),
# 	("14x10_2400-1080.npy", 14, 10),
# ])

board = get_board_image("14x10_2400-1080.npy")
cv2.imshow("board", cv2.resize(board, (540, 700)))
digitise(board, rows=14, columns=10)
# show(board)


