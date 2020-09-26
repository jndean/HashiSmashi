from collections import namedtuple
import subprocess

import cv2
import numpy as np


adb = "./platform-tools/adb"


Direction = namedtuple("Direction", ["y", "x"])
DIRECTIONS = (Direction(-1, 0), Direction(0, -1), Direction(1, 0), Direction(0, 1))

def neg(direction):
	return Direction(-direction.y, -direction.x)

def orthogonal(d1, d2):
	return not (d1.y * d2.y + d1.x * d2.x)


def get_board_image(screenshot_file=None):
	if screenshot_file is None:
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

	return image


def image_to_patches(image, rows, columns):

	# Crop the board, trim some padding and normalise the values
	trim = 13
	borders = np.nonzero(255 == image[:, image.shape[1] // 2])[0]
	top, bottom = borders[0] + trim, borders[-1] - trim
	borders = np.nonzero(255 == image[(top + bottom) // 2, :])[0]

	left, right = borders[0] + trim, borders[-1] - trim
	image = image[top:bottom, left:right]
	image -= np.min(image)
	image = image / np.max(image)

	h = image.shape[0] / rows
	w = image.shape[1] / columns

	# Crop out and trim each grid cell
	out = []
	for y_i in range(0, rows):
		row = []
		for x_i in range(0, columns):
			y1, y2 = int(y_i * h), int((y_i + 1) * h)
			x1, x2 = int(x_i * w), int((x_i + 1) * w)
			patch = image[y1:y2, x1:x2]
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

	return out, top, left, h, w


def create_number_references(file_descs):

	reference = np.zeros((8, 100, 100), dtype=np.float32)

	for (filename, n_rows, n_columns) in file_descs:
		image = get_board_image(filename)
		patches, *_ = image_to_patches(image, n_rows, n_columns)
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


class Contradiction(RuntimeError):
	pass


class Board:
	def __init__(self, image, rows, columns):
		self._build(image, rows, columns)

	def _build(self, image, rows, columns):
		patches, top, left, h, w = image_to_patches(image, rows, columns)

		self.grid = []
		self.clusters = set()
		self.bridge_list = []
		self.height = len(patches)
		self.width = len(patches[0])
		self.board_top, self.board_left = top, left
		self.cell_h, self.cell_w = h, w

		for y in range(len(patches)):
			self.grid.append([])
			for x in range(len(patches[0])):
				value = ocr(patches[y][x])
				if value is None:
					self.grid[-1].append(Space())
				else:
					node = Node(value)
					self.clusters.add(node.cluster)
					self.grid[-1].append(node)
		
	def get_unfinished_nodes(self):
		for y, row in enumerate(self.grid):
			for x, cell in enumerate(row):
				if isinstance(cell, Node) and cell.remaining > 0:
					yield y, x, cell

	def get_node_in_direction(self, y, x, d):
		while 1:
			y += d.y
			x += d.x
			if (y, x) not in self:
				return None
			square = self.grid[y][x]
			if isinstance(square, Bridge):
				if orthogonal(square.d, d):
					return None
				continue
			elif isinstance(square, Space):
				continue
			return square

	def create_bridge(self, y, x, d):
		self.bridge_list.append((y, x, d))

		node = self.grid[y][x]

		# Find the other node
		while 1:
			y += d.y
			x += d.x
			if (y, x) not in self:
				return
			square = self.grid[y][x]

			if isinstance(square, Bridge):
				assert(
					abs(d.y) == square.d.y and 
					abs(d.x) == square.d.x and 
					square.weight == 1
				)
				square.weight = 2
				continue

			if isinstance(square, Space):
				self.grid[y][x] = Bridge(d)
				continue

			other_node = square
			break

		# Join the nodes with a bridge
		node.connect(d)
		other_node.connect(neg(d))

		# Update the clusters
		if node.cluster is not other_node.cluster:
			self.clusters.remove(other_node.cluster)
			node.cluster.consume(other_node.cluster)

		# Check for contradictions
		d_sum, other_d_sum = 0, 0
		for _d in DIRECTIONS:
			d_sum += node.slots[_d]
			other_d_sum += other_node.slots[_d]

		if (d_sum < node.remaining or 
			other_d_sum < other_node.remaining or
			(node.cluster.remaining == 0 and len(self.clusters) > 1)
		):
			raise Contradiction()


	def __contains__(self, point):
		return 0 <= point[0] < self.height and 0 <= point[1] < self.width

	def is_solved(self):
		return len(self.clusters) == 1 and next(iter(self.clusters)).remaining == 0

	def copy(self):
		copy_dict = {}

		board = Board.__new__(Board)
		board.height = self.height
		board.width = self.width
		board.grid = [[item.copy(copy_dict) for item in row] for row in self.grid]
		board.clusters = set([c.copy(copy_dict) for c in self.clusters])
		board.bridge_list = list(self.bridge_list)
		board.board_top, board.board_left = self.board_top, self.board_left
		board.cell_h, board.cell_w = self.cell_h, self.cell_w

		return board

	def draw(self):
		rad = 40
		h, w = rad * self.height, rad * self.width
		canvas = np.ones((h, w), dtype=np.uint8) * 255

		font = cv2.FONT_HERSHEY_SIMPLEX

		for y_i, row in enumerate(self.grid):
			for x_i, item in enumerate(row):
				y = int((y_i + 0.5) * rad)
				x = int((x_i + 0.5) * rad)

				if isinstance(item, Space):
					continue
				elif isinstance(item, Node):
					canvas = cv2.circle(canvas, (x, y), int(rad / 2.2), 0, 2)
					canvas = cv2.putText(canvas, str(item), (x-10, y+10), font, 1, 0, 2)
					continue

				# Otherwise, Bridge
				for i in range(item.weight):
					offset = 3 * (2 * i - 1) * (item.weight - 1)
					start_x = x - 0.5 * item.d.x * rad + offset * item.d.y
					start_y = y - 0.5 * item.d.y * rad + offset * item.d.x
					start = (int(start_x), int(start_y))
					end = (int(start[0] + item.d.x * rad), int(start[1] + item.d.y * rad))
					canvas = cv2.line(canvas, start, end, 0, 2)

		cv2.imshow("board", canvas)


class Copyable:
	def copy(self, copy_dict):
		if self in copy_dict:
			return copy_dict[self]
		copy = type(self).__new__(type(self))
		copy_dict[self] = copy
		self._populate_copy(copy, copy_dict)
		return copy

	def _copy(self, copy_dict):
		raise NotImplementedError


class Space(Copyable):

	def __repr__(self):
		return '0'

	def _populate_copy(self, copy, copy_dict):
		pass


class Node(Copyable):

	def __init__(self, clue):
		self.clue = clue
		self.remaining = clue
		self.slots = dict((k, min(2, clue)) for k in DIRECTIONS)
		self.cluster = Cluster(self)

	def _populate_copy(self, copy, copy_dict):
		copy.clue = self.clue
		copy.remaining = self.remaining
		copy.slots = dict(self.slots)
		copy.cluster = self.cluster.copy(copy_dict)

	def __repr__(self):
		return f'{self.clue}'

	def connect(self, d):
		if not (self.remaining and self.slots[d]):
			raise Contradiction()
		assert(self.remaining and self.slots[d])
		self.slots[d] -= 1
		self.remaining -= 1
		self.cluster.remaining -= 1
		for _d in DIRECTIONS:
			self.slots[_d] = min(self.slots[_d], self.remaining)

	def connect_to(self, other, d):
		assert(
			self.remaining and self.slots[d] and
			other.remaining and other.slots[d]
		)
		neg_d = neg(d)

		self.slots[d] -= 1
		other.slots[neg_d] -= 1
		self.remaining -= 1
		other.remaining -= 1
		self.cluster.remaining -= 1
		other.cluster.remaining -= 1

		self.slots[d] = min(self.slots[d], self.remaining, other.slots[neg_d], other.remaining)
		other.slots[neg_d] = self.slots[d]

		for _d in DIRECTIONS:
			neg__d = neg(_d)
			self.slots[_d] = min(self.slots[_d], self.remaining)        
			other.slots[neg__d] = min(other.slots[neg__d], other.remaining)


class Bridge(Copyable):

	def __init__(self, d):
		self.d = Direction(int(bool(d.y)), int(bool(d.x)))
		self.weight = 1

	def _populate_copy(self, copy, copy_dict):
		copy.d = self.d
		copy.weight = self.weight


class Cluster(Copyable):

	def __init__(self, node):
		self.members = set([node])
		self.remaining = node.remaining

	def _populate_copy(self, copy, copy_dict):
		copy.members = set([node.copy(copy_dict) for node in self.members])
		copy.remaining = self.remaining

	def __contains__(self, x):
		return x in self.members

	def consume(self, other):
		assert(other is not self)
		for node in other.members:
			self.members.add(node)
			self.remaining += node.remaining
			node.cluster = self
		del other



def solve(board):
	while simple_solver_iteration(board):
		pass
	if board.is_solved():
		return board
	return exploratory_solver(board)

def simple_solver_iteration(board):
	new_bridge = False
	for y, x, node in board.get_unfinished_nodes():
		possible_ds = []
		total_slots = 0

		for d in DIRECTIONS:
			neighbour = board.get_node_in_direction(y, x, d)
			if neighbour is None:
				node.slots[d] = 0
				continue
			possible_ds.append(d)
			node.slots[d] = min(node.slots[d], neighbour.slots[neg(d)])
			total_slots += node.slots[d]

		for d in possible_ds:
			if total_slots - node.slots[d] < node.remaining:
				board.create_bridge(y, x, d)
				new_bridge = True
	return new_bridge

def exploratory_solver(board):
	for y, x, node in board.get_unfinished_nodes():
		for d in DIRECTIONS:
			neighbour = board.get_node_in_direction(y, x, d)
			if neighbour is None:
				continue
			if node.slots[d] and neighbour.slots[neg(d)]:
				copy = board.copy()
				copy.create_bridge(y, x, d)
				try:
					copy = solve(copy)
					return copy
				except Contradiction:
					pass

	raise Contradiction()


def send_solution_to_device(board):
	swipe_length = 50  # pixels
	swipe_time = str(30)  # ms
	instructions = []
	for y, x, d in board.bridge_list:
		y1 = int((y + 0.5) * board.cell_h + board.board_top)
		x1 = int((x + 0.5) * board.cell_w + board.board_left)
		y2 = y1 + swipe_length * d.y
		x2 = x1 + swipe_length * d.x
		instructions.append(f"input swipe {x1} {y1} {x2} {y2} {swipe_time}")

	cmd = [adb, "shell", "; ".join(instructions)] 
	subprocess.run(cmd, check=True)



if __name__ == "__main__":

	# create_number_references([
	# 	("10x7_2400-1080.npy", 10, 7),
	# 	("10x7_2_2400-1080.npy", 10, 7),
	# 	("13x9_2400-1080.npy", 13, 9),
	# 	("14x10_2400-1080.npy", 14, 10),
	# ])

	board_pixels = get_board_image()  # "13x9_2400-1080.npy")
	board = Board(board_pixels, rows=13, columns=9)
	board = solve(board)
	board.draw()
	cv2.waitKey(1)
	send_solution_to_device(board)
	cv2.waitKey(0)

	# subprocess.run([adb, "shell", "input", "swipe", "500", "950", "500", "1000", "30"], shell=True) 

