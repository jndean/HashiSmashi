from collections import namedtuple
import subprocess

import cv2
import numpy as np


Direction = namedtuple("Direction", ["y", "x"])

DIRECTIONS = (Direction(-1, 0), Direction(0, -1), Direction(1, 0), Direction(0, 1))

def neg(direction):
	return Direction(-direction.y, -direction.x)

def orthogonal(d1, d2):
	return not (d1.y * d2.y + d1.x * d2.x)



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
	grid = image_to_patches(image, rows, columns)
	for y in range(len(grid)):
		for x in range(len(grid[0])):
			grid[y][x] = ocr(grid[y][x])
	board = Board(grid)
	return board


class Contradiction(RuntimeError):
	pass


class Board:
	def __init__(self, values):
		self.height = len(values)
		self.width = len(values[0])
		self._build(values)

	def _build(self, values):
		self.grid = []
		self.clusters = set()
		for y, row in enumerate(values):
			out = []
			for x, value in enumerate(row):
				if value is None:
					out.append(Space())
				else:
					node = Node(value)
					self.clusters.add(node.cluster)
					out.append(node)
			self.grid.append(out)
		
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
		for d in DIRECTIONS:
			d_sum += node.slots[d]
			other_d_sum += other_node.slots[d]
		if (d_sum < node.remaining or 
			other_d_sum < other_node.remaining or
			(node.cluster.remaining == 0 and len(self.clusters) > 1)
		):
			raise Contradiction()

	def __contains__(self, point):
		return 0 <= point[0] < self.height and 0 <= point[1] < self.width

	def is_solved(self):
		return len(self.clusters) == 1 and next(iter(self.clusters)).remaining == 0

	def copy():
		pass

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
		cv2.waitKey(0)


class Space:

	def __repr__(self):
		return '0'


class Node:

	def __init__(self, clue):
		self.clue = clue
		self.remaining = clue
		self.slots = dict((k, min(2, clue)) for k in DIRECTIONS)
		self.cluster = Cluster(self)

	def __repr__(self):
		return f'{self.clue}'

	def connect(self, d):
		assert(self.remaining and self.slots[d])
		self.slots[d] -= 1
		self.remaining -= 1
		self.cluster.remaining -= 1
		for _d in DIRECTIONS:
			self.slots[_d] = min(self.slots[_d], self.remaining)


class Bridge:

	def __init__(self, d):
		self.d = Direction(int(bool(d.y)), int(bool(d.x)))
		self.weight = 1


class Cluster:

	def __init__(self, node):
		self.members = set([node])
		self.remaining = node.remaining

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
		return True
	# return exploratory_solver(board)

def simple_solver_iteration(board):
	new_bridge = False
	for y, x, node in board.get_unfinished_nodes():
		possible_ds = []
		total_slots = 0

		for d in DIRECTIONS:
			neighbour = board.get_node_in_direction(y, x, d)
			if neighbour is None:
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
				board.create_bridge(y, x, d)
				return

# create_number_references([
# 	("10x7_2400-1080.npy", 10, 7),
# 	("10x7_2_2400-1080.npy", 10, 7),
# 	("13x9_2400-1080.npy", 13, 9),
# 	("14x10_2400-1080.npy", 14, 10),
# ])

board_pixels = get_board_image("10x7_2400-1080.npy")
cv2.imshow("original", cv2.resize(board_pixels, (540, 700)))
board = digitise(board_pixels, rows=10, columns=7)
solve(board)
for c in board.clusters:
	print(c.members, c.remaining)
board.draw()


