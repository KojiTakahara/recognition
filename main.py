import base64
import cv2
import io
import os
import math
import numpy as np
import shutil

from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
from geojson import Point, Polygon, Feature
from google.cloud import vision
from pathlib import Path
from turfpy.measurement import boolean_point_in_polygon

app = FastAPI()
client = vision.ImageAnnotatorClient()
thresh = 50
N = 11

origins = [
	"http://127.0.0.1:8000",
	"http://localhost:8000",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

@app.get("/")
def read_root():
	return {"Hello": "World"}

@app.post("/post")
def convert(file: str = Form(...), x: int = Form(...), y: int = Form(...)):
	img_binary = base64.urlsafe_b64decode(base64_decode(file))
	img_np = np.frombuffer(img_binary, dtype=np.uint8)
	image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
	image_urls = get_image_urls(image, x, y)
	return {"urls": image_urls}

# ベクトル間の角度の余弦(コサイン)を見つけます
# pt0-pt1およびpt0-pt2のなす角のコサインを取得
def angle(pt1, pt2, pt0) -> float:
	dx1 = float(pt1[0,0] - pt0[0,0])
	dy1 = float(pt1[0,1] - pt0[0,1])
	dx2 = float(pt2[0,0] - pt0[0,0])
	dy2 = float(pt2[0,1] - pt0[0,1])
	v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
	return (dx1*dx2 + dy1*dy2) / v

# 画像上で検出された一連の正方形を返します。
def find_squares(image, squares, areaThreshold=1000):
	squares.clear()
	gray0 = np.zeros(image.shape[:2], dtype=np.uint8)

	# down-scale and upscale the image to filter out the noise
	rows, cols, _channels = map(int, image.shape)
	pyr = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
	timg = cv2.pyrUp(pyr, dstsize=(cols, rows))
	# 画像のBGRの色平面で正方形を見つける
	for c in range(0, 3):
		cv2.mixChannels([timg], [gray0], (c, 0))
		# いくつかのしきい値レベルを試す
		for l in range(0, N):
			# l:ゼロしきい値レベルの代わりにCannyを使用します。
			# Cannyはグラデーションシェーディングで正方形を
			# キャッチするのに役立ちます
			if l == 0:
				# Cannyを適用
				# スライダーから上限しきい値を取得し、下限を0に設定します
				# （これによりエッジが強制的にマージ）
				gray = cv2.Canny(gray0,thresh, 5)
				#Canny出力を拡張して、エッジセグメント間の潜在的な穴を削除します
				gray = cv2.dilate(gray, None)
			else:
				# apply threshold if l!=0:
				gray[gray0 >= (l+1)*255/N] = 0
				gray[gray0 < (l+1)*255/N] = 255

			# 輪郭をリストで取得
			contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for i, cnt in enumerate(contours):
				# 輪郭の周囲を取得
				arclen = cv2.arcLength(cnt, True)
				# 輪郭の近似
				approx = cv2.approxPolyDP(cnt, arclen*0.02, True)
				# 面積
				area = abs(cv2.contourArea(approx))
				#長方形の輪郭は、近似後に4つの角をもつ、
				#比較的広い領域
				#（ノイズの多い輪郭をフィルターで除去するため）
				#凸性(isContourConvex)になります。
				if approx.shape[0] == 4 and area > areaThreshold and cv2.isContourConvex(approx) :
					maxCosine = 0
					for j in range(2, 5):
						# ジョイントエッジ間の角度の最大コサインを見つけます
						cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
						maxCosine = max(maxCosine, cosine)
					# すべての角度の余弦定理が小さい場合（すべての角度が約90度）、
					# 結果のシーケンスにquandrange頂点を書き込みます
					if maxCosine < 0.3 :
						squares.append(approx)

def calculate_width_height(pts, add):
	top_left_cood = pts[0 + add][0]
	bottom_left_cood = pts[1 + add][0]
	bottom_right_cood = pts[2 + add][0]
	width = np.int(np.linalg.norm(bottom_left_cood - bottom_right_cood))
	height = np.int(np.linalg.norm(top_left_cood - bottom_left_cood))
	return width, height

def is_inside(x: int, y: int, pts) -> bool:
	point = Feature(geometry=Point((x, y)))
	polygon = Polygon([
		[
			(pts[0][0][0].astype(float), pts[0][0][1].astype(float)),
			(pts[1][0][0].astype(float), pts[1][0][1].astype(float)),
			(pts[2][0][0].astype(float), pts[2][0][1].astype(float)),
			(pts[3][0][0].astype(float), pts[3][0][1].astype(float))
		]
	])
	return boolean_point_in_polygon(point, polygon)

def base64_decode(data):
	if len(data) % 4:
		data += '=' * (4 - len(data) % 4)
	imgstr = data.split(';base64,')[1]
	return imgstr

def cut(image, p_cnt):
	x, y, w, h = cv2.boundingRect(p_cnt)
	img_half_width = x + w / 2
	# cv2.arcLength: 輪郭の周囲長, Trueは輪郭が閉じているという意味
	# cv2.approPolyDP: 検出した形状の近似
	epsilon = 0.1 * cv2.arcLength(p_cnt, True)
	approx = cv2.approxPolyDP(p_cnt, epsilon, True)
	# 射影変換
	try:
		pts1 = np.float32(approx)
		if pts1[0][0][0] < img_half_width: # ptsに格納されてある座標の始点が左上だったら
			width, height = calculate_width_height(pts1, 0)
			pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
		else:
			width, height = calculate_width_height(pts1, 1)
			pts2 = np.float32([[width, 0], [0, 0], [0, height], [width, height]])
	except IndexError as e:
		print('{}'.format(e))
		return
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(image, M, (width, height))
	return dst

def get_similar_images(content):
	image = vision.Image(content=content)
	response = client.web_detection(image=image)
	annotations = response.web_detection
	image_urls = []

	if annotations.pages_with_matching_images:
		for page in annotations.pages_with_matching_images:
			if page.full_matching_images:
				for image in page.full_matching_images:
					image_urls.append(image.url)
			if page.partial_matching_images:
				for image in page.partial_matching_images:
					image_urls.append(image.url)

	if annotations.visually_similar_images:
		for image in annotations.visually_similar_images:
			image_urls.append(image.url)

	if response.error.message:
		raise Exception(
			'{}\nFor more info on error messages, check: '
			'https://cloud.google.com/apis/design/errors'.format(
				response.error.message))
	return image_urls

def get_image_urls(image, x: int, y: int):
	squares = []
	if image is None:
		return []
	cv2.imwrite("temp.png", image)
	# 四角の角を見つける
	find_squares(image, squares)

	width = 1000
	height = 1000

	for s in squares:
		epsilon = 0.1 * cv2.arcLength(s, True)
		approx = cv2.approxPolyDP(s, epsilon, True)
		pts = np.float32(approx)
		if len(pts) != 4:
			continue
		if is_inside(x, y, pts):
			# 四角をカットする
			cut_img = cut(image, s)
			if cut_img is None:
				continue
			h, w, _ = cut_img.shape[:3]
			if w < width and h < height:
				cv2.imwrite("cutImage.png", cut_img)
				height = h
				width = w

	with io.open("cutImage.png", 'rb') as image_file:
		byte = image_file.read()
		os.remove('cutImage.png')

	return get_similar_images(byte)
