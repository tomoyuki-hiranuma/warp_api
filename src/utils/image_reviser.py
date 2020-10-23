# -*- coding: utf-8 -*-
import numpy as np
import cv2, math
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




#===================================================
print('### 関数')
#===================================================
# def show(img):
#   plt.figure(figsize=(8, 8))
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	plt.imshow(img, vmin=0, vmax=255)
# 	plt.gray()
# 	plt.show()
# 	plt.close()
# 	print()

class ImageReviser():
	def __init__(self, img, paper_long, paper_short, point_x, point_y):
		#可変
		self.img = img#cv2.imread(img)
		self.paper_long = paper_long
		self.paper_short = paper_short
		self.point_x = point_x
		self.point_y = point_y
		
		#基本的に固定
		self.size_max = 1000 #画像の長辺
		self.r_inter = int(self.size_max * 0.012) #色検出器の半径
		self.r_outer = int(self.size_max * 0.4) #色検出器の半径
		
		#初期化
		self.mm_per_px = 0
		
		#円形フィルタ用意
		self.f_circle = np.zeros((self.r_inter*2 - 1, self.r_inter*2 - 1), dtype=bool)
		for i in range(len(self.f_circle)):
			for j in range(len(self.f_circle[0])):
				#注目するピクセルから中心までの距離
				r = ((i + 1 - self.r_inter)**2 + (j + 1 - self.r_inter)**2)**0.5
				#どの範囲に属するか調べて波の高さを更新
				if r < self.r_inter:
					self.f_circle[i, j] = True
		print('f_circle.shape = {}'.format(self.f_circle.shape))
    
    #ビームの描画
	def draw_beam(self, img, x_0, y_0, radians, lengths):
		img_tmp = deepcopy(img)
		for i in range(len(radians)):
			radian = radians[i]
			#線
			p_x = x_0 + math.cos(radian) * self.r_outer
			p_y = y_0 + math.sin(radian) * self.r_outer
			cv2.line(img_tmp, (int(p_x), int(p_y)), (x_0, y_0), (0, 255, 0), 2)
			#点
			p_x = x_0 + math.cos(radian) * lengths[i]
			p_y = y_0 + math.sin(radian) * lengths[i]
			cv2.circle(img_tmp, (int(p_x), int(p_y)), 6, (0, 0, 255), 2)
		return img_tmp
    
    #ビーム上の色を取得する関数（p2=(y, x)からp1=(y, x)に向かう配列） 返り値の長さは角度による
	def bresenham_march(self, p1, p2):
		x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
		
		#点が画像外なら終了
		if x1 >= self.img.shape[0] or x2 >= self.img.shape[0] or y1 >= self.img.shape[1] or y2 >= self.img.shape[1]:
			#tests if line is in image, necessary because some part of the line must be inside, it respects the case that the two points are outside
			if not cv2.clipLine((0, 0, *self.img.shape), p1, p2):
				print('not in region')
				return
		
		steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
		if steep:
			x1, y1 = y1, x1
			x2, y2 = y2, x2

		# takes left to right
		also_steep = x1 > x2
		if also_steep:
			x1, x2 = x2, x1
			y1, y2 = y2, y1

		dx = x2 - x1
		dy = math.fabs(y2 - y1)
		error = 0.0
		delta_error = 0.0
		# Default if dx is zero
		if dx != 0:
			delta_error = math.fabs(dy / dx)

		y_step = 1 if y1 < y2 else -1

		y = y1
		ret = []
		for x in range(x1, x2):
			p = (y, x) if steep else (x, y)
			if p[0] < self.img.shape[0] and p[1] < self.img.shape[1]:
				#ret.append((p, self.img[p]))
				ret.append(self.img[p])
			error += delta_error
			if error >= 0.5:
				y += y_step
				error -= 1
		if also_steep:  # because we took the left to right instead
			ret.reverse()
		return np.array(ret)
    
	#白でなくなる距離を返す関数
	def get_white_len(self, colors, mean_color, radian):
		l = 0
		#平均値*0.9よりも明るいインデックス
		diff = np.sum(colors, axis=1) - np.sum(mean_color)*0.9
		ok = np.where(diff>0)[0]
		#手前から連続している長さ
		zeros = ok - np.arange(0, len(ok))
		l = np.where(zeros==0)[0][-1]
		
		#調整
		l += 2

		#角度によって長さを補正して返す
		if radian < (1/4)*math.pi:
			return l / math.cos(radian)
		elif (1/4)*math.pi < radian < (3/4)*math.pi:
			return l / math.sin(radian)
		elif (3/4)*math.pi < radian < (5/4)*math.pi:
			return l / -math.cos(radian)
		elif (5/4)*math.pi < radian < (7/4)*math.pi:
			return l / -math.sin(radian)
		elif (7/4)*math.pi < radian:
			return l / math.cos(radian)
		else:
			return l
    
	#全ビーム上の紙範囲の長さを走査
	def get_white_lens(self, x_0, y_0, radians, mean_color):
		#
		lengths = np.zeros(len(radians), dtype=int)
		for i in range(len(radians)):
			radian = radians[i]
			#行き先の点
			length = self.r_outer
			p_x = x_0 + math.cos(radian) * length
			p_y = y_0 + math.sin(radian) * length
			#紙からはみ出ないように調整
			while p_x < 0 or p_y < 0 or p_x > self.img.shape[1] or p_y > self.img.shape[0]:
				length -= 2
				p_x = x_0 + math.cos(radian) * length
				p_y = y_0 + math.sin(radian) * length
			#紙からはみ出ないように調整（最大値を取ることがあるので）
			p_x = min(p_x, self.img.shape[1]-1)
			p_y = min(p_y, self.img.shape[0]-1)
			#ビーム上の色（後ろから前に向かう配列）
			colors = self.bresenham_march((y_0, x_0), (int(p_y), int(p_x)))
			#紙の範囲の長さを取得
			lengths[i] = self.get_white_len(colors, mean_color, radian)
		return lengths
    
	def run(self):
		try:
			print('================== start =====================')
			
			#===================================================
			print('### 準備')
			#===================================================
			#サイズ取得
			print(self.img.shape[1])
			h_img = self.img.shape[0]
			w_img = self.img.shape[1]
			print('size(w, h) = ({}, {})'.format(w_img, h_img))
			
			#目標サイズにリサイズ
			if h_img < w_img:
				w_window = self.size_max
				h_window = int(h_img * self.size_max / w_img)
			else:
				h_window = self.size_max
				w_window = int(w_img * self.size_max / h_img)
			self.img = cv2.resize(self.img, (w_window, h_window), interpolation = cv2.INTER_CUBIC)
			print('resized(w, h) = ({}, {})'.format(w_window, h_window))
			
			#描画用にコピー
#            img_draw = deepcopy(self.img)
			
			
			#===================================================
			print('### メイン')
			#===================================================
			
			#show(self.img)
			
			
			#座標の取得
			x_0, y_0 = self.point_x, self.point_y
			
			#内円の色平均を取得
			scope = self.img[y_0 - self.r_inter + 1:y_0 + self.r_inter, x_0 - self.r_inter + 1:x_0 + self.r_inter]
			mean_color = np.mean(scope[self.f_circle==True], axis=0)
			print('mean_color = {}'.format(mean_color))
			
			#初期角度
			radians = np.arange(0*math.pi, 2.0*math.pi, 0.1)
			
			#tmp表示
#            lengths = self.get_white_lens(x_0, y_0, radians, mean_color)
#            img_tmp = self.draw_beam(img_draw, x_0, y_0, radians, lengths)
#            #cv2.imwrite('save/{}.png'.format(0), img_tmp)
#            show(img_tmp)
			
			steps = 20
			for i in range(steps):
				print(':', end='')
				#全ビーム上の紙範囲の長さを調べる
				lengths = self.get_white_lens(x_0, y_0, radians, mean_color)
				#保存
				radians_save = deepcopy(radians)
				lengths_save = deepcopy(lengths)
				#ずらす、pi/8から徐々に小さくする
				#delta = nr.rand(len(radians)) * (1/4)*math.pi - (1/8)*math.pi
				#delta = nr.rand(len(radians)) * (1/8)*math.pi
				delta = np.ones(len(radians)) * (1/8)*((steps**0.5 - i**0.5)/steps**0.5)*math.pi
				
				#正方向にずらして調べる
				radians += delta
				radians[radians>2.0*math.pi] -= 2.0*math.pi
				radians[radians<0.0] += 2.0*math.pi
				#全ビーム上の紙範囲の長さを調べる
				lengths = self.get_white_lens(x_0, y_0, radians, mean_color)
				#紙の長さが長くなった？
				ps = lengths > lengths_save
				
				#負方向にずらして調べる
				radians -= delta * 2
				radians[radians>2.0*math.pi] -= 2.0*math.pi
				radians[radians<0.0] += 2.0*math.pi
				#全ビーム上の紙範囲の長さを調べる
				lengths = self.get_white_lens(x_0, y_0, radians, mean_color)
				#紙の長さが長くなった？
				ms = lengths > lengths_save
				
				#更新
				#正方向で更新したもの
				radians_save[ps] += delta[ps]
				#負方向で更新したもの
				radians_save[~ps*ms] -= delta[~ps*ms]
				#更新の反映
				radians = deepcopy(radians_save)
				radians[radians>2.0*math.pi] -= 2.0*math.pi
				radians[radians<0.0] += 2.0*math.pi
				
				#全ビーム上の紙範囲の長さを調べる
				lengths = self.get_white_lens(x_0, y_0, radians, mean_color)
				
				#tmp表示
#                lengths = self.get_white_lens(x_0, y_0, radians, mean_color)
#                img_tmp = self.draw_beam(img_draw, x_0, y_0, radians, lengths)
#                #cv2.imwrite('save/{}.png'.format(i+1), img_tmp)
#                show(img_tmp)
			print()
			
			#点群を配列に格納
			p_s = np.zeros((len(radians), 2))
			for i in range(len(radians)):
				radian = radians[i]
				#点
				p_x = x_0 + math.cos(radian) * lengths[i]
				p_y = y_0 + math.sin(radian) * lengths[i]
				p_s[i, 0] = p_x
				p_s[i, 1] = p_y
			#print(p_s)
			
			#============================
			# k-means
			#============================
			#k-meansのオブジェクト
			km = KMeans(n_clusters=4)
			#計算を実行
			km.fit(p_s)
			#各クラスタの重心
			km_centers = km.cluster_centers_
			print('km_centers\n{}'.format(km_centers))
			
			#中心
			center = np.mean(km_centers, axis=0)
			print(print('center = {}'.format(center)))
			
			#辺を検出
			x_large = np.mean(km_centers[:, 0][km_centers[:, 0]>center[0]])
			x_small = np.mean(km_centers[:, 0][km_centers[:, 0]<center[0]])
			y_large = np.mean(km_centers[:, 1][km_centers[:, 1]>center[1]])
			y_small = np.mean(km_centers[:, 1][km_centers[:, 1]<center[1]])
			
			#紙が横向きの時
			if (x_large - x_small) > (y_large - y_small):
				for i in range(4):
					if km_centers[i, 0] < center[0] and km_centers[i, 1] > center[1]:
						x_0, y_0 = km_centers[i]
					if km_centers[i, 0] < center[0] and km_centers[i, 1] < center[1]:
						x_1, y_1 = km_centers[i]
					if km_centers[i, 0] > center[0] and km_centers[i, 1] < center[1]:
						x_2, y_2 = km_centers[i]
					if km_centers[i, 0] > center[0] and km_centers[i, 1] > center[1]:
						x_3, y_3 = km_centers[i]
			#紙が縦向きの時
			else:
				for i in range(4):
					if km_centers[i, 0] < center[0] and km_centers[i, 1] > center[1]:
						x_3, y_3 = km_centers[i]
					if km_centers[i, 0] < center[0] and km_centers[i, 1] < center[1]:
						x_0, y_0 = km_centers[i]
					if km_centers[i, 0] > center[0] and km_centers[i, 1] < center[1]:
						x_1, y_1 = km_centers[i]
					if km_centers[i, 0] > center[0] and km_centers[i, 1] > center[1]:
						x_2, y_2 = km_centers[i]
			
			#線で閉じる
			cv2.line(self.img, (int(x_0), int(y_0)), (int(x_1), int(y_1)), (0, 255, 0))
			cv2.line(self.img, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 255, 0))
			cv2.line(self.img, (int(x_2), int(y_2)), (int(x_3), int(y_3)), (0, 255, 0))
			cv2.line(self.img, (int(x_3), int(y_3)), (int(x_0), int(y_0)), (0, 255, 0))
			
			#座標を記入
			cv2.circle(self.img, (self.point_x, self.point_y), 5, (255, 0, 0))
			
			
			###################################
			# 以降同じ
			###################################
			#x_0, y_0 = km_centers[3]
			#x_1, y_1 = km_centers[2]
			#x_2, y_2 = km_centers[1]
			#x_3, y_3 = km_centers[0]
			
			#x_0, y_0 = (122, 471)
			#x_1, y_1 = (314, 482)
			#x_2, y_2 = (306, 708)
			#x_3, y_3 = (129, 700)
			
			# 対応するポイント（前）（左上、右上、左下、右下）
			pts1 = np.float32([[x_0, y_0],
													[x_1, y_1],
													[x_3, y_3],
													[x_2, y_2]])
			# 対応するポイント（後）（正方形）
			a, b = 2000, 2200
			pts2 = np.float32([[a, a], [b, a], [a, b], [b, b]])
			# 透視変換の行列を求める
			M = cv2.getPerspectiveTransform(pts1, pts2)
			# 変換行列を用いて画像の透視変換
			c = b + a
			self.img = cv2.warpPerspective(self.img, M, (c, c))
			
			#用紙の比率に伸ばす
			self.img = cv2.resize(self.img, (c, int(c*paper_long/paper_short)), interpolation = cv2.INTER_CUBIC)
			self.mm_per_px = paper_short / 200
			#print('mm_per_px:{}'.format(self.mm_per_px))
			
			#黒い余白を検出
			img_array = np.array(self.img)
			print('shape = {}'.format(img_array.shape))
			print(img_array.shape)
			notblack = np.where(img_array > 0)
			top = np.min(notblack[0])
			bottom = np.max(notblack[0])
			left = np.min(notblack[1])
			right = np.max(notblack[1])
			print('edge = ({}, {}, {}, {})'.format(top, bottom, left, right))
			#余白カット
			self.img = self.img[top:bottom, left:right]
			
			#表示サイズを調整
			h_img, w_img = self.img.shape[0], self.img.shape[1]
			print('size = ({}, {})'.format(w_img, h_img))
			
			#目標サイズにリサイズ
			if h_img < w_img:
				w_window = self.size_max
				h_window = int(h_img * self.size_max / w_img)
				self.mm_per_px *= (w_img / self.size_max)
			else:
				h_window = self.size_max
				w_window = int(w_img * self.size_max / h_img)
				self.mm_per_px *= (h_img / self.size_max)
			self.img = cv2.resize(self.img, (w_window, h_window), interpolation = cv2.INTER_CUBIC)
			print('resized = ({}, {})'.format(w_window, h_window))
			
			print('=================== success ======================')
				
		except:
			print('=================== error ======================')
			pass

	def get_mm_per_px(self):
		return self.mm_per_px
    
	def get_img(self):
		return self.img