# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:35:11 2017

@author: shingo

深層学習用の画像データを1枚生成するクラス
TrainImageGenerator
"""

import cv2
import numpy as np
import random
import os
import csv
import imgtool as it

class TrainImageGenerator:
    """ 深層学習用の画像データを1枚生成するクラス """
    
    
    def __init__(self):
        
        """ TrainImageGeneratorのコンストラクタ """
        self.size = 256   # 出力画像サイズ
        self.maxAspect = 2.0   # 最大アスペクト比, 最小は逆数
        self.minHeightUpperShowArea = 0.3 # 上部最小表示領域率
        self.minHeightLowerShowArea = 1.0 # 下部最小表示領域率
        self.minWidthShowArea = 0.7 # 横最小表示領域率
        self.minScale = 0.4 # 最小倍率
        self.maxScale = 1.0 # 最大倍率
        self.maxRotation = 30 # 最大回転角度[deg]
        self.maxGamma = 2.5  # 最大ガンマ補正値 最小は逆数
        self.maxSmoothSigma = 1.5 # 平滑化の最大σ値[pix]

        self.bgiList = []
        self.fgiList = []
        self.mskList = []
        
        # constant values
        self.__SMOOTH_KSIZE = (5,5) # 平滑化窓サイズ(固定)
        self.__MIN_SMOOTH_SIGMA = 0.2 #平滑化最小σ値
        

        
    def generateImageByParams( self, params ):
        
        """
        パラメータを指定して画像を生成する
        @param params 画像生成のパラメータを持つConfParamsクラス
        @return 合成された画像
        """
        
        # 背景画像の処理
        bgi = cv2.imread( params.bgi_filename, cv2.IMREAD_COLOR )
        h, w = bgi.shape[:2]
        bgi = bgi[params.bgi_y:params.bgi_y + params.bgi_size, params.bgi_x:params.bgi_x + params.bgi_size]
        if params.bgi_flip :
            bgi = cv2.flip( bgi, 1 )
        if params.bgi_only :
            bgi = it.resize_bysize( bgi, params.outputsize, params.outputsize )
            bgi = it.gamma_correction( bgi, params.gamma )
            bgi = cv2.GaussianBlur( bgi, self.__SMOOTH_KSIZE, params.smooth_sigma )
            return bgi
            
        # 前景画像の処理
        fgi = cv2.imread( params.fgi_filename, cv2.IMREAD_COLOR )
        msk = cv2.imread( params.msk_filename, cv2.IMREAD_GRAYSCALE)
        
        # Mask部分の画像だけにする(mgn分の縁は残す)
        mgn=2;
        h, w = msk.shape[:2]
        msk_rect = it.rect_of_mask(msk)
        lft = max(0, msk_rect[0]-mgn)
        rgt = min(w, msk_rect[0]+msk_rect[2]+mgn)
        top = max(0, msk_rect[1]-mgn)
        btm = min(h, msk_rect[1]+msk_rect[3]+mgn)
        fgi = fgi[top:btm, lft:rgt]
        msk = msk[top:btm, lft:rgt]

        # アスペクト比
        fgi = it.resize_byaspect( fgi, params.aspect )
        msk = it.resize_byaspect( msk, params.aspect )
        
        # 左右反転
        if params.fgi_flip:
            fgi = cv2.flip( fgi, 1 )
            msk = cv2.flip( msk, 1 )
            
        # 回転
        fgi = it.rotate_img( fgi, params.rot )
        msk = it.rotate_img( msk, params.rot )
        
        # 背景と画像サイズを合わせる
        brect = it.rect_of_mask( msk )
        bigger_size = max(brect[2:4])  # 最小正方形サイズ(=幅、高さの大きい方)
        scale = params.bgi_size / bigger_size
        scale = params.scale * scale
        fgi = it.resize_byfactor( fgi, scale )
        msk = it.resize_byfactor( msk, scale )
        
        # 合成
        result = it.chromakey(bgi, fgi, msk, params.pos )
        
        # 出力サイズにする
        result = it.resize_bysize( result, params.outputsize, params.outputsize)
        
        # ガンマ補正
        result = it.gamma_correction( result, params.gamma )
        
        # 平滑化
        result = cv2.GaussianBlur( result, self.__SMOOTH_KSIZE, params.smooth_sigma )

        #　終了
        return result
        

    def generateImageAtRandom( self, bgi_only=False ):
        
        """
        ランダムに画像を生成する．
        戻り値は生成された画像と，使用されたパラメータ．
        @param bgi_only 背景だけ生成(合成なし)フラグ
        @return 生成画像, 使用されたパラメータ
        """
        
        # 記録用 ConfigPrams 準備
        params = TrainImageGenerator.ConfParams()
        params.outputsize = self.size
        params.bgi_only = bgi_only
        
        """ -----------背景画像の処理------------- """
        # 背景画像作成
        # ランダムな正方形領域をトリミングする
        # １辺の最小サイズは self.size 最大サイズは min(h,w)
        idx = random.randrange( len(self.bgiList) )
        bgi = cv2.imread( self.bgiList[idx] )
        h, w = bgi.shape[:2]
        bgi_max_sq_size = min(h,w)
        bgi_size = random.randint(self.size, bgi_max_sq_size)
        x = random.randint(0,w-bgi_size)
        y = random.randint(0,h-bgi_size)
        bgi = bgi[y:y+bgi_size, x:x+bgi_size]
        
        # ここまでのランダム変数を記録
        params.bgi_filename = os.path.abspath(self.bgiList[idx])
        params.bgi_x = x
        params.bgi_y = y
        params.bgi_size = bgi_size
        
        # 背景画像を左右反転
        if random.randint(0,0) == 0:
            bgi = cv2.flip( bgi, 1 )
            params.bgi_flip = True
        else:
            params.bgi_flip = False

        # 背景のみ(合成なし)の場合
        if bgi_only :
            bgi = it.resize_bysize( bgi, self.size, self.size )
            log_maxGamma = np.log( self.maxGamma )
            gmm = np.exp( random.uniform( -log_maxGamma, log_maxGamma ) )
            bgi = it.gamma_correction(bgi, gmm)
            params.gamma = gmm;
            sigma = random.uniform( self.__MIN_SMOOTH_SIGMA, self.maxSmoothSigma)
            bgi = cv2.GaussianBlur( bgi, self.__SMOOTH_KSIZE, sigma)

            return bgi, params    # bgi_only = True ならここで関数終了
            
        """ ----------背景画像の処理終了----------- """

            
        """ -----ソース画像から背景との合成までの処理 ----"""
        
        # ソース画像(前面画像)の読み込み
        idx = random.randrange( len(self.fgiList) )
        fgi = cv2.imread( self.fgiList[idx], cv2.IMREAD_COLOR )
        msk = cv2.imread( self.mskList[idx], cv2.IMREAD_GRAYSCALE )
        params.fgi_filename = os.path.abspath(self.fgiList[idx])
        params.msk_filename = os.path.abspath(self.mskList[idx])
        
        # fgi & msk画像を必要なところまで小さくする
        mgn=2;
        h, w = msk.shape[:2]
        msk_rect = it.rect_of_mask(msk)
        lft = max(0, msk_rect[0]-mgn)
        rgt = min(w, msk_rect[0]+msk_rect[2]+mgn)
        top = max(0, msk_rect[1]-mgn)
        btm = min(h, msk_rect[1]+msk_rect[3]+mgn)
        fgi = fgi[top:btm, lft:rgt]
        msk = msk[top:btm, lft:rgt]

        # アスペクト比
        aspect = random.uniform( 1.0/self.maxAspect, self.maxAspect )
        fgi = it.resize_byaspect( fgi, aspect )
        msk = it.resize_byaspect( msk, aspect )
        params.aspect = aspect
        
        # 左右反転
        if random.randint(0,1) == 1:
            fgi = cv2.flip( fgi, 1 )
            msk = cv2.flip( msk, 1 )
            params.fgi_flip = True
        else:
            params.fgi_flip = False
        
        # 回転
        rot = random.uniform( -self.maxRotation, self.maxRotation )
        fgi = it.rotate_img( fgi, rot )
        msk = it.rotate_img( msk, rot )
        params.rot = rot
        
        # 加工したソース画像の Bouding rect の 最小正方サイズを取得
        brect = it.rect_of_mask( msk )
        msk_sq_size = max(brect[2:4])  # 最小正方形サイズ(=幅、高さの大きい方)
        
        # 背景の画像サイズに合わせて拡大縮小
        scale1 = bgi_size/msk_sq_size # 背景画像とのサイズ比
        scale2 = random.uniform( self.minScale, self.maxScale )
        scale = scale1 * scale2
        fgi = it.resize_byfactor( fgi, scale )
        msk = it.resize_byfactor( msk, scale )
        params.scale = scale2
        
        # 位置
        maxPosX = +0.5 + (0.5-self.minWidthShowArea) * msk.shape[1] / bgi.shape[1]
        minPosY = -0.5 - (0.5-self.minHeightLowerShowArea) * msk.shape[0] / bgi.shape[0]
        maxPosY = +0.5 + (0.5-self.minHeightUpperShowArea) * msk.shape[0] / bgi.shape[0]
        pos = ( random.uniform(-maxPosX,maxPosX), random.uniform(minPosY,maxPosY))
        params.pos = pos


        # 背景とのクロマキー合成
        result = it.chromakey( bgi, fgi, msk, pos )

        # 目的のサイズにする
        result = it.resize_bysize( result, self.size, self.size )
        
        # ガンマ補正
        log_maxGamma = np.log( self.maxGamma )
        gmm = np.exp( random.uniform( -log_maxGamma, log_maxGamma ) )
        result = it.gamma_correction( result, gmm )
        params.gamma = gmm
        
        # 平滑化
        sigma = random.uniform( self.__MIN_SMOOTH_SIGMA, self.maxSmoothSigma)
        result = cv2.GaussianBlur( result, self.__SMOOTH_KSIZE, sigma)
        params.smooth_sigma = sigma


        # 合成された前景画像の矩形を情報を計算する
        cmask = it.chromakey_mask( bgi, msk, pos)
        brect = it.rect_of_mask( cmask )
        x_ratio, y_ratio = self.size/bgi.shape[1], self.size/bgi.shape[0]
        rect =(
            brect[0]*x_ratio,
            brect[1]*y_ratio,
            brect[2]*x_ratio,
            brect[3]*y_ratio,
         )
        params.rect = rect


        # 生成画像とパラメータを返す
        return result, params

        """ generateImageAtRandom 終了 """
        
        

    def write_config_to_log( self, filename, mode='a' ):

        """
        ログファイルに設定を書き込みます
        @param filename ログファイル名
        """
        
        with open( filename, mode ) as file:
            file.write( '[Config]\n' )
            file.write( 'size={0}\n'.format(self.size)) # 出力画像サイズ
            file.write( 'maxAspect={0}\n'.format(self.maxAspect))   # 最大アスペクト比, 最小は逆数
            file.write( 'minHeightUpperShowArea={0}\n'.format(self.minHeightUpperShowArea)) # 上部最小表示領域率
            file.write( 'minHeightLowerShowArea={0}\n'.format(self.minHeightLowerShowArea)) # 下部最小表示領域率
            file.write( 'minWidthShowArea={0}\n'.format(self.minWidthShowArea)) # 横最小表示領域率
            file.write( 'minScale={0}\n'.format(self.minScale)) # 最小倍率
            file.write( 'maxScale={0}\n'.format(self.maxScale)) # 最大倍率
            file.write( 'maxRotation={0}\n'.format(self.maxRotation)) # 最大回転角度[deg]
            file.write( 'maxGamma={0}\n'.format(self.maxGamma))  # 最大ガンマ補正値 最小は逆数
            file.write( 'maxSmoothSigma={0}\n'.format(self.maxSmoothSigma)) # 平滑化の最大σ値[pix]



        
    class ConfParams:
        """ 画像生成に必要なパラメータを格納するクラス """
        
        def __init__(self):
            
            # 全般
            self.outputsize = 0  # 出力画像サイズ
            self.gamma = 1.0       # ガンマ補正値
            self.smooth_sigma = 1.5 # 平滑化の σ値[pix]
            self.bgi_only = False  # 背景のみ(前景と合成していない)フラグ

            # 背景関連
            self.bgi_filename = ""  # 背景ファイル名
            self.bgi_x = 0  # 背景 x 座標[pix]
            self.bgi_y = 0  # 背景 y 座標[pix]
            self.bgi_size = 0  # 背景サイズ[pix]
            self.bgi_flip = False  # 8.背景左右
            
            # 前景関連
            self.fgi_filename = ""  # 前景ファイル名
            self.msk_filename = "" # mskファイル名
            self.aspect = 1.0      # アスペクト比
            self.scale = 1.0       # 倍率
            self.rot = 0           # 回転角度[deg]
            self.fgi_flip = False  # 左右反転
            self.pos = (0.0, 0.0)  # 相対表示位置
            self.rect = (0.0, 0.0, 1.0, 1.0) # 絶対表示位置
            
        
        def write_to_log( self, idname, filename, mode='a' ):
            
            data = [idname]
            data.append(self.outputsize)
            data.append(self.gamma)
            data.append(self.smooth_sigma)
            data.append(self.bgi_only)
            data.append(self.bgi_filename)
            data.append(self.bgi_x)
            data.append(self.bgi_y)
            data.append(self.bgi_size)
            data.append(self.bgi_flip)
            data.append(self.fgi_filename)
            data.append(self.msk_filename)
            data.append(self.aspect)
            data.append(self.scale)
            data.append(self.rot)
            data.append(self.fgi_flip)
            data.append(self.pos[0])
            data.append(self.pos[1])
            data.append(self.rect[0])
            data.append(self.rect[1])
            data.append(self.rect[2])
            data.append(self.rect[3])
            
            with open( filename, mode, newline='' ) as file:
                writer = csv.writer(file)
                writer.writerow(data)
            
        
        
        @staticmethod
        def write_header_to_log( filename, mode='a' ):
            paramsMark = ['[ConfParams]']  # ConfParamsの開始マーク
            
            header = ['id','output_size[pix]', 'gamma', 'smooth_sigma[pix]',
                        'bgi_only', 'bgi_filename', 'bgi_x[pix]','bgi_y[pix]',
                        'bgi_size[pix]', 'bgi_flip', 'fgi_filename', 'msk_filename',
                        'aspect', 'scale', 'rotation', 'fgi_flip', 'pos_x', 'pos_y',
                        'rect_x', 'rect_y', 'rect_w', 'rect_h']
            with open( filename, mode, newline='' ) as file:
                writer = csv.writer(file)
                writer.writerow(paramsMark)
                writer.writerow(header)
                
                
        # logfileから ConfigPramsのデータを取得する
        @staticmethod
        def read_log( filename ):
            paramsMark = ['[ConfParams]']  # ConfParamsの開始マーク
            
            ret = []
            with open( filename, 'r') as f:
                reader = csv.reader(f)
                
                # ConfigParams まで移動
                for row in reader:
                    if row == paramsMark:
                        break
                    
                # データ取得(ヘッダ付)
                for row in reader:
                    ret.append(row)
            
            return ret
            
                
        
        
if __name__ == '__main__':
    
    # TrainImageGenerator クラスのチェックプログラム
    
    gen = TrainImageGenerator()
    gen.bgiList = ['test_bgi.jpg']
    gen.fgiList = ['test_fgi.jpg']
    gen.mskList = ['test_msk.png']

    """
    params = TrainImageGenerator.ConfParams()
    params.outputsize = 256
    params.bgi_filename = 'test_bgi.jpg'
    params.fgi_filename = 'test_fgi.jpg'
    params.msk_filename = 'test_msk.png'
    params.bgi_x = 0
    params.bgi_y = 0
    params.bgi_size = 256
    params.bgi_filp = False
    params.aspect = 1.0
    params.fgi_flip = False
    params.gamma = 1.0
    params.pos = (0.0,0.5)
    params.scale = 1.0
    img = gen.generateImageByParams(params, bgi_only=False)
    cv2.imshow('test image', img)
    """    
    
    logfile = 'testlog.csv'
    gen.write_config_to_log(logfile, mode='w')

    TrainImageGenerator.ConfParams.write_header_to_log(logfile)
    
    for i in range(3):
        
        # ランダムとそのパラメータで画像生成
        img1, params = gen.generateImageAtRandom(bgi_only=False)
        p1 = (int(params.rect[0]), int(params.rect[1]))
        p2 = (int(params.rect[0]+params.rect[2]-1), int(params.rect[1]+params.rect[3]-1))
        cv2.rectangle( img1, p1, p2, (0,0,255))
        
        cv2.imshow('image at random', img1)
        img2 = gen.generateImageByParams(params)
        cv2.rectangle( img2, p1, p2, (0,0,255))
        cv2.imshow('image by params', img2)
        
        params.write_to_log( 'img{0:08d}'.format(i),logfile)
        
        # 一致しているかチェック
        print( '一致:', np.allclose(img1, img2) )
        
        # ESC で終了
        if cv2.waitKey() == 27:
            break

    cv2.destroyAllWindows()
    
    
    # 最後に logファイルを読み込んで表示
    logdata = TrainImageGenerator.ConfParams.read_log( logfile )
    print(logdata)