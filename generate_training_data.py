# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:19:58 2017

@author: shingo

背景画像と前景&Mask画像から、ランダムにデータを生成するプログラム
"""

import ImageGenerator as IG
import datetime as dt
import os, sys, glob
import cv2
import imgtool as it

"""==========================

   主なグローバル変数
     
=========================="""

# 生成数画像枚数
GENERATE_CNT = 10000

# 背景のみ(合成を行わない)フラグ
BGI_ONLY = False

# ログファイル名
LOG_FILE = 'log.csv'

# 背景画像の設定
BGI_DIRS = ['/Volumes/Untitled/tsukuchare2017/TrainingData_DL/background/ImageCapture/20170708_141727/dir000000/',
            '/Volumes/Untitled/tsukuchare2017/TrainingData_DL/background/ImageCapture/20170708_141727/dir001000/',
            '/Volumes/Untitled/tsukuchare2017/TrainingData_DL/background/ImageCapture/20170708_143309/dir000000/',
            ]
BGI_EXT = 'png'  # 背景画像の拡張子

# 前景&マスク画像の設定
FGI_DIRS = ['/Volumes/Untitled/tsukuchare2017/TrainingData_DL/foreground/001/']
FGI_EXT = 'jpg' # 前景画像の拡張子
MSK_EXT = 'png' # Mask画像の拡張子
MSK_SUF = '_msk' # Mask画像ファイル名の接尾語

# 保存先
CNT_PER_DIR = 1000   #　1ディレクトリに保存する最大枚数
SAVE_RTDIR = '/Users/shingo/tmp2/'  # 画像保存先ルートディレクトリ
SAVE_EXT = 'png'                                 # 保存画像の拡張子(jpg, png, bmp, ...)


"""==========================

   main関数
     
=========================="""

# main関数
def main():
    
    """==========================
       保存ディレクトリの準備
    =========================="""
    
    # 保存先ルートディレクトリの確認
    if not os.path.exists(SAVE_RTDIR):
        print( '保存先ルートディレクトリ {0} が存在しません.'.format(SAVE_RTDIR))
        print( 'プログラムを終了します.')
        sys.exit(1)
    
    # 保存先ディレクトリ名の生成
    datetime_str = dt.datetime.now().strftime('%Y%m%d_%H%M%S')  # 現在の時刻の文字列 ex) 20170704_165001
    save_dir = SAVE_RTDIR + datetime_str + '/'
    print('保存先ディレクトリ:', save_dir)
    
    
    """==========================
       画像生成クラスの設定
    =========================="""
    ig = IG.TrainImageGenerator()
    ig.size = 256  # 出力画像サイズ
    ig.maxAspect = 2.0   # 最大アスペクト比, 最小は逆数
    ig.minHeightUpperShowArea = 0.3 # 上部最小表示領域率
    ig.minHeightLowerShowArea = 0.7 # 下部最小表示領域率
    ig.minWidthShowArea = 0.7 # 横最小表示領域率
    ig.minScale = 0.2 # 最小倍率
    ig.maxScale = 1.0 # 最大倍率
    ig.maxRotation = 30 # 最大回転角度[deg]
    ig.maxGamma = 2.5  # 最大ガンマ補正値 最小は逆数
    ig.maxSmoothSigma = 1.5 # 平滑化の最大σ値[pix]
    
    
    # 前景画像 & Mask画像のファイル名のリストを設定
    ig.fgiList, ig.mskList = get_fgi_and_msk_list(FGI_DIRS, FGI_EXT, MSK_SUF, MSK_EXT)
    
    # 背景画像のファイル名のリストを設定
    ig.bgiList = get_bgi_list(BGI_DIRS, BGI_EXT)
    
    
    """==========================
       実行確認 yes or no
    =========================="""
    print( '前景画像枚数:', len(ig.fgiList))
    print( '背景画像枚数:', len(ig.bgiList))
    print( '生成画像枚数:', GENERATE_CNT)
    if( len(ig.fgiList) == 0 or len(ig.bgiList) == 0 ):
        print( '生成画像するための画像が見つかりませんでした.')
        print( 'プログラムを終了します.')
        sys.exit(0)
    if sys.version_info.major == 3:
        res = input("実行しますか?[y/N]:").lower() #Python3
    else:
        res = input_raw("実行しますか?[y/N]:").lower() #Python2
    if not res in ['y', 'ye', 'yes']:
        print( 'プログラムを終了します.' )
        sys.exit(0)
    
    
    """==========================
       ここから画像生成開始
    =========================="""
    print( "\n画像生成の開始します.")
    
    # 保存ディレクトリ生成
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
        
    # ログファイル生成
    log_file = save_dir + LOG_FILE
    ig.write_config_to_log( log_file, mode='w')
    IG.TrainImageGenerator.ConfParams.write_header_to_log( log_file )
    
    # 生成枚数分ループ
    for i in range(GENERATE_CNT):
        
        # 画像生成
        img, params = ig.generateImageAtRandom(bgi_only=BGI_ONLY)
        
        # 画像のファイルパス生成
        sub_dir_num = int(i / CNT_PER_DIR) * int(CNT_PER_DIR)
        sub_dir = save_dir + 'dir{0:08d}/'.format(sub_dir_num)
        if not(os.path.exists(sub_dir)):
            os.makedirs(sub_dir)
        fn = sub_dir + 'image{0:08d}.{1}'.format(i, SAVE_EXT)
        
        # 画像の保存
        cv2.imwrite(fn, img)
        
        # ログファイル記録
        idname = 'image{0:08d}'.format(i)
        params.write_to_log( idname, log_file )
        
        #プログレスバー表示
        prg_str = it.get_progressbar_str(i+1, GENERATE_CNT, bufsize=200)
        if( i % 10 == 0 ):  # n回に1回表示
            sys.stdout.write('\r\033[K' + prg_str )
            sys.stdout.flush
        if i+1 == GENERATE_CNT :  # 最後はフラッシュせず改行
            sys.stdout.write('\r\033[K' + prg_str )
            sys.stdout.write('\n')
            
            
    print( "プログラムを終了します.")
    # main プログラム終了

    

# 指定ディレクトリから背景画像のファイル名のリストを取得する
def get_bgi_list( dirs, bgi_ext ):
    
    """
    指定ディレクトリから背景画像のファイル名のリストを取得する
    @param dirs 背景画像が保存されているディレクトリのリスト
    @param bgi_ext 背景画像の拡張子(jpg, png, など)
    @return 背景画像とファイル名リスト
    """
    
    assert len(bgi_ext) == 3, "Background image file extension must be 3 characters!"
    
    bgi_list = []
    for bgi_dir in dirs:
        
        if not os.path.exists(bgi_dir):
            print( "[WARNING]背景画像ディレクトリ {0} が存在しません.".format(bgi_dir))
    
        bgi_list.extend( glob.glob( bgi_dir + '*.' + bgi_ext))
    
    return bgi_list

# 指定ディレクトリから前景画像とマスク画像のファイル名のリストを取得する関数
def get_fgi_and_msk_list( dirs, fgi_ext, msk_suf, msk_ext ):

    """
    指定ディレクトリから前景画像とマスク画像のファイル名のリストを取得する関数
    @param dirs 背景画像とマスク画像が保存されているディレクトリのリスト
    @param fgi_ext 前景画像の拡張子(jpg, png, など)
    @param msk_suf マスク画像の接尾語 ( _msk, _roi, など)
    @param msk_ext マスク画像の拡張子(jpg, png, など)
    @return 背景画像とファイル名リスト
    """

    assert len(fgi_ext) == 3, "Foreground image file extension must be 3 characters!"
    assert len(msk_ext) == 3, "Mask image file extension must be 3 characters!"
    
    fgi_list = []
    msk_list = []
    for fgi_dir in dirs:
        
        if not os.path.exists(fgi_dir):
            print( "[WARNING]前景画像ディレクトリ {0} が存在しません.".format(fgi_dir))
        
        fgi_all_list = glob.glob( fgi_dir + '*.' + fgi_ext )
        for fgi_fn in fgi_all_list:
            msk_fn = fgi_fn[:-4] + msk_suf + '.' + msk_ext
            if os.path.exists(msk_fn):
                fgi_list.append(fgi_fn)
                msk_list.append(msk_fn)
                
    return fgi_list, msk_list


    
# main関数の実行
if __name__ == '__main__':
    main()
