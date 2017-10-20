# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:40:27 2017

@author: Shingo Nakamura

深層学習用の学習画像生成のためのツール関数群
"""
import sys
import numpy as np
import cv2
import math
from datetime import datetime
import time

def chromakey( bgi, src, mask, pos = (0,0) ):
    """
    src に maskを掛けて bgi にクロマキー合成する
    @param bgi 背景画像
    @param src 前面画像
    @param mask src画像のmask画像
    @param pos bgiに対する相対位置座標(x,y)
    @return 合成された画像
    """

    assert mask.dtype == np.uint8 and mask.ndim == 2, 'maskが8U1Cではありません'
    assert src.shape[:2] == mask.shape[:2], 'srcとmaskのサイズが異なります'
    
    # 高さ、幅
    bgi_h, bgi_w = bgi.shape[:2]
    src_h, src_w = src.shape[:2]

    # maskの４隅座標
    lft = int((bgi_w - src_w )/2 + pos[0]*bgi_w)
    top = int((bgi_h - src_h )/2 + pos[1]*bgi_h)
    rgt = lft + src_w
    btm = top + src_h
    
    # 重なっている領域の算出 & 抽出 ( bgi側 )
    lap_lft = max( 0, lft )
    lap_top = max( 0, top )
    lap_rgt = min( bgi_w, rgt )
    lap_btm = min( bgi_h, btm )
    img = bgi.copy()
    lap_bgi = img[lap_top:lap_btm, lap_lft:lap_rgt, :]
    
    # 重なっている領域の算出 & 抽出( src側 )
    lap_lft2 = lap_lft-lft
    lap_top2 = lap_top-top
    lap_rgt2 = lap_lft2 + (lap_rgt-lap_lft)
    lap_btm2 = lap_top2 + (lap_btm-lap_top)
    lap_src = src[lap_top2:lap_btm2, lap_lft2:lap_rgt2,:]
    mask = mask[lap_top2:lap_btm2, lap_lft2:lap_rgt2] / 255.0
    mask = cv2.merge((mask,mask,mask))
    
    # 合成
    lap_bgi = lap_src*mask + (1.0-mask)*lap_bgi
    img[lap_top:lap_btm, lap_lft:lap_rgt, :] = lap_bgi

    # 終了
    return img;
    
    
def chromakey_mask( bgi, mask, pos = (0,0) ):
    
    '''
    クロマキー合成後のマスク画像を取得する
    @param bgi 背景画像
    @param mask src画像のmask画像
    @param pos bgiに対する相対位置座標(x,y)
    @return クロマキー後のマスク画像
    '''
    assert mask.dtype == np.uint8 and mask.ndim == 2, 'maskが8U1Cではありません'
    
    # 高さ、幅
    bgi_h, bgi_w = bgi.shape[:2]
    msk_h, msk_w = mask.shape[:2]

    # maskの４隅座標
    lft = int((bgi_w - msk_w )/2 + pos[0]*bgi_w)
    top = int((bgi_h - msk_h )/2 + pos[1]*bgi_h)
    rgt = lft + msk_w
    btm = top + msk_h
    
    # 重なっている領域の算出 & 抽出 ( bgi側 )
    lap_lft = max( 0, lft )
    lap_top = max( 0, top )
    lap_rgt = min( bgi_w, rgt )
    lap_btm = min( bgi_h, btm )
    
    # 重なっている領域の算出 & 抽出( mask側 )
    lap_lft2 = lap_lft-lft
    lap_top2 = lap_top-top
    lap_rgt2 = lap_lft2 + (lap_rgt-lap_lft)
    lap_btm2 = lap_top2 + (lap_btm-lap_top)
    mask = mask[lap_top2:lap_btm2, lap_lft2:lap_rgt2]

    # 最終 mask
    img = np.zeros( bgi.shape[:2], dtype='ubyte' )
    img[lap_top:lap_btm, lap_lft:lap_rgt] = mask
    return img

    
# ガンマ補正をする
def gamma_correction( src, gamma ):
    """
    ガンマ補正をする
    @param src 原画像
    @param gamma ガンマ補正値
    @return ガンマ補正された画像
    """

    # Lookup table の作成
    lookup_table = np.zeros((256, 1), dtype = 'uint8' )
    for i in range(256):
        lookup_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    
    # ガンマ補正の適用
    return cv2.LUT(src, lookup_table)
    
    
def trim_img( src, rect ):
    """
    画像を切り出す
    @param src ソース画像
    @param rect 矩形領域を表す(left, top, width, height)
    @return 切り出した画像
    """
    
    h, w = src.shape[:2]
    assert rect[0] >= 0, "rect is not inside of src"
    assert rect[1] >= 0, "rect is not inside of src"
    assert rect[0]+rect[2] <= w, "rect is not inside of src"
    assert rect[1]+rect[3] <= h, "rect is not inside of src"
    
    if( src.ndim == 2 ):
        img = src[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    else:
        img = src[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],:]

    return img
    

# 倍率により画像サイズを変更する関数.
def resize_byfactor( src, factor, interpolation=cv2.INTER_LINEAR ):
    """
    画像サイズを変更する関数.
    @param src ソース画像
    @param factor 倍率
    @param interpolation 補間方法(default=cv2.INTER_LINEAR)
    @return サイズ変更された画像
    """
    h, w = src.shape[:2]
    newImg = cv2.resize( src, None, fx=factor, fy=factor, interpolation=interpolation )
    return newImg

# サイズ指定により画像サイズを変更する関数.
def resize_bysize( src, width, height, interpolation=cv2.INTER_LINEAR ):
    """
    画像サイズを変更する関数
    @param src ソース画像
    @param width 変換後の幅
    @param height 変換後の高さ
    @param interpolation 補間方法(default=cv2.INTER_LINEAR)
    @return サイズ変更された画像
    """
    newImg = cv2.resize( src, (width, height), interpolation=interpolation )
    return newImg

# アスペクト比に従って画像サイズを変更する
def resize_byaspect( src, aspect, interpolation=cv2.INTER_LINEAR ):
    """
    アスペクト比に従って画像サイズを変更する
    @param src ソース画像
    @param aspect アスペクト比:縦を1とした時の横の比率
    @param interpolation 補間方法
    @return サイズ変更された画像
    """
    # 面積は変えずにアスペクト比になるように縦横の倍率設定
    fx = np.sqrt(aspect)
    fy = 1.0 / fx
    newImg = cv2.resize( src, None, fx=fx, fy=fy, interpolation=interpolation )
    return newImg

    
# 画像を回転させる
def rotate_img( src, deg, interpolation=cv2.INTER_LINEAR ):
    """
    画像を中心で回転する関数.
    @param src ソース画像
    @param deg 回転角度[degree]
    @param interpolation 補間方法
    @return 回転された画像
    """
    h, w = src.shape[:2]
    size = int(math.ceil(1.42 * max(w,h)))
    trn_mat = np.float32([[1,0,(size-w)/2],[0,1,(size-h)/2],[0,0,1]])
    rot_mat = cv2.getRotationMatrix2D( (size/2, size/2), deg, 1.0 )
    new_img = cv2.warpAffine( src, np.dot(rot_mat,trn_mat), (size,size), flags=interpolation)
    return new_img
    


#mask領域を囲むBoundingBoxを返す関数.
def rect_of_mask( mask ):
    """
    mask領域を囲むBoundingBoxを返す関数.
    @param mask 8bitのマスク画像
    @return (left, top, width, height)
    """
    assert mask.dtype == np.uint8 and mask.ndim == 2, 'mask is not uint8'

    # maskの高さと幅
    h, w = mask.shape[:2]

    # top位置を探す
    for top in range(h):
        if( np.sum(mask[top,:]>0) > 0 ):
            break
    else:
        return (0,0,0,0)  # 見つからなかった

    # bottom 位置を探す
    for btm in range( h-1, -1, -1):
        if( np.sum(mask[btm,:]>0) > 0):
            break

    # left 位置を探す
    for lft in range( w ):
        if( np.sum(mask[:,lft]>0) > 0):
            break

    # right 位置を探す
    for rgt in range( w-1, -1, -1 ):
        if( np.sum(mask[:,rgt]>0) > 0):
            break

    #高さ & 幅
    height = btm - top + 1
    width = rgt - lft + 1
    
    # 終了
    return ( lft, top, width, height )



# プログレスバーを表示する関数
def get_progressbar_str(progress, max_progress, maxlen = 50, bufsize=10):
    
    """
    プログレスバーの文字列を返す関数．
    @param progress 現在の進行具合
    @param max_progress 最終進行具合
    @param maxlen 最終進行具合までの長さ[文字]
    @param bufsize 残り時間を計算するためのデータ数
    """
    
    # 初回の処理
    if not hasattr(get_progressbar_str, 'time_q' ):
        get_progressbar_str.time_q = []
        
    # プログレスバー文字列作成
    ratio = progress / max_progress
    BAR_LEN = int(maxlen * ratio)
    bar_str = ('[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < maxlen else '') +
            ' ' * (maxlen - BAR_LEN) +
            '] %.1f%%' % (ratio * 100.))
    
    # 残り何秒かかかるか計算する
    get_progressbar_str.time_q.append( [progress, datetime.now()] )
    cnt = len(get_progressbar_str.time_q)

    # 十分な数データがなければ終了
    if cnt <= 1:
        return bar_str
        
    # q_capacity以内の個数に調整    
    if cnt > bufsize:
        get_progressbar_str.time_q.pop(0)  #先頭要素削除

    # 残り時間計算        
    d_cnt = get_progressbar_str.time_q[-1][0] - get_progressbar_str.time_q[0][0]
    durat = get_progressbar_str.time_q[-1][1] - get_progressbar_str.time_q[0][1]
    rem = (max_progress - progress) * durat / d_cnt
    hour = int(rem.total_seconds()/3600)
    minu = int(rem.total_seconds())//60%60
    secs = int(rem.total_seconds())%60 + 1
    if progress == max_progress: secs = 0
    time_str = ' (残り{0:02}h{1:02}m{2:02}s)'.format(hour, minu, secs)

    return bar_str + time_str

    

    
#各関数のテストプログラム
if __name__ == '__main__':
    
    
    deg = 10 # 回転角度
    aspect = 0.9 # アスペクト比
    pos = (0.1, 0.2) # 映り込み位置
    gamma = 0.5 # ガンマ補正
    scale = 0.5 # スケール

    # 元画像
    fgi_fn = 'test_fgi.jpg'
    msk_fn = 'test_msk.png'
    bgi_fn = 'test_bgi.jpg'
    fgi = cv2.imread( fgi_fn, cv2.IMREAD_COLOR )
    msk = cv2.imread( msk_fn, cv2.IMREAD_GRAYSCALE )
    bgi = cv2.imread( bgi_fn, cv2.IMREAD_COLOR )
    
    # 回転
    print( 'rotate_img関数' )
    fgi = rotate_img( fgi, deg )
    msk = rotate_img( msk, deg )
    
    # アスペクト比
    print( 'resize_byaspect関数' )
    fgi = resize_byaspect(fgi, aspect)
    msk = resize_byaspect(msk, aspect)
    
    # スケール
    print( 'resize_byfactor関数' )
    fgi = resize_byfactor(fgi, scale)
    msk = resize_byfactor(msk, scale)
    
    # トリミング
    print( 'rect_of_mask関数' )
    rect = rect_of_mask( msk )
    print( 'trim_img関数' )
    trim = trim_img(fgi, rect);
    cv2.imshow( 'trim', resize_byfactor(trim, 0.25) )
    
    # ガンマ補正
    print( 'gamma_correction関数' )
    gmm = gamma_correction( trim, gamma )
    cv2.imshow( 'gamma', resize_byfactor(gmm, 0.25) )

    # クロマキー合成
    print( 'chromakey関数' )
    chrmk = chromakey( bgi, fgi, msk, pos )
    chrmk = cv2.GaussianBlur( chrmk, (11,11), 13 )
    cv2.imshow( 'chromakey', resize_bysize(chrmk, 450, 800) )
    
    # 画像表示のための停止
    cv2.waitKey()
    cv2.destroyAllWindows()

    # プログレスバー
    print( 'プログレスバー' )
    max_p = 123
    for i in range(max_p):
        time.sleep(0.01)
        sys.stdout.write('\r\033[K' + get_progressbar_str(i+1, max_p))
    if i+1 != max_p :
        sys.stdout.flush()
    sys.stdout.write('\n')

