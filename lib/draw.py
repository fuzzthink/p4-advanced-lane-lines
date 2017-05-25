import cv2
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = .5
fontwd = 1
linetype = cv2.LINE_AA 


def rect(img, box, color=(0,255,0), thick=2):
    ''' box: ((x0,y0),(x1,y1))
    '''
    return cv2.rectangle(img, box[0], box[1], color, thick)

def textrow_ht_y0(fontscale=fontscale):
    ''' Returns correct ht for text image rows. Needed as VideoFileClip.fl_image
        fails to write corect video if result ht is odd
    '''
    ht = 18 + int(fontscale*20)
    y0 =  9 + int(fontscale*20)
    return (ht, y0)

def _color(msg):
    return (0,255,255) if msg.startswith('OK') else (255,0,0)

def with_btm_win(img, out_ht, offcenter, radm, fitmsg, detectmsg):
    ''' Returns stacked main image and bottom texts
    offcenter: meters off-center of lane
    radm:      radius in meters
    fitmsg:    fit msg
    detectmsg: detection msg
    ''' 
    img_ht, img_wd = img.shape[:2]
    absRadm = abs(radm)
    out = np.zeros((out_ht-img_ht, img_wd, 3)).astype(np.uint8)
    radtxt = 'Curve Radius: %dm %s' % (absRadm, 'Right' if radm < 0 else 'Left')
    postxt = 'Off center to the %s by %3.2fm' % ('left' if offcenter > 0 else 'right', offcenter)
    L, R = 0, 1
    lfit = 'Left  Fit: %s' % fitmsg[L]
    rfit = 'Right Fit: %s' % fitmsg[R]
    ldet = 'Left  Detect: %s' % detectmsg
    rdet = 'Right Detect: %s' % detectmsg
    x,y = 8, 20
    h = int(fontscale*45)
    cv2.putText(out,radtxt,(x,y),    font,fontscale,(255,255,255),fontwd,linetype)
    cv2.putText(out,postxt,(x,y+h*1),font,fontscale,(255,255,255),fontwd,linetype)
    cv2.putText(out,lfit,  (x,y+h*2),font,fontscale,_color(fitmsg[L]),fontwd,linetype)
    cv2.putText(out,rfit,  (x,y+h*3),font,fontscale,_color(fitmsg[R]),fontwd,linetype)
    cv2.putText(out,ldet,  (x,y+h*4),font,fontscale,(255,255,255),fontwd,linetype)
    cv2.putText(out,rdet,  (x,y+h*5),font,fontscale,(255,255,255),fontwd,linetype)
    return np.vstack((img, out))

def side_wins(_wins):
    ''' Returns a vertically stacked image of reduced versions of _wins.
    ''' 
    nwins = len(_wins)
    win_wd_ht = (420, 260)
    wins = [cv2.resize(_wins[i],win_wd_ht,interpolation=cv2.INTER_AREA) for i in range(nwins)]

    txtimg_ht = 34
    txtimgs = []
    txtimgs = [np.zeros((txtimg_ht, win_wd_ht[0], 3)).astype(np.uint8) for i in range(nwins)]
    txtpos = (8, txtimg_ht-13)
    txts = ['Edge filter', 'Yellow filter', 'Fit: thin=current frame, thick=avg used']
    for i,txtimg in enumerate(txtimgs):
        txtimg[:,:,None] = 255
        cv2.putText(txtimgs[i], txts[i], txtpos,font,fontscale,(0,0,0),fontwd,linetype)
    return np.vstack((txtimgs[0],wins[0], txtimgs[1],wins[1], txtimgs[2],wins[2]))

def with_debug_wins(img, wins, offcenter, radm, fitmsg, detectmsg):
    side = side_wins(wins)
    with_btm = with_btm_win(img, side.shape[0], offcenter, radm, fitmsg, detectmsg)
    return np.hstack((with_btm, side))
