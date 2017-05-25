import sys, time, cv2, glob, os
import numpy as np
from lib.line_fit import process_image
from moviepy.editor import VideoFileClip
from lib.lane_detection import filtered_dict, warpedChannels


video_in = None
video_out = 'video-out.mp4'
menu = 'python run.py [in.mp4 [out.mp4]]  ## no arguments to run preprocessors on test images' 

if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg.endswith('help') or arg=='-h' or arg.startswith('--h'):
        print(menu)
        exit()
    video_in = arg
    if len(sys.argv) > 2:
        video_out = sys.argv[2]

if video_in:
    t = time.time()
    clip = VideoFileClip(video_in)

    clipped = clip.fl_image(process_image)
    clipped.write_videofile(video_out, audio=False)
    t2 = time.time()

    m, s = divmod(t2 - t, 60)
    print("%d:%02d to process video" % (m, s))
    if len(sys.argv) <= 1:
        print("You can run other files via> "+menu)

else:
    imgs_glob = glob.glob('test_images/*.jpg')
    images = [(cv2.imread(imgpath), imgpath) for imgpath in imgs_glob]

    basepath = 'output_images/test_images-'
    for dashpath in ['unwarped', 'warped', 'pipeline']:
        if not os.path.exists(basepath+dashpath):
            os.makedirs(basepath+dashpath)

    for img, imgpath in images:
        name = imgpath.split('test_images/')[1]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## run preprocessors on test images
        fdict = filtered_dict(image)
        yel = np.dstack((fdict['yelPos'], fdict['yelNeg'], fdict['yel']))
        wht = np.dstack((fdict['posEdge'], fdict['negEdge'], fdict['white']))
        unwarped = yel + wht

        warped_dict, yel, wht = warpedChannels(fdict)
        warped = yel + wht

        cv2.imwrite(basepath+'unwarped/'+name, unwarped*255)
        cv2.imwrite(basepath+'warped/'+name, warped*255)

        ## run process_image() on test images
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = process_image(image)
        # processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR) ## tried, errored
        cv2.imwrite(basepath+'pipeline/'+name, processed)
