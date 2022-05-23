import numpy as np
import cv2
from scipy.ndimage import rotate
import os

bg_dir = "background_images"
threat_dir = "threat_images"

threat_images_cv2 = [cv2.cvtColor(cv2.imread(os.path.join(threat_dir,filename)),cv2.COLOR_BGR2RGB) for filename in os.listdir(threat_dir)]
bg_images_cv2 = [cv2.cvtColor(cv2.imread(os.path.join(bg_dir,filename)),cv2.COLOR_BGR2RGB) for filename in os.listdir(bg_dir)]

alpha = 0.5 #''' Blending Ratio'''
angle = 45 # 45 degree clockwise 

submission_images = []

def masking(image, mask):

    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])

def segment(image):

    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold = 250
    image_mask = image_gray < threshold
    segmented_image = masking(image,image_mask)
    return segmented_image

def crop(image):

    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ARGS = np.argwhere(image_gray!= np.median(image_gray))
    x_length = int((max(ARGS[:,0])-min(ARGS[:,0]))/1.5)
    y_length = int((max(ARGS[:,1]) - min(ARGS[:,1]))/1.5)
    cx,cy = tuple(np.mean(ARGS,axis=0,dtype=int))

    ROI = image[cx-x_length:cx+x_length,cy-y_length:cy+y_length]
    return ROI

## lamda function to rotate the threat image

rotate_image = lambda image,angle=45: rotate(image,angle,reshape=True)

# final function to morph the threat image in the desired way

def parse(image_path,angle):
    '''
    image_path: path to the threat image to be parsed str
    angle: rotation angle (counter clockwise) should be > 10 in degrees int
    '''

    image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)

    segmented = segment(image=image)

    rotated = rotate_image(segmented,angle)

    cropped = crop(rotated)

    return cropped

# image to super impose morphed threat image onto the background image

def blend_image(background,element,alpha):
    '''
    background: Background Image ndarray
    element: Threat image ndarray
    Alpha: float between [0.0 1.0]
    
    '''
    bx,by = background.shape[:2]
    ex,ey = element.shape[:2]
    divisor = max(ex/bx,ey/by)+2
    # if divisor<2.5:
    #     divisor = 5 - divisor
    ex,ey = int(ex//divisor),int(ey//divisor)

    element = cv2.resize(element,(ey,ex),interpolation=cv2.INTER_AREA)
    mask = (element!=0)*alpha+(element==0)*1
    cover = element*(1-alpha)

    BACKGROUND = cv2.GaussianBlur(cv2.cvtColor(background,cv2.COLOR_RGB2GRAY),ksize=(9,9),sigmaX=45)
    MASK = np.uint8((BACKGROUND!=255)*255)
    CENTER = np.mean(np.argwhere(MASK==255),axis=0)
    (cnts,_) = cv2.findContours(MASK,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    CEN_X,CEN_Y = int(CENTER[0]),int(CENTER[1])
    MAX_cnt = max(cnts,key=cv2.contourArea)
    MAX_CNT = MAX_cnt.reshape(-1,2)
    MEAN = np.mean(MAX_CNT,axis=0,dtype=int)
    MIN = int(np.sqrt(min(abs(MAX_CNT-MEAN)[:,0]**2+abs(MAX_CNT-MEAN)[:,1]**2)))
    cx,cy = np.random.randint(CEN_X-MIN,CEN_X+MIN-ex),np.random.randint(CEN_Y-MIN,CEN_Y+MIN-ey)

    background[cx:cx+ex,cy:cy+ey] = background[cx:cx+ex,cy:cy+ey]*mask
    background[cx:cx+ex,cy:cy+ey] += np.uint8(cover)
    return background
        

        
res_images_cv2 = [parse(os.path.join(threat_dir,PATH),angle=angle) for PATH in os.listdir(threat_dir)]

for i in range(len(bg_images_cv2)):
    image = bg_images_cv2[i]

    for j in range(len(res_images_cv2)):
        image = blend_image(background=image.copy(),element=res_images_cv2[j].copy(),alpha=alpha)
    
    submission_images.append(image)
    cv2.imwrite(f"submission_dir/submission_image{i}.jpg",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

