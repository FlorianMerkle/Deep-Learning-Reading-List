def scale_bbox(img, bb, target_height, target_width):
    y_factor = target_height / img.shape[1]
    x_factor = target_width / img.shape[2]
    x, y, width, height = bb
    return [x*x_factor/target_width, y*y_factor/target_height, width*x_factor/target_width, height*y_factor/target_height]


def x1y1wh_to_centerwh(bb, single_img=False):
    if single_img==True:
        bb[0] = bb[0]+bb[2]/2
        bb[1] = bb[1]+bb[3]/2
        return bb
    bb[:,0] = bb[:,0]+bb[:,2]/2
    bb[:,1] = bb[:,1]+bb[:,3]/2
    return bb

def centerwh_to_x1y1wh(bb, single_img=False):
    if single_img==True:
        bb[0] = bb[0]-bb[2]/2
        bb[1] = bb[1]-bb[3]/2
        return bb
    bb[:,0] = bb[:,0]-bb[:,2]/2
    bb[:,1] = bb[:,1]-bb[:,3]/2
    return bb


def chw_to_hwc(img):
    return img.permute(1,2,0)
