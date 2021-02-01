import matplotlib
import numpy as np
import matplotlib.pyplot as plt
showid = 0

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)): overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def nms(output, nms_th):
    if len(output) == 0: return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1: bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

srslst = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260']
ctdat = np.load('/home/hugoycj/Database/Dataset/LUNA16/prepare_for_deeplung_py3/subset0/'+srslst[showid]+'_clean.npy', allow_pickle=True)
# ctlab = np.load('./CT/'+srslst[showid]+'_label.npy', allow_pickle=True)
result_path = "../detector/results/fpn3d/retrft960/val10/"

pbb = np.load(result_path+srslst[showid]+'_pbb.npy', allow_pickle=True)
lbb = np.load(result_path+srslst[showid]+'_lbb.npy', allow_pickle=True)
# print('pbb max:', pbb[:, 0].max())
# print('pbb mean:', pbb[:, 0].mean())
# print('pbb min:', pbb[:, 0].min())
# plt.hist(pbb[:, 0], bins=200, #range=(0,100),
#                 weights=None, cumulative=False, bottom=None,     
#                 histtype=u'bar', align=u'left', orientation=u'vertical',     
#                 rwidth=0.8, log=False, color=None, label=None, stacked=False) 
# plt.savefig("pbb_vis.png")
pbb = np.array(pbb[:2000, ])
pbb = np.array(pbb[pbb[:,0] > -1])
pbb = nms(pbb, 0.1)
print('Len(pbb) after nms', len(pbb))
# print pbb.shape, pbb
print('Detection Results according to confidence')
for idx in range(pbb.shape[0]):
    fig = plt.figure()
    z, x, y = int(pbb[idx,1]), int(pbb[idx,2]), int(pbb[idx,3])
#     print z,x,y
    try:
        dat0 = np.array(ctdat[0, z, :, :])
        dat0[max(0,x-10):min(dat0.shape[0],x+10), max(0,y-10)] = 255
        dat0[max(0,x-10):min(dat0.shape[0],x+10), min(dat0.shape[1],y+10)] = 255
        dat0[max(0,x-10), max(0,y-10):min(dat0.shape[1],y+10)] = 255
        dat0[min(dat0.shape[0],x+10), max(0,y-10):min(dat0.shape[1],y+10)] = 255
        plt.imsave(srslst[showid] + '_vis.png', dat0)
    except:
        continue
