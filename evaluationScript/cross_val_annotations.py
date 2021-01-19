import csv
from glob import glob
import os

original_annots_fpath = '/home/hugoycj/Database/Dataset/LUNA16/CSVFILES/annotations.csv'
orig_csv_rdr = csv.reader(open(original_annots_fpath), delimiter=',')
all_original_annotations = None

## get all excluded annotations

# all_exclude_fpath = '/raid/shadab/prateek/dgx30/prateek/DeepLung/evaluationScript/annotations/annotations_excluded.csv'
# ex_csv_rdr = csv.reader(open(all_exclude_fpath), delimiter=',')
# all_excluded_annots = None
# for row in ex_csv_rdr:
#     if all_excluded_annots == None:
#         all_excluded_annots = [row[0]]  
#     else:
#         if row[0] not in all_excluded_annots:
#             all_excluded_annots.append(row[0])

# fold=0
for fold in range(10):

    subset_fpath = '/home/hugoycj/Database/Dataset/LUNA16/prepare_for_deeplung_py3/subset'+str(fold)
    subset_fnames = sorted(glob(os.path.join(subset_fpath, '*_clean.npy')))
    #extract seriesuid and get rid of full path and suffix
    subset_fnames = [os.path.basename(f).rsplit('_clean.npy')[0] for f in subset_fnames]
    # print(subset_fnames)
    # print(subset_fnames)
    #get the annotations for only fold
    fold_annotations_fname='annotations'+str(fold)+'.csv'
    fold_annotations=[]
    fold_annots_writer = csv.writer(open(os.path.join(subset_fpath, fold_annotations_fname), 'w'), delimiter='\n')

    fold_sids_fname='seriesids'+str(fold)+'.csv'
    fold_sids=[]
    fold_sids_writer = csv.writer(open(os.path.join(subset_fpath, fold_sids_fname),'w'), delimiter='\n')

    #read original annots reader & select seriesids which are common with 
    orig_csv_rdr = csv.reader(open(original_annots_fpath), delimiter=',')
    for row in orig_csv_rdr:
        curr_sid = row[0]
        if curr_sid in subset_fnames:
            fold_annotations.append((",").join(row))
            if curr_sid not in fold_sids:
                fold_sids.append(curr_sid)

    fold_sids_writer.writerow(fold_sids)
    print('Written '+os.path.join(subset_fpath, fold_sids_fname))
    fold_annots_writer.writerow(["seriesuid,coordX,coordY,coordZ,diameter_mm"])
    fold_annots_writer.writerow(fold_annotations)
    print('Written '+os.path.join(subset_fpath, fold_annotations_fname))