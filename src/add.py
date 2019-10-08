import os
import SimpleITK as stk

path = '../build/testdata'
for name in os.listdir(path):
    data_nii = stk.ReadImage(os.path.join('../build/testresult/{}_true_data.nii'.format(name)))
    label_true_nii = stk.ReadImage(os.path.join('../build/testresult/{}_true_label.nii'.format(name)))
    label_test_nii = stk.ReadImage(os.path.join('../'))
