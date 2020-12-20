from dataset_import import *
from oneclass import *
from ml import *
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser(
        description='Biometrics system based on palmprint images.')

    parser.add_argument(
        '-t', '--test', help='Run project in test mode using provided images.', action='store_true')
    parser.add_argument(
        '-b', '--build', help='Create necessary folders, extract ROI and serialize image data contained in provided folder.', nargs='?', const='palms_data')
    parser.add_argument(
        '-oc', '--oneclass', help='One Class SVM.', action='store_true')
    parser.add_argument(
        '-mc', '--multiclass', help='Multi Class SVM.', action='store_true')
    parser.add_argument(
        '-d', '--data', help='Data file for Multi Class SVM.', nargs='?', const='data_test_multiClass_roi')
    parser.add_argument(
        '-l', '--labels', help='Labels file for Multi Class SVM.', nargs='?', const='labels_test_multiClass_roi')
    parser.add_argument(
        '-p', '--predict',
        help='Run prediction on all images in provided folder.', nargs='?', const='test_data')

    args = parser.parse_args()

    if (args.test):
        if (args.build):
            if (args.oneclass):
                create_dataset_folders(
                    'palms_data', 'test_oneClass_roi', 'test_oneClass_data', 'oneClass')
                get_and_save_roi('palms_data', 'test_oneClass_roi')
                save_data_one_class_one_hand(
                    'test_oneClass_roi', 'test_oneClass_data')

            elif (args.multiclass):
                create_dataset_folders(
                    'palms_data', 'test_multiClass_roi', None, 'multiClass')
                get_and_save_roi('palms_data', 'test_multiClass_roi')
                save_add_labels('test_multiClass_roi')

        elif (args.predict):
            if (args.oneclass):
                subjects = import_data(args.predict)
                good_subjects = filter_and_pca_subjects(subjects, 8)
                classifiers = create_classifiers(good_subjects)

                test_dataset_oneClass(good_subjects, classifiers)

            elif (args.multiclass):
                test_dataset_multiClass(args.data, args.labels)

    else:
        if (args.build):
            if (args.oneclass):
                create_dataset_folders(
                    args.build, f'{args.build}_oneClass_roi', f'{args.build}_oneClass_data', 'oneClass')
                get_and_save_roi(args.build, f'{args.build}_oneClass_roi')
                save_data_one_class_one_hand(
                    f'{args.build}_oneClass_roi', f'{args.build}_oneClass_data')

            elif (args.multiclass):
                create_dataset_folders(
                    args.build, f'{args.build}_multiClass_roi', None, 'multiClass')
                get_and_save_roi(args.build, f'{args.build}_multiClass_roi')
                save_add_labels(f'{args.build}_multiClass_roi')

        elif (args.predict):
            if (args.oneclass):
                subjects = import_data(args.predict)
                good_subjects = filter_and_pca_subjects(subjects, 8)
                classifiers = create_classifiers(good_subjects)

                test_dataset_oneClass(good_subjects, classifiers)

            elif (args.multiclass):
                # test_dataset_multiClass(args.data, args.labels)
                get_cross_val_scores(args.data, args.labels)
                # find_best_parameters(args.data, args.labels)

        else:
            parser.print_help()
