from dataset_import import *
from oneclass import *
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser(
        description='Biometrics system based on palmprint images.')

    group = parser.add_argument_group('Test')
    group.add_argument(
        '-t', '--test', help='Run project example using provided images.', action='store_true')

    group = parser.add_argument_group('Real')
    parser.add_argument(
        '-b', '--build', help='Create necessary folders, extract ROI and serialize image data contained in provided folder name.')
    parser.add_argument('-p', '--predict', help='Run prediction on all images in provided folder')

    args = parser.parse_args()

    if (args.test):
        if(args.build):
            create_dataset_folders(
            args.build, 'test_dataset', 'test_oneHand_data')
            get_and_save_roi(args.build, 'test_dataset')
            save_data_one_class_one_hand('test_dataset', 'test_oneHand_data')
        else:
            subjects = import_data(data_filename)
            good_subjects = filter_and_pca_subjects(subjects, 8)
            classifiers = create_classifiers(good_subjects)

            test_dataset(good_subjects, classifiers)

    elif (args.build):
        create_dataset_folders(
            args.build, 'args_dataset', 'args_oneHand_data')
        get_and_save_roi(args.build, 'args_dataset')
        save_data_one_class_one_hand('args_dataset', 'args_oneHand_data')
    
    elif (args.predict):
        subjects = import_data(args.predict)
        good_subjects = filter_and_pca_subjects(subjects, 8)
        classifiers = create_classifiers(good_subjects)

        test_dataset(good_subjects, classifiers)

    else:
        parser.print_help()
