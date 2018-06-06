"""
Pipeline analyzing statistical maps in Neurovault,
together with Cognitive Atlas graph.
"""

import csv
import json
import logging
import luigi
import os
import pylab as plt

import numpy as np
import pandas as pd
import pickle
import urllib.request

MAIN_DIR = '/Users/nina/code/cogatlas/'
OUTPUT_DIR = os.path.join(MAIN_DIR, 'output')
DATA_DIR = os.path.join(MAIN_DIR, 'data')
REPORT_DIR = os.path.join(MAIN_DIR, 'report')

NEUROVAULT_URL = 'https://neurovault.org/api/images/'
COGNITIVE_ATLAS_URL = 'https://www.cognitiveatlas.org/api/v-alpha/task'

DEBUG = True


class TaskMetadata(luigi.Task):
    """
    Extract metadata on mental tasks from the Cognitive Atlas.
    """

    def run(self):
        with urllib.request.urlopen(COGNITIVE_ATLAS_URL) as url:
            data = json.loads(url.read().decode())
            df_tasks = pd.io.json.json_normalize(data)

        logging.info('{:d} tasks metadata returned by the query'.format(
            len(df_tasks)))
        logging.debug('Tasks fields are: \n{}'.format(
            df_tasks.keys()))

        with open(self.output().path, 'wb') as pickle_file:
                pickle.dump(df_tasks, pickle_file)

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'task_metadata.pkl')
        return luigi.LocalTarget(path)


class ImageMetadata(luigi.Task):
    """
    Extract metadata on statistical maps from Neurovault.
    """
    # TODO(nina): parallelize
    def requires(self):
        return {'task_metadata': TaskMetadata()}

    def run(self):
        task_metadata_path = self.input()['task_metadata'].path
        with open(task_metadata_path, 'rb') as pickle_file:
            task_metadata = pickle.load(pickle_file)

        dfs = []

        page = 100
        while True:
            logging.debug('Reading Neurovault page #{:d}'.format(page))
            page_url = NEUROVAULT_URL + '?offset=%d' % page

            with urllib.request.urlopen(page_url) as url:
                data = json.loads(url.read().decode())
                page_results = pd.io.json.json_normalize(data['results'])
                dfs.append(page_results)
                if data['next'] is 'null':
                    print('Found null')
                    break
                page += 100

                if DEBUG:
                    if page > 35000:
                        break

            img_metadata = pd.concat(dfs, ignore_index=True)
            logging.info('{:d} images metadata returned by the query'.format(
                len(img_metadata)))
            logging.debug('Image metadata fields are: \n{}'.format(
                img_metadata.keys()))

        task_ids = task_metadata.id
        unique_task_ids = set(task_ids)
        n_task_ids = len(unique_task_ids)
        assert n_task_ids == len(task_ids)
        logging.info('Cognitive atlas has {:d} unique mental task IDs'.format(
            n_task_ids))

        labels = img_metadata.cognitive_paradigm_cogatlas_id
        n_imgs = len(labels)
        # Bring the NaN to None
        labels = labels.where((pd.notnull(labels)), None)

        mask_not_none = ~np.equal(labels, None)
        not_none_labels = labels[mask_not_none]
        n_imgs_with_id = len(not_none_labels)
        unique_labels = set(not_none_labels)

        logging.info('{:d}/{:d} images have paradigm ID, '
                     'taken among {:d} unique IDs.'.format(
                         n_imgs_with_id, n_imgs, len(unique_labels)))

        n_intersection = len(unique_task_ids & unique_labels)

        mask_in_task_id = np.isin(
            np.array(labels), list(unique_task_ids))
        logging.info('{:d}/{:d} images have paradigm ID corresponding'
                     'to a mental task ID in the Cognitive Atlas, '
                     ' which corresponds to {:d} unique IDs available '
                     'in Cognitive Atlas'.format(
                         sum(mask_in_task_id), n_imgs, n_intersection))

        img_files = img_metadata['file'].loc[
            mask_in_task_id].values
        img_files = np.expand_dims(img_files, axis=1)
        img_ids = img_metadata['id'].loc[
            mask_in_task_id].values
        img_ids = np.expand_dims(img_ids, axis=1)
        img_task_ids = img_metadata['cognitive_paradigm_cogatlas_id'].loc[
            mask_in_task_id].values
        img_task_ids = np.expand_dims(img_task_ids, axis=1)

        assert len(img_files) == len(img_task_ids)

        csv_rows = np.hstack([img_files, img_ids, img_task_ids])
        csv_path = self.output().path
        with open(csv_path, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(csv_rows)

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'img_metadata.csv')
        return luigi.LocalTarget(path)


class MakeDataset(luigi.Task):
    """
    Make dataset and store metadata in csv file where:
    - the first column is the path to the statistical maps nii files,
    - the second column is the ID of the mental task associated with the maps.
    """
    # TODO(nina): parallelize and check for data already there
    def requires(self):
        return {'img_metadata': ImageMetadata()}

    def run(self):
        img_metadata_path = self.input()['img_metadata'].path

        dataset_path = self.output().path
        with open(img_metadata_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            with open(dataset_path, 'w') as csv_file_to_write:
                writer = csv.writer(csv_file_to_write, delimiter=',')
                for row in reader:
                    nii_file = row[0]
                    nii_id = row[1]

                    nii_path = os.path.join(
                        DATA_DIR, '{}.nii.gz'.format(nii_id))
                    urllib.request.urlretrieve(nii_file, nii_path)
                    row_to_write = [nii_path, row[2]]

                    writer.writerow(row_to_write)

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'dataset.csv')
        return luigi.LocalTarget(path)


class DatasetStatistics(luigi.Task):
    """
    Create report with statistics on the dataset.
    """
    def requires(self):
        return {'task_metadata': TaskMetadata(),
                'img_metadata': ImageMetadata()}

    def run(self):
        task_metadata_path = self.input()['task_metadata'].path
        with open(task_metadata_path, 'rb') as pickle_file:
            task_metadata = pickle.load(pickle_file)

        csv_path = self.input()['img_metadata'].path
        dataset = np.genfromtxt(csv_path, delimiter=',', dtype='S')
        labels = dataset[:, 2].astype(str)

        unique, counts = np.unique(labels, return_counts=True)
        label_to_count = dict(zip(unique, counts))
        label_to_task_name = {
            task[1]['id']: task[1]['name']
            for task in task_metadata.iterrows()}

        task_name_to_count = {
                label_to_task_name[label]: label_to_count[label]
                for label in unique}

        fig, ax = plt.subplots(figsize=(23, 35))
        y_pos = np.arange(len(task_name_to_count))
        counts = np.fromiter(task_name_to_count.values(), dtype=int)
        ax.barh(y_pos, counts)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(task_name_to_count)
        ax.invert_yaxis()  # labels read top-to-bottom

        output_path = self.output().path
        fig.savefig(output_path, bbox_inches='tight')

    def output(self):
        basename = 'labels_distribution.png'
        path = os.path.join(REPORT_DIR, basename)
        return luigi.LocalTarget(path)


class RunAll(luigi.Task):
    """Run tasks of interest."""

    def requires(self):
        return DatasetStatistics()

    def run(self):
        pass

    def output(self):
        return luigi.LocalTarget('dummy')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    all_dir_names = [OUTPUT_DIR, DATA_DIR, REPORT_DIR]
    for dir_name in all_dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        assert os.path.isdir(dir_name)
    luigi.run()
