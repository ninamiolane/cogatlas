"""
Pipeline analyzing statistical maps in Neurovault,
together with Cognitive Atlas graph.
"""

import json
import logging
import luigi
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request

MAIN_DIR = '/code/cogatlas/'
OUTPUT_DIR = os.path.join(MAIN_DIR, 'output')
DATA_DIR = os.path.join(MAIN_DIR, 'data')
REPORT_DIR = os.path.join(MAIN_DIR, 'report')

NEUROVAULT_URL = 'https://neurovault.org/api/images/'
COGNITIVE_ATLAS_URL = 'https://www.cognitiveatlas.org/api/v-alpha/task'

DEBUG = True

# TODO(nina): check consistent names of variables


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

        df_tasks.to_csv(self.output().path, sep=',')

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'task_metadata.csv')
        return luigi.LocalTarget(path)


class NeurovaultMetadata(luigi.Task):
    """
    Extract metadata on statistical maps from Neurovault.
    """
    # TODO(nina): parallelize
    def run(self):
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
                    # TODO(nina): this condition does not work
                    logging.debug('Found null')
                    break
                page += 100

                if DEBUG:
                    if page > 200:
                        break

            df_neurovault = pd.concat(dfs, ignore_index=True)
            logging.info('{:d} images metadata returned by the query'.format(
                len(df_neurovault)))
            logging.debug('Image metadata fields are: \n{}'.format(
                df_neurovault.keys()))

            df_neurovault.to_csv(self.output().path, sep=',')

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'neurovault_metadata.csv')
        return luigi.LocalTarget(path)


class DatasetMetadata(luigi.Task):
    def requires(self):
        return {'task_metadata': TaskMetadata(),
                'neurovault_metadata': NeurovaultMetadata()}

    def run(self):
        task_metadata_path = self.input()['task_metadata'].path
        neurovault_metadata_path = self.input()['neurovault_metadata'].path

        task_metadata = pd.read_csv(task_metadata_path, sep=',')
        neurovault_metadata = pd.read_csv(neurovault_metadata_path, sep=',')

        task_ids = task_metadata.id
        unique_task_ids = set(task_ids)
        n_task_ids = len(unique_task_ids)
        assert n_task_ids == len(task_ids)
        logging.info('Cognitive atlas has {:d} unique mental task IDs'.format(
            n_task_ids))

        labels = neurovault_metadata.cognitive_paradigm_cogatlas_id
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

        img_files = neurovault_metadata['file'].loc[
            mask_in_task_id].values
        img_files = np.expand_dims(img_files, axis=1)
        img_ids = neurovault_metadata['id'].loc[
            mask_in_task_id].values
        img_ids = np.expand_dims(img_ids, axis=1)
        img_task_ids = neurovault_metadata[
            'cognitive_paradigm_cogatlas_id'].loc[
                mask_in_task_id].values
        img_task_ids = np.expand_dims(img_task_ids, axis=1)

        assert len(img_files) == len(img_task_ids)

        csv_rows = np.hstack([img_files, img_ids, img_task_ids])
        df_dataset_metadata = pd.DataFrame(csv_rows)

        df_dataset_metadata.to_csv(
            self.output().path, sep=',', index=False, header=False)

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'dataset_metadata.csv')
        return luigi.LocalTarget(path)


class MakeDataset(luigi.Task):
    """
    Make dataset and store metadata in csv file where:
    - the first column is the path to the statistical maps nii files,
    - the second column is the ID of the mental task associated with the maps.
    """
    # TODO(nina): parallelize and check for data already there
    def requires(self):
        return {'dataset_metadata': DatasetMetadata()}

    def run(self):
        dataset_metadata_path = self.input()['dataset_metadata'].path
        dataset_metadata = pd.read_csv(dataset_metadata_path, sep=',')

        nii_files = dataset_metadata.iloc[:, 0]
        nii_ids = dataset_metadata.iloc[:, 1]
        nii_paths = np.array(
            [os.path.join(DATA_DIR, '{}.nii.gz'.format(nii_id))
             for nii_id in nii_ids])
        for nii_path, nii_file in zip(nii_paths, nii_files):
            urllib.request.urlretrieve(nii_file, nii_path)

        df_dataset = pd.concat([nii_files, nii_paths], axis=1)
        df_dataset.to_csv(self.output().path, sep=',')

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'dataset.csv')
        return luigi.LocalTarget(path)


class DatasetStatistics(luigi.Task):
    """
    Create report with statistics on the dataset.
    """
    # TODO(nina): Color barplot w.r.t. which Neurovault study the data belong
    def requires(self):
        return {'task_metadata': TaskMetadata(),
                'dataset_metadata': DatasetMetadata()}

    def run(self):
        task_metadata_path = self.input()['task_metadata'].path
        csv_path = self.input()['dataset_metadata'].path

        task_metadata = pd.read_csv(task_metadata_path, sep=',')
        dataset = np.genfromtxt(csv_path, delimiter=',', dtype='S')
        labels = dataset[:, 2].astype(str)

        unique, counts = np.unique(labels, return_counts=True)
        print('labels = {}'.format(labels))
        print('unique = {}'.format(unique))
        label_to_count = dict(zip(unique, counts))
        label_to_task_name = {
            task[1]['id']: task[1]['name']
            for task in task_metadata.iterrows()}
        print('dict = {}'.format(label_to_task_name))

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
